from typing import Any, Callable, Dict, Tuple, Dict, Optional, Sequence, Type
import re
from functools import partial

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState

from langgraph_swarm.util import add_agent_name_to_messages, create_swarm_agent_as_tool

from .types import Agent, HandoffsState

__CTX_VARS_NAME__ = "context_variables"

def _generate_snake_case_name(agent_name: str):
    return re.sub(r"[^0-9a-z_]", "_", agent_name.lower())

def create_swarm_handoff(agent: Agent | str, func_name: str|None = None, description: str|None = None):
    name = agent if isinstance(agent, str) else agent.name
    # refactoring name to be a function name
    func_name = func_name or f"transfer_to_{_generate_snake_case_name(name)}"
    description = description or f"Call this function if a user is asking about a topic that should be handled by {name}."
    return create_swarm_agent_as_tool(name, func_name, description, return_direct=True)

def create_swarm_backlink(agent: Agent | str, func_name: str|None = None, description: str|None = None):
    name = agent if isinstance(agent, str) else agent.name
    # refactoring name to be a function name
    func_name = func_name or f"transfer_to_{_generate_snake_case_name(name)}"
    description = description or f"Call this function if a user is asking about a topic that is not handled by the current agent."
    return create_swarm_agent_as_tool(name, func_name, description, return_direct=True)

def wrap_swarm_tool(swarm_tool: Callable | BaseTool, name: str | None = None) -> BaseTool:
    if callable(swarm_tool):
        raise ValueError("not supported callable as a tool yet")
        swarm_tool = tool(name or swarm_tool.__name__, swarm_tool, infer_schema=True)
    if not isinstance(swarm_tool, BaseTool):
        raise ValueError(f"swarm_tool must be a callable or a BaseTool, got {type(swarm_tool)}")
    
    from langchain_core.tools.base import _get_all_basemodel_annotations, _is_injected_arg_type
    from langgraph.prebuilt.tool_node import InjectedState, _is_injection
    # following langgraph's injection to define the tool's state variables
    # TODO: more friendly way to do it (support callable, and with the variable name `context_variables`)?
    for name_, type_ in _get_all_basemodel_annotations(swarm_tool.get_input_schema()).items():
        if _is_injection(type_, InjectedState):
            # TODO: 
            pass

    return swarm_tool

def _update_context_variables(original_vars: Dict[str, Any], new_vars: Dict[str, Any]) -> Dict[str, Any]:
    return {**original_vars, **new_vars}

def _merge_context_variables(state: dict) -> dict:
    new_state = state.copy()
    context_variables = new_state.pop("context_variables", {})
    return {**new_state, **context_variables}

class HandoffAgentState(AgentState):
    context_variables: Dict[str, Any] = {}

def create_swarm_agent_and_handoffs(llm, agent: Agent, backlink_agent: Agent|None = None, agent_map={}):
    
    tools = agent.functions.copy()
    # process tools 

    # process handoffs
    tools_to_handoffs: Dict[str, str] = {}
    for handoff in agent.handoffs:
        func_name = f"transfer_to_{_generate_snake_case_name(handoff.name)}"
        tools.append(create_swarm_handoff(handoff, func_name=func_name))
        tools_to_handoffs[func_name] = handoff.name
    if backlink_agent:
        func_name = f"transfer_to_{_generate_snake_case_name(backlink_agent.name)}"
        tools.append(create_swarm_backlink(backlink_agent, func_name=func_name))
        tools_to_handoffs[func_name] = backlink_agent.name
    
    instructions = agent.instructions
    if isinstance(instructions, Runnable):
        # TODO: make it better?
        # runnable cannot see context_variables, so we need to merge them into state
        # or: only state.messages + **context_variables
        instructions = _merge_context_variables | instructions

    lc_agent = create_react_agent(
        model=llm,
        tools=tools,
        state_schema=HandoffAgentState,
        state_modifier=instructions,
        debug=False,
    )
    
    # TODO: add support handoffs from multiple agents? which means it's a graph instead of hierarchy
    #   1. There would be more than one backlink, it would be confused to current agent
    #      due to langgraph not supporting dynamic tools (likely we can set available tools dynamically from prompts)
    #   2. There would be handoff circular references with bad cases (checking for cycles)
    # For now, we raise an error if there are handoffs from multiple agents, 
    #   maybe it works to allow a graph (but not fully tested yet)?
    if agent.name in agent_map:
        raise ValueError(
            f"Agent name {agent.name} already exists in agent_map"
            f". We don't support an agent being linked to multiple handoffs yet."
        )
    else:
        agent_map[agent.name] = (agent, lc_agent, tools_to_handoffs)
    
    for handoff in agent.handoffs:
        create_swarm_agent_and_handoffs(llm, handoff, backlink_agent=agent if agent.backlink else None, agent_map=agent_map)
    
    return lc_agent

def create_swarm_workflow(
    llm,
    starting_agent: Agent,
    state_scheme: Type[HandoffsState] = HandoffsState,
    with_user_agent: bool = False,
    print_messages: None | Callable = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    interrupt_before: Optional[Sequence[str]] = None,
    interrupt_after: Optional[Sequence[str]] = None,
    debug: bool = False,
    # hold all created agents with name and lc_agent
    agent_map: Dict[str, Tuple[Agent, CompiledGraph]] | None = None
) -> CompiledGraph:
    
    agent = starting_agent
    if agent_map is None:
        agent_map = {}
    
    create_swarm_agent_and_handoffs(llm, agent, backlink_agent=None, agent_map=agent_map)
    
    # Define a new graph
    workflow = StateGraph(state_scheme)
    
    def human_agent_node(state: HandoffsState):
        user_input = input("User: ")
        if user_input.strip() == "/end":
            return {"user_end": True}
        return {"messages": state.get("messages", []) + [HumanMessage(content=user_input)]}

    def nop_agent_node(state: HandoffsState):
        return state
    
    start_node = human_agent_node if with_user_agent else nop_agent_node
    
    workflow.add_node("agent_router", start_node)
    workflow.set_entry_point("agent_router")

    # init branchs with an end node
    branchs = {'end': END}
    for name, (agent, lc_agent, tools_to_handoffs) in agent_map.items():
        def _pre_process(state: HandoffsState, agent_name: str):
            # mark the messages before this turn if not from an agent
            messages = state["messages"]
            unknown_messages = []
            # check the messages in this turn
            for _msg in reversed(messages):
                if isinstance(_msg, BaseMessage) and "agent_name" not in _msg.additional_kwargs:
                    unknown_messages.append(_msg)
            add_agent_name_to_messages("__user__", unknown_messages)
            # to `react` AgentState
            return {"messages": state["messages"], __CTX_VARS_NAME__: state.get(__CTX_VARS_NAME__, {})}
        
        def _post_process(state: HandoffAgentState, agent_name: str, tools_to_handoffs: Dict[str, str]):
            # from `react` AgentState
            messages = state["messages"]
            this_turn_messages = []
            # check the messages in this turn
            for _msg in reversed(messages):
                if not isinstance(_msg, (ToolMessage, AIMessage)) or "agent_name" in _msg.additional_kwargs:
                    break
                this_turn_messages.append(_msg)

            this_turn_messages.reverse()
            add_agent_name_to_messages(agent_name, this_turn_messages)

            context_vars = state.get(__CTX_VARS_NAME__, {})
            for _msg in this_turn_messages:
                # handle the context variables for ToolMessage
                # following the swarm's implementation, update the context variables back to state's context_variables
                # add the context variables into ToolMessage.artifact (tool.response_format=`content_and_artifact`)
                if isinstance(_msg, ToolMessage):
                    if _msg.artifact and isinstance(_msg.artifact, dict) and __CTX_VARS_NAME__ in _msg.artifact:
                        context_vars = _update_context_variables(context_vars, _msg.artifact[__CTX_VARS_NAME__])
            if print_messages:
                print_messages(this_turn_messages)

            next_agent = None
            # check the tool messages only in this turn
            for _msg in reversed(messages):
                if not isinstance(_msg, ToolMessage):
                    break
                if _msg.name in tools_to_handoffs:
                    # use the latest agent as the next agent
                    next_agent = tools_to_handoffs[_msg.name]
                    # TODO: mark handoff (tool) message
                    #   we can use this to filter out the messages and combine the history in the future
                    _msg.additional_kwargs["handoff_msg"] = f"transferred to {next_agent} from {agent_name}"
                    break

            if debug and next_agent:
                print("==> handoff to: ", next_agent)
            return {
                "messages": messages, 
                "agent_name": next_agent if next_agent else agent_name, 
                "handoff": next_agent is not None, 
                __CTX_VARS_NAME__: context_vars
            }
        
        # chained functions with lc_agent
        chain = partial(_pre_process, agent_name=agent.name) | lc_agent | partial(_post_process, agent_name=agent.name, tools_to_handoffs=tools_to_handoffs.copy())
        
        node_name = _generate_snake_case_name(agent.name)
        # workflow.add_node(node_name, partial(call_agent_node, agent_name=name, lc_agent=lc_agent))
        workflow.add_node(node_name, chain)
        workflow.add_edge(node_name, "agent_router")
        branchs[agent.name] = node_name

    def handoff_or_end(state: HandoffsState):
        # if handoffs, then we call the next agent
        handoff = state.get("handoff", False)
        if handoff:
            return state["agent_name"]
        elif with_user_agent and not state.get("user_end", False) :
            # continue next loop with user input, with the current agent
            if debug:
                print("==> continue with current agent: ", state["agent_name"])
            return state["agent_name"]
        else:
            return "end"
        
    workflow.add_conditional_edges(
        "agent_router",
        handoff_or_end,
        branchs,
    )
    
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )