from typing import Callable, Dict, Tuple, Dict, Optional, Sequence, Type, TypedDict
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
from langchain_core.tools import tool

from .agent_v03 import AgentState, _create_swarm_agent, __CTX_VARS_NAME__
from .types import Agent, HandoffsState
from .util import add_agent_name_to_messages, create_swarm_agent_as_tool

def _generate_snake_case_name(agent_name: str):
    return re.sub(r"[^0-9a-z_]", "_", agent_name.lower())

def create_swarm_handoff(agent: Agent | str, func_name: str|None = None, description: str|None = None):
    name = agent if isinstance(agent, str) else agent.name
    # refactoring name to be a function name
    func_name = func_name or f"transfer_to_{_generate_snake_case_name(name)}"
    description = description or f"Call this function if a user is asking about a topic that should be handled by {name}."
    return create_swarm_agent_as_tool(name, func_name, description)

def create_swarm_backlink(agent: Agent | str, func_name: str|None = None, description: str|None = None):
    name = agent if isinstance(agent, str) else agent.name
    # refactoring name to be a function name
    func_name = func_name or f"transfer_to_{_generate_snake_case_name(name)}"
    description = description or f"Call this function if a user is asking about a topic that is not handled by the current agent."
    return create_swarm_agent_as_tool(name, func_name, description)

def _merge_context_variables(state: dict) -> dict:
    new_state = state.copy()
    context_variables = new_state.pop("context_variables", {})
    return {**new_state, **context_variables}

def create_swarm_agent_and_handoffs(llm, agent: Agent, backlink_agent: Agent|None = None, agent_map={}):
    
    tools = agent.functions.copy()
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
    lc_agent = _create_swarm_agent(
        model=llm,
        tools=tools,
        agent_name=agent.name,
        handoff_map=tools_to_handoffs,
        state_modifier=instructions,
        debug=False
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
    agent_map: Dict[str, Tuple[Agent, CompiledGraph, Dict[str, str]]] | None = None
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
    for name, (agent, lc_agent, _) in agent_map.items():
        def _pre_process(state: HandoffsState, agent_name: str):
            # mark the messages before this turn if not from an agent
            messages = state["messages"]
            unknown_messages = []
            # check the messages in this turn
            for _msg in reversed(messages):
                if isinstance(_msg, BaseMessage) and "agent_name" not in _msg.additional_kwargs:
                    unknown_messages.append(_msg)
                else:
                    break
            add_agent_name_to_messages("__user__", unknown_messages)
            return {"messages": state["messages"], 
                    "next_agent": None, 
                    "agent_name": agent_name, 
                    __CTX_VARS_NAME__: state.get(__CTX_VARS_NAME__, {})
                }
        
        def _post_process(state: AgentState, agent_name: str):
            messages = state["messages"]
            this_turn_messages = []
            # check the messages in this turn
            for _msg in reversed(messages):
                if isinstance(_msg, (ToolMessage, AIMessage)) and "agent_name" not in _msg.additional_kwargs:
                    this_turn_messages.append(_msg)
            this_turn_messages.reverse()
            add_agent_name_to_messages(agent_name, this_turn_messages)
            if print_messages:
                print_messages(this_turn_messages)
            next_agent = state.get("next_agent", None)
            if next_agent in ["__end__"]:
                # post_process here to handle `__end__` from `react_agent`
                next_agent = None
            if debug and next_agent:
                print("==> handoff to: ", next_agent)
            return {
                "messages": messages, 
                "agent_name": next_agent if next_agent else state["agent_name"], 
                "handoff": next_agent is not None, 
                __CTX_VARS_NAME__: state.get(__CTX_VARS_NAME__, {})
            }
        chain = partial(_pre_process, agent_name=agent.name) | lc_agent | partial(_post_process, agent_name=agent.name)
        
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