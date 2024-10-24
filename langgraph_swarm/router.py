from typing import Annotated, Any, Callable, Dict, Generator, Iterator, List, Literal, Tuple, Dict, Optional, Sequence, Type, TypedDict
import re
from functools import partial

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, ToolCall, SystemMessage
from langchain_core.tools import tool, BaseTool
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.tools import InjectedToolArg

from langgraph_swarm.util import add_agent_name_to_messages, create_default_llm, default_print_messages

from .types import Agent, Response

__CTX_VARS_NAME__ = "context_variables"

# TODO: make this user configurable from `create_handoff`, 
#   and resolve the agent name even it is a f-string
AGENT_RESPONSE_PREFIX = "transferred to "

def _generate_snake_case_name(agent_name: str):
    return re.sub(r"[^0-9a-z_]", "_", agent_name.lower())
# _tool_with_injected_state(state: Annotated[dict, InjectedState]):

def create_swarm_agent_as_tool(
        agent: Agent | str, 
        func_name: str, 
        description: str,
        return_direct: bool = False,
        injected_state: bool = False
    ):
    name = agent if isinstance(agent, str) else agent.name
    
    @tool(func_name, return_direct=return_direct)
    def _agent_as_tool():
        """{description}"""
        return f"{AGENT_RESPONSE_PREFIX}{name}"
    _agent_as_tool.description = description
    return _agent_as_tool

def create_swarm_handoff(agent: Agent | str, func_name: str|None = None, description: str|None = None):
    name = agent if isinstance(agent, str) else agent.name
    # refactoring name to be a function name
    func_name = func_name or f"transfer_to_{_generate_snake_case_name(name)}"
    description = description or f"Call this function if a user is asking about a topic that should be handled by {name}."
    return create_swarm_agent_as_tool(agent, func_name, description, return_direct=True)

def create_swarm_backlink(agent: Agent | str, func_name: str|None = None, description: str|None = None):
    name = agent if isinstance(agent, str) else agent.name
    # refactoring name to be a function name
    func_name = func_name or f"transfer_to_{_generate_snake_case_name(name)}"
    description = description or f"Call this function if a user is asking about a topic that is not handled by the current agent."
    return create_swarm_agent_as_tool(agent, func_name, description, return_direct=True)

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
            

# following swarm's implementation, pass the context variables to the tools if needed
#   TODO: maybe there is any better way to do it?
#   also, swarm removes the context variables from the model calls, and fills it back before tool calls
def _try_fill_tool_calls(tool_calls: Sequence[ToolCall], state: AgentState) -> bool:
    changed = False
    for tool_call in tool_calls:
        args = tool_call.get("args", {})
        if __CTX_VARS_NAME__ in args:
            args[__CTX_VARS_NAME__] = state.get("context_variables", {})
            changed = True

    return changed

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


class HandoffsState(TypedDict):
    """The state of the agent handoffs to other agents."""

    messages: Sequence[BaseMessage] = None
    agent_name: Optional[str] = None
    handoff: bool = False
    user_end: bool = False
    context_variables: dict = {}

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
        
        def _post_process(state: AgentState, agent_name: str, tools_to_handoffs: Dict[str, str]):
            # from `react` AgentState
            messages = state["messages"]
            this_turn_messages = []
            # check the messages in this turn
            for _msg in reversed(messages):
                if isinstance(_msg, AIMessage):
                    _msg.name = agent_name
                if isinstance(_msg, (ToolMessage, AIMessage)) and "agent_name" not in _msg.additional_kwargs:
                    this_turn_messages.append(_msg)
                else:
                    break

            this_turn_messages.reverse()
            add_agent_name_to_messages(agent_name, this_turn_messages)
            if print_messages:
                print_messages(this_turn_messages)

            next_agent = None
            context_vars = state.get(__CTX_VARS_NAME__, {})
            # check the tool messages only in this turn
            for _msg in reversed(messages):
                if isinstance(_msg, ToolMessage):
                    if not next_agent and _msg.name in tools_to_handoffs:
                        # use the latest agent as the next agent
                        next_agent = tools_to_handoffs[_msg.name]
                    # TODO: to be checked. handle tool response if context variables exist (swarm's implementation)
                    # following the swarm's implementation, update the context variables back to state's context_variables
                    # add the context variables into ToolMessage.artifact (tool.response_format=`content_and_artifact`)
                    if _msg.artifact and isinstance(_msg.artifact, dict) and __CTX_VARS_NAME__ in _msg.artifact:
                        context_vars = _update_context_variables(context_vars, _msg.artifact[__CTX_VARS_NAME__])
                else:
                    break
            # next_agent = state.get("next_agent", None)
            # if next_agent in ["__end__"]:
                # post_process here to handle `__end__` from `react_agent`
            #     next_agent = None
            if next_agent:  # TODO: add debug
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

class Swarm:
    def __init__(self, 
        agent: Agent,
        llm=None,
        state_scheme: Type[HandoffsState] = HandoffsState,
        debug: bool = False,
        print_messages: None | Callable = None,
    ):
        self.llm = llm
        self.print_messages = print_messages or default_print_messages
        self.debug = debug
        self.state_scheme = state_scheme
        self.start_agent = agent
        self.agent = agent
        self.agent_map = {}
        if agent:
            self.workflow = self._create_workflow()

    def _create_workflow(self) -> CompiledGraph:
        llm = self.llm or create_default_llm(self.agent.model)
        return create_swarm_workflow(llm, self.agent, state_scheme=self.state_scheme, print_messages=self.print_messages, debug=self.debug, agent_map=self.agent_map)

    def get_agent(self, name: str, fail_if_not_found: bool = True):
        if name in self.agent_map:
            return self.agent_map[name][0]
        if fail_if_not_found:
            raise ValueError(f"Agent {name} not found")
        return None
    
    def run_stream(self, input: dict, max_turns: int):
        init_len = len(input["messages"])
        for _chunk in self.workflow.stream(
            input=input,
            config={"recursion_limit": max_turns},
            debug=self.debug,
            subgraphs=self.debug,
            ):
            if isinstance(_chunk, Dict):
                for _agent, _resp in _chunk.items():
                    yield {_agent: _resp}
                resp = _resp
            # TODO: parse chunk when workflow debug is enabled and subgraphs is True
            elif isinstance(_chunk, Tuple):
                for _item in _chunk:
                    if isinstance(_item, Dict):
                        for _agent, _resp in _item.items():
                            yield {_agent: _resp}
                        resp = _resp
                    else:
                        yield _item
        agent_name = resp["agent_name"]
        self.agent = self.get_agent(agent_name, False)
        if self.print_messages:
            self.print_messages(resp["messages"][init_len:])
        yield Response(
            messages=resp["messages"][init_len:],
            agent=resp["agent_name"],
            context_variables=resp.get(__CTX_VARS_NAME__, {}),
            handoff=resp.get("handoff", False),
        )
    
    # for now, trying to be compatible with OpenAI Swarm
    def run(
        self,
        messages: List,
        agent: Agent = None,
        context_variables: dict = {},
        stream: bool = False,
        max_turns: int = 25,
        execute_tools: bool = True,
    ) -> Response:
        if agent:
            self.agent = agent
            # in case you want to create a new workflow with a different agent
            if agent.name not in self.agent_map:
                self.start_agent = agent
                self.agent_map = {}
                self.workflow =self._create_workflow()
        agent_name = self.agent.name
        # TODO: use max_turns instead of Graph's recursion
        init_len = len(messages)
        input={"messages": messages, "agent_name": agent_name, "handoff": True, __CTX_VARS_NAME__: context_variables}
        if stream:
            return self.run_stream(input, max_turns)
        else:
            resp = self.workflow.invoke(
                input=input,
                config={"recursion_limit": max_turns},
            )
            agent_name = resp["agent_name"]
            self.agent = self.get_agent(agent_name, False)
            if self.print_messages:
                self.print_messages(resp["messages"][init_len:])
            return Response(
                messages=resp["messages"][init_len:],
                agent=resp["agent_name"],
                context_variables=resp.get(__CTX_VARS_NAME__, {}),
                handoff=resp.get("handoff", False),
            )