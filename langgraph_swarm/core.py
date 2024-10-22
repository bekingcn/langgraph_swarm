from typing import Any, Callable, Dict, Generator, List, Literal, Tuple, Dict, Optional, Sequence, Type, TypedDict
import re
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

from langgraph_swarm.util import add_agent_name_to_messages, create_default_llm, default_print_messages

from .agent_v03 import _create_swarm_agent  , AGENT_RESPONSE_PREFIX
from .types import Agent, Response

def _generate_function_name(agent_name: str):
    return re.sub(r"[^0-9a-z_]", "_", agent_name.lower())

def create_swarm_handoff(agent: Agent | str, func_name: str|None = None):
    name = agent if isinstance(agent, str) else agent.name
    # refactoring name to be a function name
    func_name = _generate_function_name(name)
    
    @tool
    def _agent_as_tool():
        """Call this function if a user is asking about a topic that should be handled by {name}."""
        return f"{AGENT_RESPONSE_PREFIX}{name}"

    _agent_as_tool.description = f"Call this function if a user is asking about a topic that should be handled by {name}."
    _agent_as_tool.name = f"transfer_to_{func_name}"
    return _agent_as_tool

def create_swarm_backlink(agent: Agent | str, func_name: str|None = None):
    name = agent if isinstance(agent, str) else agent.name
    # refactoring name to be a function name
    func_name = _generate_function_name(name)
    
    @tool
    def _agent_as_tool():
        """Call this function if a user is asking about a topic that is not handled by the current agent."""
        return f"{AGENT_RESPONSE_PREFIX}{name}"

    _agent_as_tool.description = f"Call this function if a user is asking about a topic that is not handled by the current agent."
    _agent_as_tool.name = f"transfer_to_{func_name}"
    return _agent_as_tool

def create_swarm_agent_and_handoffs(llm, agent: Agent, backlink_agent: Agent|None = None, agent_map={}):
    
    tools = agent.functions.copy()
    # process handoffs
    for handoff in agent.handoffs:
        tools.append(create_swarm_handoff(handoff))
    if backlink_agent:
        tools.append(create_swarm_backlink(backlink_agent))
    lc_agent = _create_swarm_agent(
        model=llm,
        tools=tools,
        agent_name=agent.name,
        state_modifier=agent.instructions,
        debug=False
    )
    
    agent_map[agent.name] = (agent, lc_agent)
    
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
    agent_map: Dict[str, Tuple[Agent, CompiledGraph]] = {}
) -> CompiledGraph:
    
    agent = starting_agent
    
    create_swarm_agent_and_handoffs(llm, agent, backlink_agent=None, agent_map=agent_map)
    
    # Define a new graph
    workflow = StateGraph(state_scheme)
    
    def human_agent_node(state: HandoffsState):
        user_input = input("User: ")
        if user_input.strip() == "/end":
            return {"user_end": True}
        return {"messages": [HumanMessage(content=user_input)]}

    def nop_agent_node(state: HandoffsState):
        return state
    
    start_node = human_agent_node if with_user_agent else nop_agent_node
    
    workflow.add_node("entrypoint", start_node)
    workflow.set_entry_point("entrypoint")
    branchs = {}
    from functools import partial
    for name, (agent, lc_agent) in agent_map.items():
        # TODO: use a chain instead?
        def call_agent_node(state: HandoffsState, agent_name: str, lc_agent):
            if debug:
                print("==> entering: ", agent_name)
            messages = state["messages"]
            init_len = len(messages)
            resp = lc_agent.invoke(
                input={
                    "messages": state["messages"], 
                    "next_agent": None, 
                    "agent_name": agent_name, 
                    "context_variables": state.get("context_variables", {})
                }
            )
            messages = resp["messages"]
            add_agent_name_to_messages(agent_name, messages[init_len:])
            if print_messages:
                print_messages(messages[init_len:])
            # if None, reponse only, otherwise next agent
            next_agent = resp.get("next_agent", None)
            if next_agent in ["__end__"]:
                # post_process here to handle `__end__` from `react_agent`
                next_agent = None
            if debug and next_agent:
                print("==> handoff to: ", next_agent)
            return {"messages": messages, "agent_name": next_agent if next_agent else state["agent_name"], "handoff": next_agent is not None}
        
        workflow.add_node(agent.get_node_name(), partial(call_agent_node, agent_name=name, lc_agent=lc_agent))
        workflow.add_edge(agent.get_node_name(), "entrypoint")
        branchs[agent.name] = agent.get_node_name()
    branchs["end"] = END     
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
        "entrypoint",
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
        if stream:
            for _chunk in self.workflow.stream(
                input={"messages": messages, "agent_name": agent_name, "handoff": True, "context_variables": context_variables},
                config={"recursion_limit": max_turns},
                ):
                for _agent, _resp in _chunk.items():
                    if self.debug:
                        print(_agent, ": ", _resp)
                    # yield {_agent: _resp}
                resp = _resp
        else:
            resp = self.workflow.invoke(
                input={"messages": messages, "agent_name": agent_name, "handoff": True, "context_variables": context_variables},
                config={"recursion_limit": max_turns},
            )
        agent_name = resp["agent_name"]
        self.agent = self.get_agent(agent_name, False)
        return Response(
            messages=resp["messages"][init_len:],
            agent=resp["agent_name"],
            context_variables={},
        )