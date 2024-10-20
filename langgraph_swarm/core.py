from typing import Dict, Tuple, Dict, Optional, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.runnables import Runnable

from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph



from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.tools import tool

from .agent import _create_swarm_agent, AGENT_RESPONSE_PREFIX
from .types import Agent

def create_swarm_handoff(agent: Agent | str, func_name: str|None = None):
    name = agent if isinstance(agent, str) else agent.name
    # refactoring name to be a function name
    func_name = func_name or name.replace(" ", "_").lower()
    
    @tool
    def _agent_as_tool():
        """Call this function if a user is asking about a topic that should be handled by {name}."""
        return f"{AGENT_RESPONSE_PREFIX}{name}"

    _agent_as_tool.description = f"Call this function if a user is asking about a topic that should be handled by {name}."
    _agent_as_tool.name = func_name
    return _agent_as_tool

def create_swarm_backlink(agent: Agent | str, func_name: str|None = None):
    name = agent if isinstance(agent, str) else agent.name
    # refactoring name to be a function name
    func_name = name.replace(" ", "_").lower()
    
    @tool
    def _agent_as_tool():
        """Call this function if a user is asking about a topic that is not handled by the current agent."""
        return f"{AGENT_RESPONSE_PREFIX}{name}"

    _agent_as_tool.description = f"Call this function if a user is asking about a topic that is not handled by the current agent."
    _agent_as_tool.name = func_name
    return _agent_as_tool

def add_agent_name_to_messages(agent_name: str, messages: Sequence[BaseMessage]):
    for message in messages:
        if isinstance(message, AIMessage) and message.name is None:
            message.name = agent_name
        message.additional_kwargs["agent_name"] = agent_name
        
def get_agent_name_from_message(message: BaseMessage):
    return message.additional_kwargs.get("agent_name", None)

def create_swarm_agent_and_handoffs(llm, agent: Agent, backlink_agent: Agent|None = None, agent_map={}):
    
    tools = agent.functions.copy()
    # process handoffs
    for handoff in agent.handoffs:
        tools.append(create_swarm_handoff(handoff))
    if backlink_agent:
        tools.append(create_swarm_backlink(backlink_agent))
    print(f"{agent.name} tools",tools)
    lc_agent = _create_swarm_agent(
        model=llm,
        tools=tools,
        agent_name=agent.name,
        messages_modifier=agent.instructions,
        debug=False
    )
    
    agent_map[agent.name] = (agent, lc_agent)
    
    for handoff in agent.handoffs:
        create_swarm_agent_and_handoffs(llm, handoff, backlink_agent=agent if handoff.backlink else None)
    
    return lc_agent


class HandoffsState(TypedDict):
    """The state of the agent handoffs to other agents."""

    messages: Sequence[BaseMessage] = []
    agent_name: Optional[str] = None
    handoff: bool = False
    user_end: bool = False
    
def create_swarm_workflow(
    llm,
    entrypoint: Agent,
    with_user_agent: bool = False,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    interrupt_before: Optional[Sequence[str]] = None,
    interrupt_after: Optional[Sequence[str]] = None,
    debug: bool = False,
) -> Runnable:
    
    agent = entrypoint
    # hold all created agents with name and lc_agent
    agent_map: Dict[str, Tuple[Agent, CompiledGraph]] = {}
    
    create_swarm_agent_and_handoffs(llm, agent, backlink_agent=None, agent_map=agent_map)
    
    # Define a new graph
    workflow = StateGraph(HandoffsState)
    
    def print_messages(state: HandoffsState, messages_from=0):
        messages = state["messages"]
        agent_name = state["agent_name"]      
        for message in messages[messages_from:]:
            print(f"{agent_name}[{message.type}]: {message}")
    
    def human_agent_node(state: HandoffsState):
        user_input = input("User: ")
        if user_input.strip() == "/end":
            return {"user_end": True}
        return {"messages": [HumanMessage(content=user_input)]}

    def nop_agent_node(state: HandoffsState):
        return {}
    
    start_node = human_agent_node if with_user_agent else nop_agent_node
    
    workflow.add_node("entrypoint", start_node)
    workflow.set_entry_point("entrypoint")
    branchs = {}
    from functools import partial
    for name, (agent, lc_agent) in agent_map.items():
        def call_agent_node(state: HandoffsState, agent_name: str, lc_agent):
            print("==> entering: ", agent_name)
            messages = state["messages"]
            init_len = len(messages)
            agent_response = lc_agent.invoke(input={"messages": state["messages"], "next_agent": None, "agent_name": agent_name})
            messages = agent_response["messages"]
            add_agent_name_to_messages(agent_name, messages[init_len:])
            print_messages(agent_response, messages_from=init_len)
            # if None, reponse only, otherwise next agent
            next_agent = agent_response.get("next_agent", None)
            if next_agent:
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

