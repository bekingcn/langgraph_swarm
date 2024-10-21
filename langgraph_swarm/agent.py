from typing import Any, Callable, Dict, Optional, Sequence, Union, TypedDict, Annotated

from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep

from langchain_core.messages import ToolMessage

AGENT_RESPONSE_PREFIX = "transfer to "

class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    agent_name: str
    # the handoff agent from current agent
    next_agent: Optional[str] = None
    # swarm style variables holder, not supported
    # maybe it would be merged with the State in langgraph, need to declare the variables
    # it's used to parse the instructions
    context_variables: Dict[str, Any] = {}

    is_last_step: IsLastStep


def _agent_response(response: str) -> str | None:
    if response.startswith(AGENT_RESPONSE_PREFIX):
        return response[len(AGENT_RESPONSE_PREFIX):]
    return None

# NOTE: this is a refactoring of the langchain v0.2 create_react_agent function
#       to support swarm agent.
#       added a branch to exit the agent if there is a agent handoff as swarm style
def _create_swarm_agent(
    model: LanguageModelLike,
    tools: Union[ToolExecutor, Sequence[BaseTool]],
    agent_name: str = None,
    messages_modifier: Optional[Union[SystemMessage, str, Callable, Runnable]] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    interrupt_before: Optional[Sequence[str]] = None,
    interrupt_after: Optional[Sequence[str]] = None,
    debug: bool = False,
) -> CompiledGraph:

    if isinstance(tools, ToolExecutor):
        tool_classes = tools.tools
    else:
        tool_classes = tools
    model = model.bind_tools(tool_classes)

    # Define the function that determines whether to continue or not
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not last_message.tool_calls:
            return "end"
        # Otherwise if there is, we continue
        else:
            return "continue"

    # Add the message modifier, if exists
    if messages_modifier is None:
        model_runnable = model
    elif isinstance(messages_modifier, str):
        _system_message: BaseMessage = SystemMessage(content=messages_modifier)
        model_runnable = (lambda messages: [_system_message] + messages) | model
    elif isinstance(messages_modifier, SystemMessage):
        model_runnable = (lambda messages: [messages_modifier] + messages) | model
    elif isinstance(messages_modifier, (Callable, Runnable)):
        model_runnable = messages_modifier | model
    else:
        raise ValueError(
            f"Got unexpected type for `messages_modifier`: {type(messages_modifier)}"
        )

    # Define the function that calls the model
    def call_model(
        state: AgentState,
        config: RunnableConfig,
    ):
        messages = state["messages"]
        response = model_runnable.invoke(messages, config)
        if state["is_last_step"] and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                        name=agent_name,
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        response.name = agent_name
        return {"messages": [response]}

    async def acall_model(state: AgentState, config: RunnableConfig):
        messages = state["messages"]
        response = await model_runnable.ainvoke(messages, config)
        if state["is_last_step"] and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                        name=agent_name,
                    )
                ]
            }
        # We return a list, because this will get added to the existing list
        response.name = agent_name
        return {"messages": [response]}
    
    def check_tool_result(state: AgentState):
        messages = state["messages"]
        # 
        # TODO: this is a trick to check and get agent name to handoff
        #   otherwise, we have to rewrite the tools executor to handle agent's responses
        # if mutliple tools, we should find the latest one with handoff agent 
        #   following the swarm's implementation
        agent_name = None
        for last_message in messages[::-1]:
            if not isinstance(last_message, ToolMessage):
                break
            agent_name = _agent_response(last_message.content)
            if agent_name:
                return {"next_agent": agent_name}
        else:
            return {}
    
    def should_handoff(state: AgentState):
        if state.get("next_agent", None):
            return "end"
        else:
            return "continue"

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", RunnableLambda(call_model, acall_model))
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("post_tools", check_tool_result)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    workflow.set_entry_point("agent")

    # We now add a conditional edge
    workflow.add_conditional_edges(
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        "agent",
        # Next, we pass in the function that will determine which node is called next.
        should_continue,
        # Finally we pass in a mapping.
        # The keys are strings, and the values are other nodes.
        # END is a special node marking that the graph should finish.
        # What will happen is we will call `should_continue`, and then the output of that
        # will be matched against the keys in this mapping.
        # Based on which one it matches, that node will then be called.
        {
            # If `tools`, then we call the tool node.
            "continue": "tools",
            # Otherwise we finish.
            "end": END,
        },
    )

    # We now add a normal edge from `tools` to `post_tools` for checking tools result.
    workflow.add_edge("tools", "post_tools")    
    workflow.add_conditional_edges(
        "post_tools",
        should_handoff,
        {
            # if handoffs, then we end this agent loop
            "end": END,
            # Otherwise we continue and back to `agent` node
            "continue": "agent",
        }
    )

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )

