from typing import (
    Annotated,
    Callable,
    Literal,
    Optional,
    Sequence,
    Type,
    TypedDict,
    TypeVar,
    Union,
    Dict,
    Any,
)

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage, ToolCall
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool

from langgraph._api.deprecation import deprecated_parameter
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt.tool_node import ToolNode
from langgraph.types import Checkpointer

from langchain_core.messages import ToolMessage


__CTX_VARS_NAME__ = "context_variables"
AGENT_RESPONSE_PREFIX = "transfer to "

class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    agent_name: str
    # the handoff agent from current agent
    next_agent: Optional[str] = None
    # swarm style variables holder
    context_variables: Dict[str, Any] = {}

    is_last_step: IsLastStep

StateSchema = TypeVar("StateSchema", bound=AgentState)
StateSchemaType = Type[StateSchema]

STATE_MODIFIER_RUNNABLE_NAME = "StateModifier"

MessagesModifier = Union[
    SystemMessage,
    str,
    Callable[[Sequence[BaseMessage]], Sequence[BaseMessage]],
    Runnable[Sequence[BaseMessage], Sequence[BaseMessage]],
]

StateModifier = Union[
    SystemMessage,
    str,
    Callable[[StateSchema], Sequence[BaseMessage]],
    Runnable[StateSchema, Sequence[BaseMessage]],
]


def _get_state_modifier_runnable(state_modifier: Optional[StateModifier]) -> Runnable:
    state_modifier_runnable: Runnable
    if state_modifier is None:
        state_modifier_runnable = RunnableLambda(
            lambda state: state["messages"], name=STATE_MODIFIER_RUNNABLE_NAME
        )
    elif isinstance(state_modifier, str):
        _system_message: BaseMessage = SystemMessage(content=state_modifier)
        state_modifier_runnable = RunnableLambda(
            lambda state: [_system_message] + state["messages"],
            name=STATE_MODIFIER_RUNNABLE_NAME,
        )
    elif isinstance(state_modifier, SystemMessage):
        state_modifier_runnable = RunnableLambda(
            lambda state: [state_modifier] + state["messages"],
            name=STATE_MODIFIER_RUNNABLE_NAME,
        )
    elif callable(state_modifier):
        state_modifier_runnable = RunnableLambda(
            state_modifier, name=STATE_MODIFIER_RUNNABLE_NAME
        )
    elif isinstance(state_modifier, Runnable):
        state_modifier_runnable = state_modifier
    else:
        raise ValueError(
            f"Got unexpected type for `state_modifier`: {type(state_modifier)}"
        )

    return state_modifier_runnable


def _convert_messages_modifier_to_state_modifier(
    messages_modifier: MessagesModifier,
) -> StateModifier:
    state_modifier: StateModifier
    if isinstance(messages_modifier, (str, SystemMessage)):
        return messages_modifier
    elif callable(messages_modifier):

        def state_modifier(state: AgentState) -> Sequence[BaseMessage]:
            return messages_modifier(state["messages"])

        return state_modifier
    elif isinstance(messages_modifier, Runnable):
        state_modifier = (lambda state: state["messages"]) | messages_modifier
        return state_modifier
    raise ValueError(
        f"Got unexpected type for `messages_modifier`: {type(messages_modifier)}"
    )


def _get_model_preprocessing_runnable(
    state_modifier: Optional[StateModifier],
    messages_modifier: Optional[MessagesModifier],
) -> Runnable:
    # Add the state or message modifier, if exists
    if state_modifier is not None and messages_modifier is not None:
        raise ValueError(
            "Expected value for either state_modifier or messages_modifier, got values for both"
        )

    if state_modifier is None and messages_modifier is not None:
        state_modifier = _convert_messages_modifier_to_state_modifier(messages_modifier)

    return _get_state_modifier_runnable(state_modifier)


AGENT_RESPONSE_PREFIX = "transfer to "

def _agent_response(response: str) -> str | None:
    if response.startswith(AGENT_RESPONSE_PREFIX):
        return response[len(AGENT_RESPONSE_PREFIX):]
    return None

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

# NOTE: this is a refactoring of the langchain v0.3 create_react_agent function
#       to support swarm agent.
#       added a branch to exit the agent if there is a agent handoff as swarm style
def _create_swarm_agent(
    model: BaseChatModel,
    tools: Union[ToolExecutor, Sequence[BaseTool], ToolNode],
    *,
    agent_name: str = None,
    state_schema: Optional[StateSchemaType] = None,
    messages_modifier: Optional[MessagesModifier] = None,
    state_modifier: Optional[StateModifier] = None,
    checkpointer: Checkpointer = None,
    interrupt_before: Optional[list[str]] = None,
    interrupt_after: Optional[list[str]] = None,
    debug: bool = False,
) -> CompiledGraph:
    """Creates a graph that works with a chat model that utilizes tool calling.

    Args:
        model: The `LangChain` chat model that supports tool calling.
        tools: A list of tools, a ToolExecutor, or a ToolNode instance.
        state_schema: An optional state schema that defines graph state.
            Must have `messages` and `is_last_step` keys.
            Defaults to `AgentState` that defines those two keys.
        messages_modifier: An optional
            messages modifier. This applies to messages BEFORE they are passed into the LLM.

            Can take a few different forms:

            - SystemMessage: this is added to the beginning of the list of messages.
            - str: This is converted to a SystemMessage and added to the beginning of the list of messages.
            - Callable: This function should take in a list of messages and the output is then passed to the language model.
            - Runnable: This runnable should take in a list of messages and the output is then passed to the language model.
            !!! Warning
                `messages_modifier` parameter is deprecated as of version 0.1.9 and will be removed in 0.2.0
        state_modifier: An optional
            state modifier. This takes full graph state BEFORE the LLM is called and prepares the input to LLM.

            Can take a few different forms:

            - SystemMessage: this is added to the beginning of the list of messages in state["messages"].
            - str: This is converted to a SystemMessage and added to the beginning of the list of messages in state["messages"].
            - Callable: This function should take in full graph state and the output is then passed to the language model.
            - Runnable: This runnable should take in full graph state and the output is then passed to the language model.
        checkpointer: An optional checkpoint saver object. This is useful for persisting
            the state of the graph (e.g., as chat memory).
        interrupt_before: An optional list of node names to interrupt before.
            Should be one of the following: "agent", "tools".
            This is useful if you want to add a user confirmation or other interrupt before taking an action.
        interrupt_after: An optional list of node names to interrupt after.
            Should be one of the following: "agent", "tools".
            This is useful if you want to return directly or run additional processing on an output.
        debug: A flag indicating whether to enable debug mode.

    Returns:
        A compiled LangChain runnable that can be used for chat interactions.
    """

    if state_schema is not None:
        if missing_keys := {"messages", "is_last_step"} - set(
            state_schema.__annotations__
        ):
            raise ValueError(f"Missing required key(s) {missing_keys} in state_schema")

    if isinstance(tools, ToolExecutor):
        tool_classes: Sequence[BaseTool] = tools.tools
        tool_node = ToolNode(tool_classes)
    elif isinstance(tools, ToolNode):
        tool_classes = list(tools.tools_by_name.values())
        tool_node = tools
    else:
        tool_classes = tools
        tool_node = ToolNode(tool_classes)
    model = model.bind_tools(tool_classes)

    # Define the function that determines whether to continue or not
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        # If there is no function call, then we finish
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return "__end__"
        # Otherwise if there is, we continue
        else:
            return "tools"

    preprocessor = _get_model_preprocessing_runnable(state_modifier, messages_modifier)
    model_runnable = preprocessor | model

    # Define the function that calls the model
    def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
        response = model_runnable.invoke(state, config)
        if (
            state["is_last_step"]
            and isinstance(response, AIMessage)
            and response.tool_calls
        ):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                        name=agent_name,
                    )
                ]
            }
        # swarm added
        response.name = agent_name
        if response.tool_calls:
            _try_fill_tool_calls(response.tool_calls, state)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}

    async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
        response = await model_runnable.ainvoke(state, config)
        if (
            state["is_last_step"]
            and isinstance(response, AIMessage)
            and response.tool_calls
        ):
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, need more steps to process this request.",
                        name=agent_name,
                    )
                ]
            }
        # swarm added
        response.name = agent_name
        if response.tool_calls:
            _try_fill_tool_calls(response.tool_calls, state)
        # We return a list, because this will get added to the existing list
        return {"messages": [response]}


    # If any of the tools are configured to return_directly after running,
    # our graph needs to check if these were called
    should_return_direct = {t.name for t in tool_classes if t.return_direct}

    # Define a new graph
    workflow = StateGraph(state_schema or AgentState)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", RunnableLambda(call_model, acall_model))
    workflow.add_node("tools", tool_node)

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
    )

    # swarm added    
    def check_tools_result(state: AgentState):
        context_vars = state.get("context_variables", {})
        for last_message in reversed(state["messages"]):
            if not isinstance(last_message, ToolMessage):
                break
            # TODO: to be checked. handle tool response if context variables exist (swarm's implementation)
            # following the swarm's implementation, update the context variables back to state's context_variables
            # add the context variables into ToolMessage.artifact (tool.response_format=`content_and_artifact`)
            context_vars = state.get("context_variables", {})
            if last_message.artifact and isinstance(last_message.artifact, dict) and __CTX_VARS_NAME__ in last_message.artifact:
                context_vars = _update_context_variables(context_vars, last_message.artifact[__CTX_VARS_NAME__])
                
            # NOTE: for now, this is a trick to identify handoff's name
            # if mutliple tools, we should find the latest one which has `next_agent` following the swarm's implementation
            agent_name = _agent_response(last_message.content)
            if agent_name:
                return {"next_agent": agent_name, "context_variables": context_vars}
            if last_message.name in should_return_direct:
                # should handle `__end__` in the core.py
                return {"next_agent": "__end__", "context_variables": context_vars}
        return {"next_agent": None, "context_variables": context_vars}

    # We now add a normal edge from `tools` to `post_tools` for checking tools response.
    workflow.add_node("post_tools", check_tools_result)
    workflow.add_edge("tools", "post_tools")

    # swarm modified
    def route_tool_responses(state: AgentState) -> Literal["agent", "__end__"]:
        # additional check for handoff
        if state.get("next_agent", None):
            return "__end__"
            
        return "agent"

    workflow.add_conditional_edges("post_tools", route_tool_responses)

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
    )