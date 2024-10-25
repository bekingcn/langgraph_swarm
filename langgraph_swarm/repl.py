from typing import Sequence

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from .types import Agent, Response, HandoffsState
from .util import default_print_messages, create_default_llm
from .core import Swarm

HANDOFF_MSG_MARKER = "handoff_msg"

def filter_messages(messages: Sequence[BaseMessage], including_handoffs: bool = True) -> Sequence[BaseMessage]:
    """
    Filter out intermediate messages, and handoff messages if `including_handoffs` is False
    This assume that only response messages are useful for following turns

    Args:
        messages (Sequence[BaseMessage]): messages (in last turn) to be filtered
        including_handoffs (bool, optional): whether to include handoff messages. Defaults to True.

    Returns:
        Sequence[BaseMessage]: filtered messages to be used for following turns
    """

    filtered_messages = []
    for message in messages:
        # maybe human message which is from tool execution?
        if isinstance(message, (AIMessage, HumanMessage)) and message.content.strip():
            filtered_messages.append(message)
        if isinstance(message, ToolMessage) and including_handoffs and HANDOFF_MSG_MARKER in message.additional_kwargs:
            filtered_messages.append(message)
    last_message = messages[-1]
    last_added = filtered_messages[-1] if filtered_messages else None
    # TODO: here we assume that last message should be returned (could be a tool message with `return_direct=True`)
    if last_added and last_added != last_message:
        filtered_messages.append(last_message)
    return filtered_messages

def run_demo_loop(
    starting_agent: Agent, 
    llm=None, 
    context_variables={}, 
    stream=False, 
    debug=False, 
    print_messages="default",
    max_turns: int = 25,
    user_inputs=[],
    with_filter: bool = False,
) -> None:
    if print_messages == "default":
        print_messages = default_print_messages
    llm = llm or create_default_llm(starting_agent.model)
    client = Swarm(
        agent=starting_agent,
        llm=llm,
        state_scheme=HandoffsState,
        debug=debug,
        print_messages=print_messages,
    )
    messages = []
    current_agent = starting_agent.name
    handoff = True
    user_inputs_index = 0
    while True:
        if user_inputs:
            if user_inputs_index >= len(user_inputs):
                user_input = "/end"
            else:
                user_input = user_inputs[user_inputs_index]
            user_inputs_index += 1
        else:
            user_input = input("User: ")
        if user_input.strip() == "/end":
            break
        user_message = HumanMessage(content=user_input, name="User")
        if print_messages:
            print_messages([user_message])
        messages.append(user_message)
        ret = client.run(
            messages=messages,
            context_variables=context_variables,
            stream=stream,
            max_turns=max_turns,
        )

        if stream:
            # TODO: handle streaming responses
            for _chunk in ret:
                if isinstance(_chunk, Response):
                    resp = _chunk
                elif isinstance(_chunk, dict):
                    for _agent, _resp in _chunk.items():
                        if debug:
                            print(f"==> {_agent}: {_resp}")
                else:
                    if debug:
                        print(f"==> {_chunk}")
        else:
            resp = ret
        messages.extend(filter_messages(resp.messages) if with_filter else resp.messages)        
        current_agent = resp.agent
        context_variables = resp.context_variables
        handoff = resp.handoff  # for next turn, handoff always is true

        print(f"Agent Response ({current_agent}):\n", messages[-1].content)
        if context_variables:
            print(f"Context Variables:\n", context_variables)
