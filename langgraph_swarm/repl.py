import os
from typing import Literal
from langchain_core.messages import HumanMessage
from .router import create_swarm_workflow, Swarm, HandoffsState
from .types import Agent, Response
from .util import default_print_messages, create_default_llm, get_agent_name_from_message

def run_demo_loop(
    starting_agent: Agent, 
    llm=None, 
    context_variables={}, 
    stream=False, 
    debug=False, 
    print_messages="default",
    max_turns: int = 25,
    user_inputs=[],
) -> None:
    if print_messages == "default":
        print_messages = default_print_messages
    llm = llm or create_default_llm(starting_agent.model)
    client = Swarm(
        agent=starting_agent,
        llm=llm,
        state_scheme=HandoffsState,
        debug=debug,
        print_messages=print_messages if debug else None,
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
        messages.extend(resp.messages)
        current_agent = resp.agent
        context_variables = resp.context_variables
        handoff = resp.handoff  # for next turn, handoff always is true
        print(f"Final Response ({current_agent}):\n", messages[-1].content)
