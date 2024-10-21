import os
from typing import Literal
from langchain_core.messages import HumanMessage
from .core import create_swarm_workflow
from .types import Agent
from .util import default_print_messages, create_default_llm, get_agent_name_from_message

def run_demo_loop(
    starting_agent: Agent, 
    llm=None, 
    context_variables=None, 
    stream=False, 
    debug=False, 
    print_messages="default",
    user_inputs=[],
) -> None:
    if print_messages == "default":
        print_messages = default_print_messages
    llm = llm or create_default_llm(starting_agent.model)
    wf = create_swarm_workflow(
        llm=llm,
        starting_agent=starting_agent,
        print_messages=print_messages,
        with_user_agent=False,
        debug=debug,
    )
    resp = {"messages": [], "agent_name": starting_agent.name, "handoff": True}
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
        messages = resp["messages"] + [user_message]
        current_agent = resp["agent_name"]
        if stream:
            for _chunk in wf.stream(input={"messages": messages, "agent_name": current_agent, "handoff": True}):
                for _agent, _resp in _chunk.items():
                    if debug:
                        print(f"==> {_agent}: {_resp}")
                resp = _resp
        else:
            resp = wf.invoke(input={"messages": messages, "agent_name": current_agent, "handoff": True})
        # pretty_print(resp["messages"])
