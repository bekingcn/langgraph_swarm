from typing import Optional

from langgraph_swarm import Swarm, Agent, tool
from langgraph_swarm.core import HandoffsState
import dotenv

from langgraph_swarm.types import Response
dotenv.load_dotenv()


# TODO: not working, support update function result back to state or context variables
class HandoffsWithContextState(HandoffsState):
    user_id: Optional[str] = None
    name: Optional[str] = None


# the instructions should follow the format: Callable[[StateSchema], Sequence[BaseMessage]]
# should include the `messages`
from langchain_core.messages import SystemMessage

def instructions(state: dict):
    context_variables = state.get("context_variables", {})
    name = context_variables.get("name", "User")
    prompt = f"You are a helpful agent. Your name is {name}. Process the user's request and respond with result."
    return [SystemMessage(content=prompt)] + state["messages"]

from langchain_core.prompts.chat import ChatPromptTemplate
# TODO: we should fix this to support ChatPromptTemplate directly (in agent_v03.py)
#   this is a trick to merge context variables into state
def _merge_context_variables(state: dict) -> dict:
    new_state = state.copy()
    context_variables = new_state.pop("context_variables", {})
    return {**new_state, **{f"context_variables_{k}": v for k, v in context_variables.items()}}

instructions_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful agent. Your name is {name}. Process the user's request and respond with result."),
        ("placeholder", "{messages}"),
    ]
)

@tool
def print_account_details(context_variables: dict):
    """
    Print account details if user desires, otherwise not called.
    
    Args:
        context_variables (dict): The context variables.
    
    Returns:
        str: A success message.
    """
    user_id = context_variables.get("user_id", None)
    name = context_variables.get("name", None)
    print(f"Account Details: {name} {user_id}")
    return "Success"

@tool(response_format="content_and_artifact")
def set_current_language(language: str):
    """
    Set the current language.
    
    Args:
        language (str): The language to set.
    
    Returns:
        str: A success message.
    """
    print(f"Setting language to {language}")
    return (f"done with setting language to {language}", {"context_variables": {"language": language}})


agent = Agent(
    name="Agent",
    # instructions="You are a helpful agent. Handle user's request and respond with result."
    # instructions = instructions,
    instructions=instructions_template,
    functions=[print_account_details, set_current_language]
)

client = Swarm(agent=agent, debug=True)
context_variables = {"name": "James", "user_id": 123, "language": "English"}

def print_stream_response(response):
    for resp in response:
        if isinstance(resp, dict):
            for _agent, _resp in resp.items():
                print(_agent, ": ", _resp)
        elif isinstance(resp, Response):
            print(resp.messages[-1].content)
        else:
            print(resp)

# ----
response = client.run(
    messages=[{"role": "user", "content": "Hi!"}],
    agent=agent,
    context_variables=context_variables,
    stream=True,
)

print_stream_response(response)

# ----
response = client.run(
    messages=[{"role": "user", "content": "change current language to Spanish."}],
    agent=agent,
    context_variables=context_variables,
    stream=True,
)

print_stream_response(response)

# ----
response = client.run(
    messages=[{"role": "user", "content": "Print my account details!"}],
    agent=agent,
    context_variables=context_variables,
)
print(response.messages[-1].content)
