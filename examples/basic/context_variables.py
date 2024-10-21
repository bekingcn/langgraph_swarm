import sys
from typing import Optional
sys.path.append("../../")
import dotenv

from langgraph_swarm import Swarm, Agent, tool
from langgraph_swarm.core import HandoffsState
dotenv.load_dotenv()


# TODO: not working, support update function result back to state or context variables
class HandoffsWithContextState(HandoffsState):
    user_id: Optional[str] = None
    name: Optional[str] = None


def instructions(context_variables):
    name = context_variables.get("name", "User")
    return f"You are a helpful agent. Greet the user by name ({name})."

@tool
def print_account_details(context_variables: dict):
    """
    Print account details
    
    Args:
        context_variables (dict): The context variables.
    
    Returns:
        str: A success message.
    """
    user_id = context_variables.get("user_id", None)
    name = context_variables.get("name", None)
    print(f"Account Details: {name} {user_id}")
    return "Success"


agent = Agent(
    name="Agent",
    instructions=instructions,
    functions=[print_account_details],
)

client = Swarm()
context_variables = {"name": "James", "user_id": 123}

response = client.run(
    messages=[{"role": "user", "content": "Hi!"}],
    agent=agent,
    context_variables=context_variables,
    stream=True,
)
print(response.messages[-1].content)

response = client.run(
    messages=[{"role": "user", "content": "Print my account details!"}],
    agent=agent,
    context_variables=context_variables,
)
print(response.messages[-1].content)