import sys
sys.path.append("../../")
import dotenv

from langgraph_swarm import Swarm, Agent, tool
dotenv.load_dotenv()


@tool
def get_weather(location) -> str:
    """
    Returns the weather in a given location

    Args:
        location (str): The location to get the weather for

    Returns:
        str: The weather in the location
    """
    return "{'temp':67, 'unit':'F'}"


agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
    functions=[get_weather],
)


client = Swarm(agent=agent)
messages = [{"role": "user", "content": "What's the weather in NYC?"}]

response = client.run(agent=agent, messages=messages)
print(response.messages[-1].content)
