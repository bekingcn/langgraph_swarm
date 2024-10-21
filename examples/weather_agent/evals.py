import sys
sys.path.append("../../")

from langchain_core.messages import AIMessage
from agents import weather_agent
from langgraph_swarm import Swarm, Agent
import pytest

import dotenv
dotenv.load_dotenv()

# TODO: we could not store tool execution for now.
#    so this fix make tools return as quickly.
for tool in weather_agent.functions:
    tool.return_direct = True
client = Swarm(agent=weather_agent, debug=True)

def run_and_get_tool_calls(agent: Agent, query):
    message = {"role": "user", "content": query}
    response = client.run(
        agent=agent,
        messages=[message],
        execute_tools=False,
    )
    for m in reversed(response.messages):
        if isinstance(m, AIMessage) and m.tool_calls:
            return m.tool_calls
    return []


@pytest.mark.parametrize(
    "query",
    [
        "What's the weather in NYC?",
        "Tell me the weather in London.",
        "Do I need an umbrella today? I'm in chicago.",
    ],
)
def test_calls_weather_when_asked(query):
    tool_calls = run_and_get_tool_calls(weather_agent, query)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "get_weather"


@pytest.mark.parametrize(
    "query",
    [
        "Who's the president of the United States?",
        "What is the time right now?",
        "Hi!",
    ],
)
def test_does_not_call_weather_when_not_asked(query):
    tool_calls = run_and_get_tool_calls(weather_agent, query)

    assert not tool_calls

if __name__ == "__main__":
    tool_calls = run_and_get_tool_calls(weather_agent, "What's the weather in NYC?")
    print(tool_calls)