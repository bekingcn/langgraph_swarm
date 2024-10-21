import sys
sys.path.append("../../")
import dotenv

from langgraph_swarm import Swarm, Agent
dotenv.load_dotenv()

client = Swarm()

agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
)

messages = [{"role": "user", "content": "Hi!"}]
response = client.run(agent=agent, messages=messages)

print(response.messages[-1].content)