from langgraph_swarm import Swarm, Agent
import dotenv
dotenv.load_dotenv()

agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
)


client = Swarm(agent=agent)
messages = [{"role": "user", "content": "Hi!"}]
response = client.run(agent=agent, messages=messages)

print(response.messages[-1].content)
