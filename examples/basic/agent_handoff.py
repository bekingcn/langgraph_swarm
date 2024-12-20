from langgraph_swarm import Swarm, Agent
import dotenv
dotenv.load_dotenv()

english_agent = Agent(
    name="English Agent",
    instructions="You only speak English.",
)

spanish_agent = Agent(
    name="Spanish Agent",
    instructions="You only speak Spanish.",
)

english_agent.handoffs.append(spanish_agent)

client = Swarm(agent=english_agent)
messages = [{"role": "user", "content": "Hola. ¿Como estás?"}]
response = client.run(agent=english_agent, messages=messages)

print(response.messages[-1].content)
