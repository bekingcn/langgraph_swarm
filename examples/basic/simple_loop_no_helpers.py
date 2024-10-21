import sys
sys.path.append("../../")
import dotenv

from langgraph_swarm import Swarm, Agent, tool, get_agent_name_from_message
dotenv.load_dotenv()

my_agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
)


def pretty_print_messages(messages):
    for message in messages:
        if message.content is None:
            continue
        print(f"{get_agent_name_from_message(message)}: {message.content}")


messages = []
agent = my_agent

client = Swarm(agent=agent)
while True:
    user_input = input("> ")
    messages.append({"role": "user", "content": user_input})

    response = client.run(agent=agent, messages=messages)
    messages = response.messages
    agent = client.get_agent(response.agent)
    pretty_print_messages(messages)
