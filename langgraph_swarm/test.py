from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from .core import Agent, create_swarm_workflow

@tool
def process_refund(item_id: str, reason: str="NOT SPECIFIED"):
    """
    Refund an item. Refund an item. Make sure you have the item_id of the form item_... Ask for user confirmation before processing the refund.
    
    Args:
        item_id: The ID of the item to refund.
        reason: The reason for the refund. Defaults to "NOT SPECIFIED".
    """
    print(f"[mock] Refunding item {item_id} because {reason}...")
    return "Success!"

@tool
def apply_discount():
    """
    Apply a discount to the user's cart.
    
    Args:
        None
    """
    print("[mock] Applying discount...")
    return "Applied discount of 11%"



triage_agent = Agent(
    name="Triage Agent",
    instructions="Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent.",
    backlink=False
)
sales_agent = Agent(
    name="Sales Agent",
    instructions="Be super enthusiastic about selling bees.",
    backlink=True
)
refunds_agent = Agent(
    name="Refunds Agent",
    instructions="Help the user with a refund. If the reason is that it was too expensive, offer the user a refund code. If they insist, then process the refund.",
    backlink=True
    # functions=[process_refund, apply_discount],
)


triage_agent.handoffs = [sales_agent, refunds_agent]
# sales_agent.handoffs = transfer_back_to_triage
refunds_agent.functions = [process_refund, apply_discount]

wf = create_swarm_workflow(
    llm=ChatOpenAI(temperature=0.3, model_name="llama3.2:latest", openai_api_key="OPENAI_API_KEY", base_url="http://localhost:11434/v1"),
    entrypoint=triage_agent,
    with_user_agent=False,
    debug=False,
)

def pretty_print(messages):
    for message in messages:
        print(message)

def run_demo_loop(wf, user_inputs):
    resp = {"messages": [], "agent_name": triage_agent.name, "handoff": True}
    user_inputs_index = 0
    while True:
        # user_input = input("User: ")
        user_input = user_inputs[user_inputs_index]
        user_inputs_index += 1
        if user_input.strip() == "/end":
            break
        print(f"User Input[human]: {user_input}")
        messages = resp["messages"] + [HumanMessage(content=user_input, name="User")]
        current_agent = resp["agent_name"]
        resp = wf.invoke({"messages": messages, "agent_name": current_agent, "handoff": True})
        # pretty_print(resp["messages"])
    


def run_demo_loop_stream(wf, user_inputs):
    resp = {"messages": [], "agent_name": triage_agent.name, "handoff": True}
    user_inputs_index = 0
    # user_input = input("User: ")
    user_input = user_inputs[user_inputs_index]
    user_inputs_index += 1
    # if user_input.strip() == "/end":
    #     break
    print(f"User Input[human]: {user_input}")
    messages = resp["messages"] + [HumanMessage(content=user_input, name="User")]
    current_agent = resp["agent_name"]
    for resp in wf.stream(input={"messages": messages, "agent_name": current_agent, "handoff": True}):
        print(resp)

if __name__ == "__main__":            
    user_inputs = [
        "I need a refund",
        "Fine, it's done. and I want to talk to sales",
        "I want to talk to sales",    
        "/end",
    ]
    run_demo_loop(wf, user_inputs)