from langgraph_swarm import Agent, tool

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
refunds_agent.functions = [process_refund, apply_discount]