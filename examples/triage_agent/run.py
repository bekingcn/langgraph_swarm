import dotenv
from langgraph_swarm.repl import run_demo_loop
from agents import triage_agent

dotenv.load_dotenv()
if __name__ == "__main__":
    run_demo_loop(starting_agent=triage_agent)

    user_inputs = [
    "I need a refund for item_12345",
    "Fine, it's done. and I want to talk to sales",
    "I want to talk to sales",    
    "/end",
    ]
    # run_demo_loop(starting_agent=triage_agent, user_inputs=user_inputs)