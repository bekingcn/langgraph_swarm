import sys
sys.path.append("../../")

from langgraph_swarm.repl import run_demo_loop
from agents import weather_agent

import dotenv
dotenv.load_dotenv()

if __name__ == "__main__":
    run_demo_loop(weather_agent, stream=True)
