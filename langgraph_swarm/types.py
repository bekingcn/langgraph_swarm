from typing import List, Callable, Union
from pydantic import BaseModel

from langchain_core.tools import BaseTool

# following swarm's model
class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    handoffs: List["Agent"] = []
    backlink: bool = False
    functions: List[BaseTool] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True
    
    def get_node_name(self):
        return self.name.replace(" ", "_").lower()