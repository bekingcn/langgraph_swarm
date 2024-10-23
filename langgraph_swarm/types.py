from typing import List, Callable, Optional, Union
from pydantic import BaseModel

from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage

# following swarm's model
class Agent(BaseModel):
    name: str = "Agent"
    model: str = "llama3.2"
    # TODO: use prompt template instead of Callable
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    handoffs: List["Agent"] = []
    backlink: bool = False
    functions: List[BaseTool] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True

class Response(BaseModel):
    messages: List[BaseMessage] = []
    agent: Optional[str] = None
    context_variables: dict = {}
    handoff: bool = False