import os
from typing import Any, Dict, List, Callable, Optional, Sequence, TypedDict, Union
from pydantic import BaseModel

from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable

# following swarm's model
class Agent(BaseModel):
    name: str = "Agent"
    model: str = os.environ.get("MODEL_NAME") or "llama3.2"
    instructions: Union[str, SystemMessage, Callable[[], Sequence[BaseMessage]], Runnable] = "You are a helpful agent."
    handoffs: List["Agent"] = []
    backlink: bool = False
    functions: List[BaseTool] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True

    # TODO: fix type mismatch, for Runnable[?, Sequence[BaseMessage]], ref StateModifier instead
    #  also, to support `context_variables` in instructions, see `_merge_context_variables` in /examples/basic/context_variables.py
    model_config: Dict[str, Any] = {"arbitrary_types_allowed": True}

class Response(BaseModel):
    messages: List[BaseMessage] = []
    agent: Optional[str] = None
    context_variables: dict = {}
    handoff: bool = False


class HandoffsState(TypedDict):
    """The state of the agent handoffs to other agents."""

    messages: Sequence[BaseMessage] = None
    agent_name: Optional[str] = None
    handoff: bool = False
    user_end: bool = False
    context_variables: dict = {}