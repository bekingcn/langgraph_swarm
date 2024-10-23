import os
from typing import Literal, Sequence
from datetime import datetime

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)


DEFAULT_LITERAL = Literal["default"]

def debug_print(debug: bool, *args: str) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def add_agent_name_to_messages(agent_name: str, messages: Sequence[BaseMessage]):
    for message in messages:
        if isinstance(message, AIMessage) and message.name is None:
            message.name = agent_name
        message.additional_kwargs["agent_name"] = agent_name
        
def get_agent_name_from_message(message: BaseMessage):
    return message.additional_kwargs.get("agent_name", None)

def default_print_messages(messages, debug: bool = True):
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for message in messages:
        agent_name = get_agent_name_from_message(message) or "User"
        print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {agent_name}[{message.type}]: {message}\033[0m")

def create_default_llm(model_name=None):
    from langchain_openai import ChatOpenAI
    model_name = model_name or os.environ.get("MODEL_NAME", "gpt-4o")
    return ChatOpenAI(model=model_name)
