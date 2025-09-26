from enum import Enum
from typing import Any

from pydantic import BaseModel


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    ACT = "system"


class Message(BaseModel):
    role: Role
    content: str


class AIMessage(BaseModel):
    type: str
    message: str


class Action(BaseModel):
    name: str
    args: dict[str, Any]
