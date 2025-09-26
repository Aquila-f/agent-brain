from agent_brain.models import Message, Role
from agent_brain.tool import BaseTool

from .base import Memory


class MessagesMemory(Memory):
    def __init__(self, tools: list[BaseTool]) -> None:
        self._history: list[Message] = []
        self._goal: Message | None = None
        super().__init__(tools)

    async def set_goal(self, goal: str) -> None:
        self._goal = Message(role=Role.USER, content=goal)

    async def update(self, messages: list[Message]) -> None:
        self._history.extend(messages)

    async def goal(self) -> list[Message]:
        return [self._goal] if self._goal else []

    async def dump(self) -> list[Message]:
        return self._history
