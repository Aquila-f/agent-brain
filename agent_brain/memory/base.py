from abc import ABC, abstractmethod

from agent_brain.models import Action, Message
from agent_brain.tool import BaseTool


class Memory(ABC):
    def __init__(self, tools: list[BaseTool]) -> None:
        self.done: bool = False
        self.next_action: Action | None = None
        self.available_tools: dict[str, BaseTool] = {tool.name: tool for tool in tools}

    def list_tools(self) -> str:
        return ",".join([str(tool.to_dict()) for tool in self.available_tools.values()])

    def get_tool(self, name: str) -> BaseTool | None:
        return self.available_tools.get(name)

    @abstractmethod
    async def set_goal(self, goal: str) -> None: ...

    @abstractmethod
    async def update(self, messages: list[Message]) -> None: ...

    @abstractmethod
    async def goal(self) -> list[Message]: ...

    @abstractmethod
    async def dump(self) -> list[Message]: ...
