from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import Enum

from agent_brain.memory import Memory


class State(ABC):
    def on_enter(self, memory: "Memory") -> None:
        return

    def on_exit(self, memory: "Memory") -> None:
        return

    @abstractmethod
    async def run(self, memory: "Memory") -> AsyncIterator[str]:
        yield ""

    @abstractmethod
    async def next_state(self, memory: "Memory") -> Enum: ...
