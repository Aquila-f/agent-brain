from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_brain.memory import MemoryEnv


class State(ABC):
    def on_enter(self, env: "MemoryEnv") -> None:
        return

    def on_exit(self, env: "MemoryEnv") -> None:
        return

    async def run(self, env: "MemoryEnv") -> AsyncIterator[str]:
        async for output in self._run(env):
            yield output
        await self._transition(env)

    @abstractmethod
    async def _run(self, env: "MemoryEnv") -> AsyncIterator[Any]:
        yield ""

    @abstractmethod
    async def _transition(self, env: "MemoryEnv") -> None: ...
