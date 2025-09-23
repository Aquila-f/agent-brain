from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from agent_brain.memory import MemoryEnv


class State(ABC):
    async def run(self, env: MemoryEnv) -> AsyncGenerator[str]:
        async for chunk in self._run(env):
            yield chunk

        await self._transition(env)

    @abstractmethod
    async def _run(self, env: MemoryEnv) -> AsyncGenerator[str]: ...

    @abstractmethod
    async def _transition(self, env: MemoryEnv) -> None: ...
