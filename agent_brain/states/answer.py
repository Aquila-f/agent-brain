from collections.abc import AsyncGenerator

from agent_brain.memory import MemoryEnv

from .base import State


class AnswerState(State):
    async def _run(self, env: MemoryEnv) -> AsyncGenerator[str]:
        yield "answer"

    async def _transition(self, env: MemoryEnv) -> None:
        env.done = True
