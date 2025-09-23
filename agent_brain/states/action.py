from collections.abc import AsyncGenerator

from agent_brain.memory import MemoryEnv

from .base import State
from .state_type import StateType


class ActionState(State):
    def __init__(self) -> None:
        self.counter = 0

    async def _run(self, env: MemoryEnv) -> AsyncGenerator[str]:
        yield f"action {self.counter}"

    async def _transition(self, env: MemoryEnv) -> None:
        self.counter += 1
        env.set_state(StateType.REASONING)
