from collections.abc import AsyncGenerator

from agent_brain.memory import MemoryEnv

from .base import State
from .state_type import StateType


class ReasoningState(State):
    def __init__(self) -> None:
        self.counter = 0

    async def _run(self, env: MemoryEnv) -> AsyncGenerator[str]:
        # TODO: implement thinking and parse action logic
        yield f"reasoning {self.counter}"

    async def _transition(self, env: MemoryEnv) -> None:
        self.counter += 1
        if self.counter >= 2:
            env.set_state(StateType.ANSWER)
        else:
            env.set_state(StateType.ACTION)
