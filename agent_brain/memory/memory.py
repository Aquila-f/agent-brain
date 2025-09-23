from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_brain.states import State, StateType


class MemoryEnv:
    def __init__(self, initial_state: "State", states_map: dict["StateType", "State"]) -> None:
        self.state: State = initial_state
        self.states: dict[StateType, State] = states_map
        self.done = False

    async def step(self) -> AsyncGenerator[str]:
        async for chunk in self.state.run(self):
            yield chunk

    def set_state(self, state: "StateType") -> None:
        self.state = self.states[state]
