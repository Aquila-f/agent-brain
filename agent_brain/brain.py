import asyncio
from collections.abc import AsyncGenerator

from agent_brain.memory import MemoryEnv
from agent_brain.states import (
    ActionState,
    AnswerState,
    ReasoningState,
    State,
    StateType,
)


class Brain:
    def __init__(self, initial_state: State, state_map: dict[StateType, State]) -> None:
        self.env = MemoryEnv(initial_state, state_map)
        self.loop_limit = 5

    async def answer(self) -> AsyncGenerator[str]:
        loop_count = 0
        while not self.env.done and loop_count < self.loop_limit:
            async for chunk in self.env.step():
                yield chunk
            loop_count += 1


if __name__ == "__main__":
    register_states = {
        StateType.REASONING: ReasoningState(),
        StateType.ACTION: ActionState(),
        StateType.ANSWER: AnswerState(),
    }
    brain = Brain(register_states[StateType.REASONING], register_states)

    async def main() -> None:
        async for chunk in brain.answer():
            print(chunk)

    asyncio.run(main())
