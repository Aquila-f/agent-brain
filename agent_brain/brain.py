import asyncio
from collections.abc import AsyncIterator

from agent_brain.memory import MessagesMemory
from agent_brain.thinking_net import create_react_net
from agent_brain.tool import BaseTool


class AddTwoNumbersTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="add_two_numbers",
            description="Add two numbers.",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."},
                },
                "required": ["a", "b"],
            },
        )

    async def execute(self, **kwargs) -> str:
        a = kwargs.get("a", 0)
        b = kwargs.get("b", 0)
        return f"tool result: {a} + {b} = {a + b}"


class Brain:
    def __init__(self, tools: list[BaseTool]) -> None:
        self.thinking_net = create_react_net()
        self.state = list(self.thinking_net.values())[0]
        self.memory = MessagesMemory(tools=tools)
        self.loop_limit = 10

    async def _step(self) -> AsyncIterator[str]:
        self.state.on_enter(self.memory)
        async for chunk in self.state.run(self.memory):
            yield chunk
        self.state.on_exit(self.memory)

        next_state_enum = await self.state.next_state(self.memory)
        self.state = self.thinking_net[next_state_enum]

    async def answer(self, task: str) -> AsyncIterator[str]:
        await self.memory.set_goal(task)

        while not self.memory.done and self.loop_limit > 0:
            async for chunk in self._step():
                yield chunk
            self.loop_limit -= 1


if __name__ == "__main__":
    brain = Brain(tools=[AddTwoNumbersTool()])
    task = "What is the value of 12345 + 67890 + 999999?"

    async def main() -> None:
        async for chunk in brain.answer(task):
            print(chunk, end="", flush=True)

    asyncio.run(main())
