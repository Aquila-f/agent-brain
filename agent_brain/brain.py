import asyncio
from collections.abc import AsyncGenerator

from agent_brain.memory import MemoryEnv
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
        return f"Sum: {a + b}"


class Brain:
    def __init__(self) -> None:
        self.thinking_net = create_react_net()
        self.env = MemoryEnv(states=self.thinking_net, tools=[AddTwoNumbersTool()])
        self.loop_limit = 10

    async def answer(self, task: str) -> AsyncGenerator[str]:
        self.env.start_task(task)
        loop_count = 0
        while not self.env.done and loop_count < self.loop_limit:
            async for chunk in self.env.step():
                yield chunk
            loop_count += 1


if __name__ == "__main__":
    brain = Brain()
    task = "What is the sum of 123 and 456?"

    async def main() -> None:
        async for chunk in brain.answer(task):
            print(chunk, end="", flush=True)

    asyncio.run(main())
