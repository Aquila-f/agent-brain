import asyncio
from collections.abc import AsyncIterator
from enum import Enum

from agent_brain.memory import Memory, MessagesMemory
from agent_brain.thinking_net import create_react_net
from agent_brain.tool import BaseTool

from .thinking_net import State

NET_MAP = {"ReAct": create_react_net()}
MEM_MAP = {"Messages": MessagesMemory}


class Brain:
    def __init__(
        self,
        thinking_net: str | dict[Enum, State],
        memory_struct: str | Memory,
        tools: list[BaseTool],
    ) -> None:
        self.net = self._create_net(thinking_net)
        self.memory = self._create_memory(memory_struct, tools)
        self.state = list(self.net.values())[0]
        self.loop_limit = 10

    async def _step(self) -> AsyncIterator[str]:
        self.state.on_enter(self.memory)
        async for chunk in self.state.run(self.memory):
            yield chunk
        self.state.on_exit(self.memory)

        next_state_enum = await self.state.next_state(self.memory)
        self.state = self.net[next_state_enum]

    async def answer(self, task: str) -> AsyncIterator[str]:
        await self.memory.set_goal(task)

        while not self.memory.done and self.loop_limit > 0:
            async for chunk in self._step():
                yield chunk
            self.loop_limit -= 1

    def _create_net(self, thinking_net: str | dict[Enum, State]) -> dict[Enum, State]:
        if isinstance(thinking_net, str):
            if thinking_net not in NET_MAP:
                raise ValueError(f"Unknown thinking net: {thinking_net}")
            return NET_MAP[thinking_net]
        return thinking_net

    def _create_memory(self, memory_struct: str | Memory, tools: list[BaseTool]) -> Memory:
        if isinstance(memory_struct, str):
            if memory_struct not in MEM_MAP:
                raise ValueError(f"Unknown memory struct: {memory_struct}")
            return MEM_MAP[memory_struct](tools=tools)
        return memory_struct


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


if __name__ == "__main__":
    brain = Brain(thinking_net="ReAct", memory_struct="Messages", tools=[AddTwoNumbersTool()])
    task = "What is the value of 12345 + 67890 + 999999?"

    async def main() -> None:
        async for chunk in brain.answer(task):
            print(chunk, end="", flush=True)

    asyncio.run(main())
