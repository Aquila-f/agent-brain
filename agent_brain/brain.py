from collections.abc import AsyncIterator
from enum import Enum

from agent_brain.memory import Memory, MessagesMemory
from agent_brain.models import AIMessage
from agent_brain.net import State, create_react_net
from agent_brain.tool import BaseTool

NET_MAP = {"ReAct": create_react_net()}
MEM_MAP = {"Messages": MessagesMemory}


class Brain:
    def __init__(
        self,
        net: str | dict[Enum, State],
        memory: str | Memory,
        tools: list[BaseTool],
    ) -> None:
        self.net = self._create_net(net)
        self.memory = self._create_memory(memory, tools)
        self.state = list(self.net.values())[0]
        self.loop_limit = 10

    async def reset(self) -> None:
        self.state = list(self.net.values())[0]
        self.memory.done = False
        self.memory.next_action = None
        self.loop_limit = 10
        await self.memory.set_goal("")  # reset memory content

    async def _step(self) -> AsyncIterator[str]:
        self.state.on_enter(self.memory)
        async for chunk in self.state.run(self.memory):
            yield AIMessage(type=type(self.state).__name__, message=chunk).model_dump_json()
        self.state.on_exit(self.memory)

        next_state_enum = await self.state.next_state(self.memory)
        self.state = self.net[next_state_enum]

    async def answer(self, task: str) -> AsyncIterator[str]:
        await self.reset()
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
