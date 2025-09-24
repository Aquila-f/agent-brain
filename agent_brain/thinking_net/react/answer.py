from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from agent_brain.models import Message, Role
from agent_brain.utils import llm

from ..base import State

if TYPE_CHECKING:
    from agent_brain.memory import MemoryEnv

SYSTEM_PROMPT = """
You are an AI assistant for answering user questions according to the tool use result.
"""
ANSERING_MODEL = "gpt-4.1"


async def get_messages(env: "MemoryEnv") -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
        *env.history,
    ]


class AnswerState(State):
    async def _run(self, env: "MemoryEnv") -> AsyncIterator[str]:
        generated_response = ""
        messages = await get_messages(env)
        async for chunk in llm.stream_response(model_name=ANSERING_MODEL, messages=messages):
            if chunk:
                generated_response += chunk
                yield chunk

        env.history.append(Message(role=Role.ASSISTANT, content=generated_response))

    async def _transition(self, env: "MemoryEnv") -> None:
        env.done = True
