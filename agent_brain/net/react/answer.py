from collections.abc import AsyncIterator
from enum import Enum

from agent_brain.memory import Memory
from agent_brain.models import Message, Role
from agent_brain.utils import llm

from ..base import State
from .net import ReAct

SYSTEM_PROMPT = """
You are an AI assistant for answering user questions according to the tool use result.
"""
ANSWERING_MODEL = "gpt-4.1-mini"


async def get_messages(memory: "Memory") -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
        *await memory.dump(),
    ]


class AnswerState(State):
    async def run(self, memory: "Memory") -> AsyncIterator[str]:
        generated_response = ""
        messages = await get_messages(memory)
        async for chunk in llm.stream_response(model_name=ANSWERING_MODEL, messages=messages):
            if chunk:
                generated_response += chunk
                yield chunk

        await memory.update([Message(role=Role.ASSISTANT, content=generated_response)])

    async def next_state(self, memory: "Memory") -> Enum:
        memory.done = True
        return ReAct.ANSWER
