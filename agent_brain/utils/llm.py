from openai import AsyncOpenAI

from agent_brain.models import Message, Role

client = AsyncOpenAI()


async def stream_response(model_name: str, messages: list[Message]):
    stream = await client.chat.completions.create(model=model_name, messages=messages, stream=True)

    async for chunk in stream:
        yield chunk.choices[0].delta.content


if __name__ == "__main__":
    print("Streaming response:")
    import asyncio

    async def main():
        messages = [
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="Hello! How are you?"),
        ]
        async for chunk in stream_response("gpt-4.1-nano", messages):
            print(chunk, end="", flush=True)

    asyncio.run(main())
