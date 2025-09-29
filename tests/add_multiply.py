import asyncio

from agent_brain import BaseTool, Brain


class AddTwoNumbersTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="add_two_numbers",
            description="Add two numbers.",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "float", "description": "The first float."},
                    "b": {"type": "float", "description": "The second float."},
                },
                "required": ["a", "b"],
            },
        )

    async def execute(self, **kwargs) -> str:
        a = kwargs.get("a", 0.0)
        b = kwargs.get("b", 0.0)
        return f"tool result: {a} + {b} = {a + b}"


class MultiplyTwoNumbersTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(
            name="multiply_two_numbers",
            description="Multiply two numbers.",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "float", "description": "The first float."},
                    "b": {"type": "float", "description": "The second float."},
                },
                "required": ["a", "b"],
            },
        )

    async def execute(self, **kwargs) -> str:
        a = kwargs.get("a", 0.0)
        b = kwargs.get("b", 0.0)
        return f"tool result: {a} * {b} = {a * b}"


brain = Brain(net="ReAct", memory="Messages", tools=[AddTwoNumbersTool(), MultiplyTwoNumbersTool()])


if __name__ == "__main__":
    task = "What is the value of 12345 + 67890 + 999999?"

    async def main() -> None:
        async for chunk in brain.answer(task):
            print(chunk, end="", flush=True)

    asyncio.run(main())
