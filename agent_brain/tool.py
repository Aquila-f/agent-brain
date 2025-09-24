from abc import ABC, abstractmethod


class BaseTool(ABC):
    def __init__(self, name: str, description: str, input_schema: dict):
        self.name = name
        self.description = description
        self.input_schema = input_schema

    def to_dict(self) -> dict:
        return {
            self.name: {
                "description": self.description,
                "input_schema": self.input_schema,
            }
        }

    @abstractmethod
    async def execute(self, **kwargs) -> str: ...
