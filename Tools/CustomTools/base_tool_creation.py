from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import json

class MultiplyInput(BaseModel):
    a: int = Field(description='The first number')
    b: int = Field(description='The second number')

class MultiplyTool(BaseTool):
    name: str = 'multiply'
    description: str = 'Multiply two numbers'
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b
    
multiply_tool = MultiplyTool()
res = multiply_tool.invoke({'a': 3, 'b': 5})
print(res)
print(multiply_tool.name)
print(multiply_tool.description)
print(json.dumps(multiply_tool.args, indent=2))
print(json.dumps(multiply_tool.args_schema.model_json_schema(), indent=2))