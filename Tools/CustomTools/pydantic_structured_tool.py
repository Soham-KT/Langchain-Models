from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
import json

class MultiplyInput(BaseModel):
    a: int = Field(description='The first number')
    b: int = Field(description='The second number')

def multiply(a: int, b: int) -> int:
    '''
    Multiplies two numbers
    a [int]: number 1
    b [int]: number 2

    returns a * b [int]
    '''
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name='multiply',
    description='Multiply 2 numbers',
    args_schema=MultiplyInput
)

res = multiply_tool.invoke({'a': 3, 'b': 5})
print(res)
print(multiply_tool.name)
print(multiply_tool.description)
print(json.dumps(multiply_tool.args, indent=2))
print(json.dumps(multiply_tool.args_schema.model_json_schema(), indent=2))