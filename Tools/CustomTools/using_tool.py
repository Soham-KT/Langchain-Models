from langchain_core.tools import tool
import json

@tool
def multiply(a: int, b: int) -> int:
    '''
    Multiplies two numbers
    a [int]: number 1
    b [int]: number 2

    returns a * b [int]
    '''
    return a * b

res = multiply.invoke({'a': 3, 'b': 5})
print(res)
print(multiply.name)
print(multiply.description)
print(json.dumps(multiply.args, indent=2))
print(json.dumps(multiply.args_schema.model_json_schema(), indent=2))