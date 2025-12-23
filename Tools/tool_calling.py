from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
import json


# tool creation

@tool
def multiply(a: int, b: int) -> int:
    '''Given 2 numbers a and b, this tool returns their product'''
    return a * b


# tool binding

llm = ChatOllama(model='llama3.1')

llm_with_tools = llm.bind_tools([multiply])


# tool calling

query = HumanMessage('can you multiply 3 with 1000')
messages = [query]

res_tool_call = llm_with_tools.invoke(messages)
messages.append(res_tool_call)

# tool execution

tool_res = multiply.invoke(res_tool_call.tool_calls[0])
messages.append(tool_res)

res = llm_with_tools.invoke(messages).content
print(res)