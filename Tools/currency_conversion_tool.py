from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from langchain_community.tools import tool
from typing import Annotated
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()

exchange_rate_api = os.getenv('EXCHANGE_RATE_API_KEY')

@tool
def get_conversion_factor(base_curr: str, target_curr: str) -> float:
    '''
    This function fetches the currency conversion factor between a given base currency and a target currency
    '''

    url = f'https://v6.exchangerate-api.com/v6/{exchange_rate_api}/pair/{base_curr}/{target_curr}'

    response = requests.get(url)

    return response.json()

@tool
def convert(base_curr_val: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    '''
    Given a currency conversion rate, this function calculates the target currency value from the given base currency value
    '''

    return base_curr_val * conversion_rate


llm = ChatOllama(model='llama3.1')
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion factor between USD and INR, based on that can you convert 10 USD to INR')]

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    if tool_call['name'] == 'get_conversion_factor':
        tool_msg_1 = get_conversion_factor.invoke(tool_call)
        conversion_rate = json.loads(tool_msg_1.content)['conversion_rate']
        messages.append(tool_msg_1)

    if tool_call['name'] == 'convert':
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_msg_2 = convert.invoke(tool_call)
        messages.append(tool_msg_2)

res = llm_with_tools.invoke(messages).content
print(res)