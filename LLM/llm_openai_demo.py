from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm_openai = OpenAI(model='gpt-3.5-turbo-instruct')

res = llm_openai.invoke('what is the value of pi upto 10 digits')

print(res)