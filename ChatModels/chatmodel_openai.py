from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.9)

res = model.invoke('what is the best pizza in the world?')

print(res)
print(res.content)