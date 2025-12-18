from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel


model = ChatOllama(model='llama2')

class Schema(BaseModel):
    reply: str
    sentiment: str

structured_model = model.with_structured_output(Schema)

res = structured_model.invoke('how are you my guy')

print(res)
print(res.reply)
print(res.sentiment)