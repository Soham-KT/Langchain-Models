from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

model = ChatOllama(model='llama2')

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='Name of the city that person lives in')

template = PromptTemplate(
    template='generate the name, age and city of a fictional {place} person.',
    input_variables=['place']
)

structured_output = model.with_structured_output(Person)

chain = template | structured_output

res = chain.invoke({'place': 'indian'})

print(res)
