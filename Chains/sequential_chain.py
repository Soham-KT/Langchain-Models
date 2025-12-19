from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template='Generate a 5 liner summary on the following text : \n {text}',
    input_variables=['text']
)
model = ChatOllama(model='llama2')

parser = StrOutputParser()

chain = prompt_1 | model | parser | prompt_2 | model | parser

res = chain.invoke({'topic': 'AI bubble'})

print(res)

chain.get_graph().print_ascii()