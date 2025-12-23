from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

template = '''
You are a helpful assistant. Answer the following question : {question}
Based on the following context : {context}
'''

prompt = PromptTemplate(
    template=template,
    input_variables=['question', 'context']
)

model = ChatOllama(model='deepseek-r1')

parser = StrOutputParser()

question = input('Enter your question: ')

search_tool = DuckDuckGoSearchRun()
context = search_tool.invoke(question)

chain = prompt | model | parser

res = chain.invoke({'question': question, 'context': context})

print(f'\nResult: {res}')