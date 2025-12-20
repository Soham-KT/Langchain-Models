from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template='write a summary for this poem:\n{poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

model = ChatOllama(model='llama2')

loader = TextLoader('./DocumentLoaders/ai_bubble_burst.txt')

docs = loader.load()

chain = prompt | model | parser

res = chain.invoke({'poem': docs[0].page_content})

print(res)

# print(docs)
# print('\n\n----------------------')
# print(type(docs))
# print('\n\n----------------------')
# print(docs[0])
# print('\n\n----------------------')
# print(type(docs[0]))
# print('\n\n----------------------')
# print(docs[0].page_content)
# print('\n\n----------------------')
# print(docs[0].metadata)