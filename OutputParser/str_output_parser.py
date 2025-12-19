from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model='llama2')

# 1st prompt -> detailed prompt
template_1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template_2 = PromptTemplate(
    template='write a 5 line summary on the following text.\n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template_1 | model | parser | template_2 | model | parser

res = chain.invoke({'topic': 'black hole'})

print(res)