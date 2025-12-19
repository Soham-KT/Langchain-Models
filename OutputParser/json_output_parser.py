from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

model = ChatOllama(model='llama2')
parser = JsonOutputParser()

template = PromptTemplate(
    template='give me the name, age and city of a fictional person \n {format_inst}',
    input_variables=[],
    partial_variables={'format_inst': parser.get_format_instructions()}
)

chain = template | model | parser

final_res = chain.invoke({})

print(final_res)