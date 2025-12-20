from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

prompt_1 = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template='explain the following joke: {joke}',
    input_variables=['joke']
)

model = ChatOllama(model='deepseek-r1')

parser = StrOutputParser()

chain = RunnableSequence(prompt_1, model, parser, prompt_2, model, parser)

print(chain.invoke({'topic': 'AI'}))