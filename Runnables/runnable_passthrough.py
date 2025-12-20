from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt_1 = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template='explain the following joke: {joke}',
    input_variables=['joke']
)

model = ChatOllama(model='llama2')
parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt_1, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt_2, model, parser)
})
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

res = final_chain.invoke({'topic': 'Stock Market Crash'})

print(res['joke'])
print('\n\n--------------------------------------------')
print(res['explanation'])