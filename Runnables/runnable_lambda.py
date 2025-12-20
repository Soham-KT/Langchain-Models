from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def word_count(text: str):
    return len(text.split())

prompt = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOllama(model='llama2')
parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'length': RunnableLambda(word_count)
    # 'length': RunnableLambda(lambda x: len(x.split())) -------> alternative
})
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

res = final_chain.invoke({'topic': 'Stock Market Crash'})

print(res['joke'])
print('\n\n--------------------------------------------')
print(res['length'])