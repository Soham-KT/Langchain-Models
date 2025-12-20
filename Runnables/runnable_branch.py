from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

prompt_1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template='summarize the following text: {text}',
    input_variables=['text']
)

model = ChatOllama(model='llama2')

parser = StrOutputParser()

report_gen_chain = RunnableSequence(prompt_1, model, parser)
report_sum_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt_2, model, parser)),
    RunnablePassthrough()
)
report_len_chain = RunnableParallel({
    'report': RunnablePassthrough(),
    'length': RunnableLambda(lambda x: len(x.split()))
})
final_chain = RunnableSequence(report_gen_chain, report_sum_chain, report_len_chain)

res = final_chain.invoke({'topic': 'black holes'})
print(res['report'])
print('\n\n ------------------------------------')
print(res['length'])