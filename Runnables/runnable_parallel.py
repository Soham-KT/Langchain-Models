from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

class Post(BaseModel):
    post: str = Field('The string containing the post')

prompt_1 = PromptTemplate(
    template='generate a tweet about the following topic: {topic}',
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template='generate a linkedin post about the following topic: {topic}',
    input_variables=['topic']
)

model = ChatOllama(model='deepseek-r1')
structured_model = model.with_structured_output(Post)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt_1, model, parser),
    'linkedin': RunnableSequence(prompt_2, model, parser)
})

res = parallel_chain.invoke({'topic': 'AI'})

print(res['tweet'])
print('\n\n-------------------------')
print(res['linkedin'])