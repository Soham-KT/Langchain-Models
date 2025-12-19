from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of feedback')

prompt_1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into either positive or negative: \n {feedback}',
    input_variables=['feedback']
)

prompt_2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback: \n {feedback}',
    input_variables=['feedback']
)

prompt_3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback: \n {feedback}',
    input_variables=['feedback']
)

model = ChatOllama(model='llama2')

parser = StrOutputParser()

structured_model = model.with_structured_output(Feedback)

classifier_chain = prompt_1 | structured_model

branch_chain = RunnableBranch(
    (lambda x: x.sentiment=='positive', prompt_2 | model | parser),
    (lambda x: x.sentiment=='negative', prompt_3 | model | parser),
    RunnableLambda(lambda x: 'could not find sentiment')
)

chain = classifier_chain | branch_chain

res = chain.invoke({'feedback': 'This is a beautiful documentary'})
print(res)

chain.get_graph().print_ascii()