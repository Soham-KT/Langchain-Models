from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel


llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100
    }
)

model = ChatHuggingFace(llm=llm)

class Schema(BaseModel):
    reply: str
    sentiment: str

messages = [
    SystemMessage(content='Output should be strictly JSON type'),
    HumanMessage(content='how are you my guy')
]

structured_model = model.with_structured_output(Schema)

res = structured_model.invoke(messages)

print(res)
# print(res["reply"])
# print(res["sentiment"])