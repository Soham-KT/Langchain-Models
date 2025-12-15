from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage
from typing import TypedDict


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100
    }
)

model = ChatHuggingFace(llm=llm)

class Schema(TypedDict):
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