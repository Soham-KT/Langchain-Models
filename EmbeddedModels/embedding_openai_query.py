from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='test-embedding-3-large', dimensions=32)

docs = [
    'hello everybody my name is markiplier',
    'top o the mording to ya ladies',
    'wsup gamers'
]

res = embedding.embed_documents(docs)
# res = embedding.embed_query('hello everybody my name is markiplier')

print(str(res))