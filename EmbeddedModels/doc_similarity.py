from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docs = [
    'dont turn left at the crossroads',
    'too bad for you, i have a pickaxe',
    'he was mining bedrock for 6 years'
]

query = 'how long was he mining for?'

doc_embedding = embedding.embed_documents(docs)
query_embedding = embedding.embed_query(query)

res = cosine_similarity([query_embedding], doc_embedding)[0]
index, score = sorted(list(enumerate(res)), key=lambda x:x[1])[-1]

print(query)
print(docs[index])
print(f'Similarity score is: {score}')