from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

docs = [
    'dont turn left at the crossroads',
    'too bad for you, i have a pickaxe',
    'he was mining bedrock for 6 years'
]

# text = 'dont turn left at the crossroads'

# vect = embedding.embed_query(text)
vect = embedding.embed_documents(docs)

print(str(vect))