from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang='en')

query = 'goty 2025 nominees and winner'

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f'\n\n ---------------- Result {i+1} ---------------- \n\n')
    print(f'Content: \n{doc.page_content}...')