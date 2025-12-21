from langchain_chroma import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embeddings_model = OllamaEmbeddings(model='mxbai-embed-large')

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    collection_name='retreiver_store',
    persist_directory='./Retrievers/vector_store'
)

retriever = vector_store.as_retriever(search_kwargs={'k': 2})

res = retriever.invoke('what is chroma used for?')

for i, doc in enumerate(res):
    print(f'\n------ Result {i+1} ------\n')
    print(doc.page_content)