from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader = TextLoader(file_path='./DocumentLoaders/ai_bubble_burst.txt')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(docs[0].page_content)

for chunk in chunks:
    print(chunk)
    print('\n\n-------------------\n\n')
