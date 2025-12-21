from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path='./DocumentLoaders/dl-curriculum.pdf')
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

res = splitter.split_documents(docs)

print(res[0])
print('----------------------')
print(res[0].page_content)
print('----------------------')
print(res[0].metadata)
