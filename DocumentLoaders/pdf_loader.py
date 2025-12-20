from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('./DocumentLoaders/dl-curriculum.pdf')

docs = loader.load()

print(docs)
print('\n\n-------------------------')
print(len(docs))
print('\n\n-------------------------')
print(docs[0].page_content)
print('\n\n-------------------------')
print(docs[0].metadata)