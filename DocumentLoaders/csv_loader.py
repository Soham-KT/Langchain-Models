from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='./DocumentLoaders/realistic_restaurant_reviews.csv')

docs = loader.load()

print(len(docs))
print(docs[0])