from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template='you must answer using only the text below. if the answer is not explicitly present, reply exactly with: "not mentioned". answer the following question : {question} \n from the following text:\n {text}',
    input_variables=['question', 'text']
)

parser = StrOutputParser()

model = ChatOllama(model='llama2')

url = 'https://www.amazon.in/Apple-2025-MacBook-Laptop-10%E2%80%91core/dp/B0FWD7JSX7/ref=sr_1_1_sspa?crid=UGK6QANYAYIX&dib=eyJ2IjoiMSJ9.NXG4biOkzXZ8ec2Ng71So-0s9RUSHCHBzz007-74Hvs-wS0Pn9yd1t8Z5Jhnq0YUKF-xry2P8DdJPYiui2woa7xUDpQKGeynOSFUL6pkx1dSoqmxB-lrgjsEVL6C2Yb6tageaHS1ZT3th5795QsN1YtyXBhW0jmyyTeeJzwq4s4fsdBAfDNypS8JHf7jJuePZv7tfnwy25leV6dmJ3UpU9tGqRi3-IBBCWVnfScZMA0.UyA20WjDYCUf55qqrBPwLqTphG4leFeMI-ZwdaLHX4o&dib_tag=se&keywords=macbook+pro+m4&qid=1766233097&sprefix=mabook+%2Caps%2C213&sr=8-1-spons&aref=r8QRflvGJK&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1'
loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

res = chain.invoke({'question': 'what product are we talking about?', 'text': docs[0].page_content})

print(res)