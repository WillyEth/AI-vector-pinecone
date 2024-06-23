import os

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    # loader = PyPDFLoader("William_Kroll_Draft.pdf")
    loader = TextLoader("WilliamKroll_Draft.txt")
    document = loader.load()

    if not isinstance(document, list):
        document = [document]
    # # Join the list of pages into a single string 1536
    print(document)
    print("splitting..")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(texts)
    print(f"{len(texts)} chunks")

    embeddings = OpenAIEmbeddings()

    print("ingesting...")

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["PINECONE_INDEX_NAME"])
    print("done")
