import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import pprint

load_dotenv()

if __name__ == "__main__":
    print(" Retrieving... from PineCone Resume Model Data")

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    llm = ChatOpenAI()

    query = "Summarize William Krolls Resume, short version"
    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input": query})
    print(f"Engineered Prompt from Hub: {query}")
    result_text = result['answer']
    formatted_text = '\n'.join([result_text[i:i + 200] for i in range(0, len(result_text), 200)])
    print(f"LLM Result:\n{formatted_text}")
