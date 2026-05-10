import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_CHAT"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("MODEL_EMBED"),
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

loader = PyPDFLoader('Art11.pdf')
docs = loader.load()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer ONLY from the provided context.

Keep answers concise and easy to understand.

If the answer is not present in context,
say "I don't know based on the document."

Context:
{context}

Question:
{question}

Answer:
""")

parser = StrOutputParser()

chain = prompt | llm | parser

while True:
    query = input("\nAsk Question: ")

    if query.lower() == "exit":
        break

    retrieved_docs = retriever.invoke(query)

    context = "\n\n".join([
        doc.page_content
        for doc in retrieved_docs
    ])

    print("\nRetrieved Chunks:\n")

    for doc in retrieved_docs:
        print(doc.page_content[:300])
        print("\nMetadata:", doc.metadata)
        print("-" * 50)

    response = chain.invoke({
        "context": context,
        "question": query
    })

    print("\nAI Answer:\n")
    print(response)

    print("\nSources:")

    for doc in retrieved_docs:
        print(f"- Page {doc.metadata.get('page') + 1}")


