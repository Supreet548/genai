import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

text = """
Dogs are loyal animals and are commonly kept as pets.

Cats are independent animals and are popular pets.

Cars are used for transportation and run on fuel or electricity.

Python is a programming language used for AI and web development.
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

chunks = splitter.split_text(text)

docs = [Document(page_content=chunk)for chunk in chunks]

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Use ONLY the provided context and chat history.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
""")

parser = StrOutputParser()
chat_history = []

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break


    full_query = f"""
Chat History:
{chat_history}

Current Question:
{question}
"""

    retrieved_docs = retriever.invoke(full_query)

    context = "\n".join([
        doc.page_content for doc in retrieved_docs
    ])

    chain = prompt | llm | parser

    response = chain.invoke({
        "chat_history": chat_history,
        "context": context,
        "question": question
    })

    print("\nAI:", response)
    chat_history.append(f"User: {question}")
    chat_history.append(f"AI: {response}")