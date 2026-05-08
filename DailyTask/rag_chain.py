import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
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

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

prompt = PromptTemplate(
    input_variables=["context","question"],
    template="""
You are a helpful AI assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}

Answer:


If you do not know the answer, say:
"I don't know."

"""

)

parser = StrOutputParser()

rag_chain = (
    {"context":retriever|format_docs,
     "question":RunnablePassthrough()
    }
    |prompt
    |llm
    |parser

)

query = input("Ask something: ")

response = rag_chain.invoke(query)

print("\nFinal Answer:\n")
print(response)
