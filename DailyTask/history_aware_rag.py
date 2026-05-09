import os
import time

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# =========================================
# LLM
# =========================================

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_CHAT"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# =========================================
# EMBEDDINGS
# =========================================

embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("MODEL_EMBED"),
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# =========================================
# DATA
# =========================================

text = """
Dogs are loyal animals and are commonly kept as pets.
They are known for companionship and protection.

Cats are independent animals and are popular pets.
They require less maintenance and are quiet.

Python is used for AI and web development.
"""

# =========================================
# CHUNKING
# =========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

docs = [
    Document(page_content=chunk)
    for chunk in chunks
]

# =========================================
# VECTOR STORE
# =========================================

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# =========================================
# PROMPT
# =========================================

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

chain = prompt | llm | parser

# =========================================
# SAFE RETRIEVAL
# =========================================

def safe_retrieve(query, retriever):

    for attempt in range(3):

        try:

            print("\nSearching...\n")

            return retriever.invoke(query)

        except Exception as e:

            print(f"\nAttempt {attempt + 1} failed.")
            print("Error:", e)

            time.sleep(2)

    return []

# =========================================
# CHAT LOOP
# =========================================

chat_history = []

while True:

    query = input("\nYou: ")

    if query.lower() == "exit":
        print("\nGoodbye!")
        break

    # =====================================
    # CREATE FULL QUERY
    # =====================================

    full_query = f"""
Chat History:
{chat_history}

Current Question:
{query}
"""

    # =====================================
    # RETRIEVE DOCS
    # =====================================

    retrieved_docs = safe_retrieve(
        full_query,
        retriever
    )

    if not retrieved_docs:

        print("\nAI: Retrieval failed.")
        continue

    # =====================================
    # CONTEXT
    # =====================================

    context = "\n".join([
        doc.page_content
        for doc in retrieved_docs
    ])

    # =====================================
    # DEBUG RETRIEVAL
    # =====================================

    print("\nRetrieved Context:\n")

    for doc in retrieved_docs:
        print("-", doc.page_content)

    # =====================================
    # GENERATE RESPONSE
    # =====================================

    try:

        response = chain.invoke({
            "chat_history": chat_history,
            "context": context,
            "question": query
        })

        print("\nAI:", response)

        # =================================
        # SAVE HISTORY
        # =================================

        chat_history.append(f"User: {query}")
        chat_history.append(f"AI: {response}")

    except Exception as e:

        print("\nGeneration Error:")
        print(e)