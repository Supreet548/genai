import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# LLM
llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_CHAT"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("MODEL_EMBED"),
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Documents
docs = [
    Document(page_content="Dogs are loyal animals and are commonly kept as pets."),
    Document(page_content="Cats are independent animals and are popular pets."),
    Document(page_content="Cars run on fuel.")
]

# Vector store
vectorstore = FAISS.from_documents(docs, embeddings)


# Query
query = "Which animals are good pets?"

retrieved_docs = vectorstore.similarity_search(query, k=2)

context = "\n".join([doc.page_content for doc in retrieved_docs])

# Final LLM call
response = llm.invoke(f"""
Answer the question using the context below.

- Be concise but informative
- Include reasoning from context
- Do not add external knowledge

Context:
{context}

Question:
{query}
""")


print("Retrieved docs:")
for doc in retrieved_docs:
    print(doc.page_content)

print(response.content)