import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
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


text = """
Dogs are loyal animals and are commonly kept as pets.
They are known for companionship and protection.

Cats are independent animals and are also popular pets.
They require less maintenance and are quiet.

Cars are vehicles used for transportation.
They run on fuel or electricity.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

docs = [Document(page_content=chunk, metadata={"source": "pets_doc"}) for chunk in chunks]

vectorstore = FAISS.from_documents(docs, embeddings)




retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
query = input("Ask something: ")
results = retriever.invoke(query)


context = "\n".join([doc.page_content for doc in results])

response = llm.invoke(f"""
You are a helpful assistant.

Answer clearly using ONLY the context.

Context:
{context}

Question:
{query}

Give a short explanatory answer.
""")

print("\nRetrieved Chunks:\n")

for doc in results:
    print(doc.page_content)
    print("-" * 40)

print("\nFinal Answer:\n")
print(response.content)