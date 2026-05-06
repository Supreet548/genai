#Embedding+similarity 

import os
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
#from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("MODEL_EMBED"),
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

text = "What is recursion?"

vector = embeddings.embed_query(text)

#embed_query() → for single query
#embed_documents() → for list of docs

print("Vector length:", len(vector))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

e1 = embeddings.embed_query("dog")
e2 = embeddings.embed_query("puppy")
e3 = embeddings.embed_query("car")

print("dog vs puppy:", cosine_similarity(e1, e2))
print("dog vs car:", cosine_similarity(e1, e3))

docs = [
    "Python is a programming language",
    "Dogs are loyal animals",
    "Cars run on fuel",
    "Cats are independent animals"
]

doc_embeddings = embeddings.embed_documents(docs)

query = "Which animals are good pets?" 

query_embedding = embeddings.embed_query(query)

scores = []

for i, emb in enumerate(doc_embeddings):
    score = cosine_similarity(query_embedding, emb)
    scores.append((docs[i], score))

scores.sort(key=lambda x: x[1], reverse=True)

for doc, score in scores:
    print(f"{doc} → {score:.4f}")
