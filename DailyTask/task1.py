import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_CHAT"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

response = llm.invoke([
    SystemMessage(content="You are a stand-up comedian. Be funny"),
    HumanMessage(content="Explain recursion")
])

print(response.content)



