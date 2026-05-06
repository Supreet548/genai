import os
import json
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
    SystemMessage(content="""
You are a JSON generator.

STRICT RULES:
- Output MUST be raw JSON
- DO NOT wrap in markdown (no ```json)
- DO NOT add explanation
- DO NOT add any text before or after

If you include anything else, the output is invalid.

Schema:
{
  "concept": "string",
  "example": "string"
}
"""),
    HumanMessage(content="Explain recursion")
])

try:
    data = json.loads(response.content)
    print("Parsed JSON:", data)
except:
    print("Invalid JSON — retrying...")


