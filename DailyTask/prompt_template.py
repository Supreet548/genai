import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

# LLM

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_CHAT"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

#Prompt Template

prompt = PromptTemplate(
    input_variables=['topic'],
    template="""
You are a helpful teacher.
Explain {topic} in simple words. 
"""
)

final_prompt = prompt.format(
    topic = "embeddings"
)

response = llm.invoke(final_prompt)

print("\nLLM Response:\n")
print(response.content)