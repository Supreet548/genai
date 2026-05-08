import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

# LLM

llm = ChatGoogleGenerativeAI(
    model=os.getenv("MODEL_CHAT"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
You are an strict interviewer.

Explain {topic} in {language} in 3 lines
"""
)

parser = StrOutputParser()

chain = prompt | llm | parser

topic = input("Enter topic: ")
language=input("Enter language: ")
response = chain.invoke({
    "topic": topic,
    "language":language
})

print(response)