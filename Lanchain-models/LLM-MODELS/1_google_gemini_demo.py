from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
result=model.invoke("What is the capital of India")
print(os.getenv("GOOGLE_API_KEY"))
print(result)
