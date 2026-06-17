from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
import os
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of India")
print(result.content)
# print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))