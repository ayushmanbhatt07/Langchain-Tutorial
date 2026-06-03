from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1"
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of India")
print(result.content)