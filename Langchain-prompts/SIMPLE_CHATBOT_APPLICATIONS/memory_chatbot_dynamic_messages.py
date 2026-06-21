from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    max_new_tokens=300,
    temperature=1.0
)
model=ChatHuggingFace(llm=llm)

