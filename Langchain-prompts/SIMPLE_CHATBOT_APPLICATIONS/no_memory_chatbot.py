from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    max_new_tokens=300,
    temperature=1.0
)
model=ChatHuggingFace(llm=llm)


while True:
    user_input=input('You: ')
    if user_input=='exit':
        break
    response=model.invoke(user_input)
    print("AI: ",response.content)