import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()
import os

print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    
)


model = ChatHuggingFace(llm=llm)
st.header('Research Tool')
user_input=st.text_input('Enter your prompt')

if st.button('Summarize'):
    result=model.invoke(user_input)
    st.write(result.content)