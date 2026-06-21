from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

llm=HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    max_new_tokens=300,
    temperature=1.0
)
model=ChatHuggingFace(llm=llm)

user_input=st.text_input("Enter a topic")

if user_input:
    prompt=PromptTemplate(
        template='Explain {topic} in very simple words',
        input_variables=['topic'],
        validate_template=True # validates the template so if error code is displayed here at localhost only
    )
    st.write("Prompt created")
    st.write(prompt)
    formatted_prompt = prompt.format(topic=user_input)
    response = model.invoke(formatted_prompt)
    st.write(response.content)