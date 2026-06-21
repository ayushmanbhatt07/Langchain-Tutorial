# Import Hugging Face LLM and chat wrapper
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Used to create prompt templates with roles
from langchain_core.prompts import ChatPromptTemplate

# Load API keys from .env file
from dotenv import load_dotenv
load_dotenv()


# Create the base LLM
llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',  # model name
    max_new_tokens=500,                          # max response length
    temperature=1.0                              # randomness/creativity
)

# Convert LLM into a chat model
model = ChatHuggingFace(llm=llm)


# ChatPromptTemplate helps create structured prompts
# Variables inside {} are filled dynamically later
chat_template = ChatPromptTemplate([
    ('system', 'You are an expert in {domain}'),
    ('human', 'explain me the {topic} in very detail')
])


while True:

    # Take user input
    user_input = input('Enter the topic you wanna know about: ')

    # Exit chatbot
    if user_input == 'exit':
        break


    # Fill template variables and create final prompt
    # invoke() replaces placeholders with actual values
    prompt = chat_template.invoke({
        'domain': 'explainable ai',
        'topic': 'Gradcam with its mathematics'
    })


    # Send generated prompt to model
    response = model.invoke(prompt)

    # response is an AIMessage object
    print(response.content)

