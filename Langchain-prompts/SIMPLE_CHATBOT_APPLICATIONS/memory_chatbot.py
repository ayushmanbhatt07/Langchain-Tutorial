
# Import Hugging Face LLM wrapper and chat model wrapper from LangChain
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Import message classes used by chat models
# These create structured conversations instead of plain strings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

# Used to load environment variables from .env file
# (for example: HUGGINGFACEHUB_API_TOKEN)
from dotenv import load_dotenv


# Loads all variables from .env into the current environment
# Without this, your API key may not be accessible
load_dotenv()


# -----------------------------------------------------------
# LLM INITIALIZATION
# -----------------------------------------------------------

# HuggingFaceEndpoint connects to a model hosted on Hugging Face Inference API.
#
# repo_id:
#     Specifies which model to use.
#
# max_new_tokens:
#     Maximum number of tokens the model can generate in one response.
#     Larger value = longer responses.
#
# temperature:
#     Controls randomness.
#
#     0.0 -> deterministic, focused answers
#     0.5 -> balanced
#     1.0 -> creative/random
#
# Since this is a chatbot, temperature=1.0 makes responses more conversational.
llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    max_new_tokens=300,
    temperature=1.0
)


# -----------------------------------------------------------
# CHAT MODEL WRAPPER
# -----------------------------------------------------------

# ChatHuggingFace converts the LLM into a chat model.
#
# Why needed?
#
# HuggingFaceEndpoint alone mainly works with prompts.
#
# ChatHuggingFace allows us to use:
#
# SystemMessage
# HumanMessage
# AIMessage
#
# which is the preferred format for modern chat applications.
model = ChatHuggingFace(llm=llm)


# -----------------------------------------------------------
# CONVERSATION MEMORY
# -----------------------------------------------------------

# This list stores the entire conversation.
#
# Why use a list?
#
# Chat models expect a sequence of messages:
#
# [
#   SystemMessage(...),
#   HumanMessage(...),
#   AIMessage(...),
#   HumanMessage(...)
# ]
#
# Every new message gets appended here.
messages = []


# -----------------------------------------------------------
# SYSTEM MESSAGE
# -----------------------------------------------------------

# SystemMessage defines the AI's behaviour.
#
# Think of it as permanent instructions.
#
# It is usually added ONCE at the beginning.
#
# Examples:
#
# "You are a coding tutor."
# "You are a friendly assistant."
# "Answer only in Hindi."
#
# The model sees this before every conversation turn.
messages.append(
    SystemMessage(
        content='You are chill guy assistant and my friend'
    )
)


# -----------------------------------------------------------
# CHAT LOOP
# -----------------------------------------------------------

# Infinite loop keeps the chatbot running
# until user types "exit".
while True:

    # Take user input from terminal
    user_input = input('You: ')

    # Exit condition
    if user_input == 'exit':
        break


    # -------------------------------------------------------
    # HUMAN MESSAGE
    # -------------------------------------------------------

    # Store user's message in conversation history.
    #
    # Why store it?
    #
    # So the model remembers previous context.
    #
    # Example:
    #
    # User: My name is Ayushman
    # AI: Nice to meet you
    # User: What's my name?
    #
    # Without storing previous messages,
    # the model would forget the name.
    messages.append(
        HumanMessage(content=user_input)
    )


    # -------------------------------------------------------
    # MODEL INVOCATION
    # -------------------------------------------------------

    # invoke() sends the ENTIRE conversation history
    # to the model.
    #
    # Not just the latest user message.
    #
    # Example:
    #
    # [
    #   SystemMessage(...),
    #   HumanMessage(...),
    #   AIMessage(...),
    #   HumanMessage(...)
    # ]
    #
    # This is how conversational memory works.
    response = model.invoke(messages)


    # response is usually an AIMessage object.
    #
    # response.content contains only the text generated
    # by the model.
    print("AI: ", response.content)


    # -------------------------------------------------------
    # AI MESSAGE
    # -------------------------------------------------------

    # Save model response into chat history.
    #
    # Why?
    #
    # Future responses should know what the assistant
    # already said.
    #
    # Example:
    #
    # User: Tell me a joke
    # AI: Joke...
    # User: Explain it
    #
    # The AI needs access to its previous reply.
    messages.append(
        AIMessage(content=response.content)
    )


# -----------------------------------------------------------
# FINAL CHAT HISTORY
# -----------------------------------------------------------

# Prints complete conversation after exiting.
#
# Useful for debugging and understanding how
# LangChain stores messages internally.
print(messages)

