import streamlit as st
import os
import asyncio

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from retail_plugin import RetailPlugin
from retial_service import RetailService
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings)
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

# Get info from environment
load_dotenv('../.env')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USER = os.getenv('NEO4J_USERNAME', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
service_id = "contract_search"

# Streamlit app configuration
st.set_page_config(layout="wide")
st.title("📄 Agent for Retail Analytics")

# Initialize Kernel, Chat History, and Settings in Session State
if 'semantic_kernel' not in st.session_state:
    # Initialize the kernel
    kernel = Kernel()

    # Add the Contract Search plugin to the kernel
    retail_analytics_neo4j = RetailService(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    kernel.add_plugin(RetailPlugin(retail_service=retail_analytics_neo4j), plugin_name="retail_analytics")

    # Add the OpenAI chat completion service to the Kernel
    kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o", api_key=OPENAI_KEY, service_id=service_id))

    # Enable automatic function calling
    settings: OpenAIChatPromptExecutionSettings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto(filters={"included_plugins": ["retail_analytics"]})

    # Create a history of the conversation
    st.session_state.semantic_kernel = kernel
    st.session_state.kernel_settings = settings
    st.session_state.chat_history = ChatHistory()
    st.session_state.ui_chat_history = []  # For displaying messages in UI

if 'user_question' not in st.session_state:
    st.session_state.user_question = ""  # To retain the input text value


# Function to get a response from the agent
async def get_agent_response(user_input):
    kernel = st.session_state.semantic_kernel
    history = st.session_state.chat_history
    settings = st.session_state.kernel_settings

    # Add user input to the chat history
    history.add_user_message(user_input)
    st.session_state.ui_chat_history.append({"role": "user", "content": user_input})


    retry_attempts = 3
    for attempt in range(retry_attempts):

    # Get the response from the agent
        try:
            chat_completion: OpenAIChatCompletion = kernel.get_service(type=ChatCompletionClientBase)
            
            result = (await chat_completion.get_chat_message_contents(
                chat_history=history,
                settings=settings,
                kernel=kernel,
                #arguments=KernelArguments(),
            ))[0]
            

            # Add the agent's reply to the chat history
            history.add_message(result)
            st.session_state.ui_chat_history.append({"role": "agent", "content": str(result)})
            
            return # Exit after successful response
        
        except Exception as e:
            if attempt < retry_attempts - 1:
                #st.warning(f"Connection error: {str(e)}. Retrying ...")
                time.sleep(0.2)  # Wait before retrying
            else:
                print ("get_agent_response-error" + str(e))
                st.session_state.ui_chat_history.append({"role": "agent", "content": f"Error: {str(e)}"})

# UI for Q&A interaction
st.subheader("Chat with Your Agent")

# Container for chat history
chat_placeholder = st.container()

# Function to display the chat history
def display_chat():
    with chat_placeholder:
        for chat in st.session_state.ui_chat_history:
            if chat['role'] == 'user':
                st.markdown(f"**User:** {chat['content']}")
            else:
                st.markdown(f"**Agent:** {chat['content']}")


# Create a form for the input so that pressing Enter triggers the form submission
with st.form(key="user_input_form"):
    #user_question = st.text_input("Enter your question:", key="user_question")
    user_question = st.text_input("Enter your question:", value=st.session_state.user_question, key="user_question_")
    send_button = st.form_submit_button("Send")

# Execute the response action when the user clicks "Send" or presses Enter
if send_button and user_question.strip() != "":
    # Retain the value of user input in session state to display it in the input box
    st.session_state.user_question = user_question
    # Run the agent response asynchronously in a blocking way
    print(f"Questions: {user_question} ")
    print("---------------------------")
    asyncio.run(get_agent_response(st.session_state.user_question))
    print("=============================\n\n")
    # Clear the session state's question value after submission
    st.session_state.user_question = ""
    display_chat()
    
elif send_button:
    st.error("Please enter a question before sending.")

# Input for user question
#user_question = st.text_input("Enter your question:")



# Button to send the question
#if st.button("Send"):
#    if user_question.strip() != "":
        # Run the agent response asynchronously
#        asyncio.run(get_agent_response(user_question))
#        # Update chat history in UI
#        #display_chat()
#        st.rerun()
#    else:
#        st.error("Please enter a question before sending.")

# Footer
st.markdown("---")
