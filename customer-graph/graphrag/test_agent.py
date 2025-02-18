import os
import asyncio

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from semantic_kernel.contents.chat_history import ChatHistory
from RetailPlugin import RetailPlugin
from RetailService import RetailService
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings)
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments
import logging


logging.basicConfig(level=logging.INFO)

#get info from environment
load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
NEO4J_URI=os.getenv('NEO4J_URI')
NEO4J_USER=os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD=os.getenv('NEO4J_PASSWORD')
service_id = "retail_search"

# Initialize the kernel
kernel = Kernel()

# Add the Contract Search plugin to the kernel
retail_analysis_neo4j = RetailService(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
kernel.add_plugin(RetailPlugin(retail_service=retail_analysis_neo4j), plugin_name="retail_analysis")

# Add the OpenAI chat completion service to the Kernel
kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o-mini", api_key=OPENAI_KEY, service_id=service_id))

# Enable automatic function calling
settings: OpenAIChatPromptExecutionSettings = kernel.get_prompt_execution_settings_from_service_id(service_id=service_id)
settings.function_choice_behavior = FunctionChoiceBehavior.Auto(filters={"included_plugins": ["retail_analysis"]})


# Create a history of the conversation
history = ChatHistory()

async def basic_agent() :
    userInput = None
    while True:
        # Collect user input
        userInput = input("User > ")

        # Terminate the loop if the user says "exit"
        if userInput == "exit":
            break

        # Add user input to the history
        history.add_user_message(userInput)

        # 3. Get the response from the AI with automatic function calling
        chat_completion : OpenAIChatCompletion = kernel.get_service(type=ChatCompletionClientBase)
        result = (await chat_completion.get_chat_message_contents(
            chat_history=history,
            settings=settings,
            kernel=kernel,
            arguments=KernelArguments(),
        ))[0]

        # Print the results
        print("Assistant > " + str(result))
        print("=============================\n\n")

        # Add the message from the agent to the chat history
        history.add_message(result)

if __name__ == "__main__":
    
    asyncio.run(basic_agent())


    
