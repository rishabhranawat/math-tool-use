import json
import time
import os
import requests
import openai

# Set OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

def chat_completion_request(messages, model, functions=None, function_call=None):
    """
    Sends a request to the OpenAI ChatCompletion endpoint to generate a response.
    
    Args:
        messages (list): List of message objects for the chat.
        model (str): The model to use for the chat completion.
        functions (dict, optional): Additional tools or functions for the model to use.
        function_call (dict, optional): Specific function call instructions.

    Returns:
        response: The response object from the API, or an exception if an error occurs.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}",
    }
    
    json_data = {
        "model": model,
        "messages": messages,
    }

    if functions is not None:
        json_data["tools"] = functions

    if function_call is not None:
        json_data["function_call"] = function_call

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data
        )
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()  # Return JSON directly for easy access
    except requests.exceptions.RequestException as e:
        print("Error generating ChatCompletion response:", e)
        return {"error": str(e)}
