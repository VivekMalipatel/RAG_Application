import openai
import os

# Replace with your actual API key if needed, or use a test key if authentication is set up that way
# Ensure the API key is valid for your ModelRouterAPI setup
API_KEY = os.getenv("MODEL_ROUTER_API_KEY", "test-key") 
# Replace with the actual base URL of your running ModelRouterAPI
BASE_URL = os.getenv("MODEL_ROUTER_BASE_URL", "http://localhost:8000/v1") 

# Initialize the OpenAI client to point to your local API
client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

print(f"Attempting to connect to API at: {BASE_URL}")

try:
    # Example using a model available in your ModelRouterAPI (e.g., llama2 or gpt-3.5-turbo if configured)
    print("Sending streaming chat completion request...")
    stream = client.chat.completions.create(
        model="meta-llama_Llama-3.1-8B-Instruct_Q8_0", # Or another model configured in your API
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain the concept of RAG in large language models in detail, step by step."}
        ],
        stream=True # Enable streaming
    )

    print("\nStreaming Chat Completion Response:")
    full_response = ""
    for chunk in stream:
        # Check if the chunk has content and print it
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True) # Print content immediately without newline
            full_response += content
        # Optional: Print other chunk details if needed for debugging
        # print(f"\nChunk received: {chunk}")

    print("\n\n--- End of Stream ---")
    print(f"\nFull reconstructed response:\n{full_response}")


except openai.APIConnectionError as e:
    print(f"\nConnection Error: Failed to connect to the API at {BASE_URL}.")
    print(f"Please ensure your ModelRouterAPI server is running and accessible at this URL.")
    print(f"Error details: {e}")
except openai.AuthenticationError as e:
     print(f"\nAuthentication Error: Check if the API key '{API_KEY}' is correct and valid.")
     print(f"Error details: {e}")
except openai.APIStatusError as e:
    print(f"\nAPI Status Error: The API returned an error status code.")
    print(f"Status Code: {e.status_code}")
    print(f"Response: {e.response}")
    print(f"Error details: {e}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")

