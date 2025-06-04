import openai
import os

API_KEY = os.getenv("MODEL_ROUTER_API_KEY", "test-key")
BASE_URL = os.getenv("MODEL_ROUTER_BASE_URL", "http://localhost:8000/v1")

client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country e.g. Bogotá, Colombia"
                }
            },
            "required": [
                "location"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}]

print(f"Attempting to connect to API at: {BASE_URL}")

try:
    print("Sending tool calling chat completion request...")
    completion = client.chat.completions.create(
        model="qwen3:1.7b-fp16",
        messages=[{"role": "user", "content": "What is the weather like in Paris today?"}],
        tools=tools
    )

    print("\nTool Calling Response:")
    message = completion.choices[0].message

    print(message)
    
    if message.tool_calls:
        print(f"Tool calls found: {len(message.tool_calls)}")
        for i, tool_call in enumerate(message.tool_calls, 1):
            print(f"\nTool Call {i}:")
            print(f"  ID: {tool_call.id}")
            print(f"  Type: {tool_call.type}")
            print(f"  Function Name: {tool_call.function.name}")
            print(f"  Function Arguments: {tool_call.function.arguments}")
    else:
        print("No tool calls found in response")
        print(f"Message content: {message.content}")

except openai.APIConnectionError as e:
    print(f"\nConnection Error: Failed to connect to the API at {BASE_URL}.")
    print(f"Please ensure your ModelRouterAPI server is running and accessible at this URL.")
    print(f"Error details: {e}")
except openai.AuthenticationError as e:
    print(f"\nAuthentication Error: Check if the API key '{API_KEY}' is correct and valid.")
    print(f"Error details: {e}")
except Exception as e:
    print(f"\nUnexpected Error: {e}")
    print(f"Error type: {type(e)}")
