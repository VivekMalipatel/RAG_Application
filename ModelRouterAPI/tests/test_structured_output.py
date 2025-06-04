import openai
import os
from pydantic import BaseModel

API_KEY = os.getenv("MODEL_ROUTER_API_KEY", "test-key")
BASE_URL = os.getenv("MODEL_ROUTER_BASE_URL", "http://localhost:8000/v1")

client = openai.OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

class Step(BaseModel):
    explanation: str
    output: str

class MathReasoning(BaseModel):
    steps: list[Step]
    final_answer: str

print(f"Attempting to connect to API at: {BASE_URL}")

try:
    print("Sending structured output chat completion request...")
    completion = client.beta.chat.completions.parse(
        model="qwen3:1.7b-fp16",
        messages=[
            {"role": "system", "content": "You are a helpful math tutor. Guide the user through the solution step by step."},
            {"role": "user", "content": "how can I solve 8x + 7 = -23"}
        ],
        response_format=MathReasoning,
    )

    math_reasoning = completion.choices[0].message.parsed

    print(type(math_reasoning))

    print("\nStructured Output Response:")
    print(f"Final Answer: {math_reasoning.final_answer}")
    print("\nSolution Steps:")
    for i, step in enumerate(math_reasoning.steps, 1):
        print(f"Step {i}:")
        print(f"  Explanation: {step.explanation}")
        print(f"  Output: {step.output}")

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
