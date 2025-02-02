import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OllamaClient:
    def __init__(self, ollama_url: str):
        self.ollama_url = ollama_url

    def check_and_pull_model(self, model: str):
        """Checks if the model exists in Ollama and pulls it if missing."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            response.raise_for_status()
            available_models = response.json().get("models", [])

            if any(model in m for m in available_models):
                print(f"Model '{model}' is already available.")
                return

            print(f"Model '{model}' not found. Pulling from Ollama library...")
            pull_response = requests.post(f"{self.ollama_url}/api/pull", json={"model": model, "stream": False})
            pull_response.raise_for_status()
            
            if pull_response.json().get("status") == "success":
                print(f"Model '{model}' successfully pulled!")
            else:
                print(f"Model pull failed: {pull_response.json().get('status')}")
        except requests.exceptions.RequestException as e:
            print(f"Error checking or pulling model: {str(e)}")

    def generate_response(self, prompt: str, model: str):
        """Generates a response using a local LLM via Ollama API."""
        self.check_and_pull_model(model)
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "No response received from LLM")
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"

    def generate_embedding(self, text: str, model: str, embedding_type: str):
        """Generate embeddings with flexible vector sizes based on embedding type."""
        self.check_and_pull_model(model)
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": model, "prompt": text}
            )
            response.raise_for_status()
            embedding = response.json().get("embedding", [])

            if embedding_type == "matryoshka":
                return embedding[:64], embedding[:128], embedding[:256]
            return embedding
        except requests.exceptions.RequestException as e:
            return f"Error generating embedding: {str(e)}"

if __name__ == "__main__":
    ollama_url = os.getenv("OLLAMA_URL")
    ollama_client = OllamaClient(ollama_url)