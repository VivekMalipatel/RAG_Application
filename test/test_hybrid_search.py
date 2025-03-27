import requests
import json

# Define the API endpoint for hybrid search
HYBRID_SEARCH_URL = "http://0.0.0.0:8000/api/v1/query/ask"

def test_hybrid_search():
    """Test the hybrid search query processing."""
    
    # Define test payload
    query = input("Please enter your query: ")
    top_k = int(input("Please enter the number of search results to return: "))
    payload = {
        "user_id": "1234324",
        "query": query,
        "top_k": top_k
    }

    # Send POST request to the API
    response = requests.post(
        HYBRID_SEARCH_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )

    # Validate response
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    response_data = response.json()

    # Ensure the response contains the expected fields
    assert "answer" in response_data, "Response missing 'answer' field"
    assert "sources" in response_data, "Response missing 'sources' field"
    assert isinstance(response_data["sources"], list), "'sources' should be a list"

    print("✅ Hybrid search test passed!")
    print("🔹 Answer:", response_data["answer"])
    #print("🔹 Sources:", response_data["sources"])

# Run the test
if __name__ == "__main__":
    test_hybrid_search()