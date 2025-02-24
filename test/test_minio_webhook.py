import requests
import json

WEBHOOK_URL = "http://0.0.0.0:8000/api/v1/minio/webhook/"

def test_minio_webhook_event():
    payload = {
        "EventName": "s3:ObjectCreated:CompleteMultipartUpload",
        "Key": "userbucket/1234324/standard/uploads/ragas_papers.pdf",
        "EventTime": "2023-01-01T00:00:00Z"
    }
    response = requests.post(
        WEBHOOK_URL,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 202, f"Unexpected status code: {response.status_code}"
    print("Webhook event test passed, response:", response.json())

if __name__ == "__main__":
    test_minio_webhook_event()
