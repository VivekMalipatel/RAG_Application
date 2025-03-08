import requests
import json

WEBHOOK_URL = "http://0.0.0.0:8000/api/v1/minio/webhook/"

#Resume.pdf
#ragas_papers.pdf

def test_minio_webhook_event():
    file_names = ["Resume.pdf", "ragas_papers.pdf"]
    
    for file_name in file_names:
        payload = {
            "EventName": "s3:ObjectCreated:CompleteMultipartUpload",
            "Key": f"userbucket/1234324/standard/uploads/{file_name}",
            "EventTime": "2023-01-01T00:00:00Z"
        }
        response = requests.post(
            WEBHOOK_URL,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 202, f"Unexpected status code: {response.status_code}"
        print(f"Webhook event test passed for {file_name}, response:", response.json())

if __name__ == "__main__":
    test_minio_webhook_event()
