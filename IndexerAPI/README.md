curl -X POST "http://192.168.0.20:8082/ingest/file"   -H "Content-Type: application/json"   -d '{
    "user_id": "test_user",
    "org_id": "test_org",
    "s3_url": "http://192.168.0.20:9001/indexerapi/DSC03186.jpeg",
    "source": "image_test",
    "metadata": {
      "filename": "DSC03186.jpeg",
      "file_type": "jpeg",
      "uploaded_at": "2025-07-16T00:00:00Z"
    }
  }'

curl -X POST "http://192.168.0.20:8082/ingest/file"  -H "Content-Type: application/json"   -d '{
    "user_id": "test_user",
    "org_id": "test_org", 
    "s3_url": "http://192.168.0.20:9001/indexerapi/cube.PDF",
    "source": "minio_test",
    "metadata": {
      "filename": "cube.PDF",
      "file_type": "pdf",
      "uploaded_at": "2025-07-15T20:23:00Z"
    }
  }'

curl -X POST "http://192.168.0.20:8082/ingest/file" -H "Content-Type: application/json"   -d '{
    "user_id": "test_user",
    "org_id": "test_org", 
    "s3_url": "http://192.168.0.20:9001/indexerapi/Pokemon.csv",
    "source": "csv_test",
    "metadata": {
      "filename": "Pokemon.csv",
      "file_type": "csv",
      "uploaded_at": "2025-07-16T00:00:00Z"
    }
  }'