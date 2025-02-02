from app.core.storage_bin.minio import MinIOClient
import tempfile
import os

def test_minio_connection():
    """Test basic MinIO connection"""
    try:
        client = MinIOClient()
        # Test connection by listing buckets
        buckets = client.list_buckets()
        print("✓ MinIO connection successful")
        print(f"  Current buckets: {buckets}")
        return client
    except Exception as e:
        print(f"✗ MinIO connection failed: {str(e)}")
        return None

def test_user_bucket_creation(client, test_user_id="test-user"):
    """Test creating a user-specific bucket"""
    try:
        success = client.create_user_bucket(test_user_id)
        if success:
            print(f"✓ User bucket '{test_user_id}' created successfully")
        else:
            print(f"✗ Failed to create user bucket '{test_user_id}'")
    except Exception as e:
        print(f"✗ User bucket creation error: {str(e)}")

def create_test_file():
    """Create a temporary test file"""
    temp_dir = tempfile.gettempdir()
    test_file_path = os.path.join(temp_dir, "test.txt")
    with open(test_file_path, "w") as f:
        f.write("This is a test file for MinIO storage")
    return test_file_path

def run_tests():
    """Execute all tests"""
    print("Starting MinIO Tests...")
    print("-" * 50)
    
    # Test 1: Connection
    client = test_minio_connection()
    if not client:
        return
    
    # Test 2: User Bucket Creation
    test_user_bucket_creation(client)

    print("-" * 50)
    print("Tests completed")

if __name__ == "__main__":
    run_tests()
