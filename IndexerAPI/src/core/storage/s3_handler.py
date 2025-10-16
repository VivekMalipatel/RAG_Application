import os
import logging
import asyncio
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from core.config import settings

logger = logging.getLogger(__name__)

_global_s3_session = None
_global_s3_handler = None
_s3_handler_lock = asyncio.Lock()

class S3Handler:
    def __init__(self):
        self.endpoint_url = settings.MINIO_ENDPOINT_URL
        self.bucket_name = settings.S3_BUCKET_NAME
        self.aws_access_key_id = settings.AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY
        self.region_name = settings.AWS_REGION
        logger.info(
            f"Initializing S3 handler with endpoint: {self.endpoint_url}, bucket: {self.bucket_name}"
        )
        self.session = _get_global_s3_session()
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _get_s3_client(self):
        return self.session.create_client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        )

    async def initialize(self):
        async with self._init_lock:
            if self._initialized:
                return
            try:
                async with await self._get_s3_client() as client:
                    try:
                        await client.head_bucket(Bucket=self.bucket_name)
                        logger.info(f"Bucket {self.bucket_name} exists")
                    except ClientError as exc:
                        error_code = exc.response.get("Error", {}).get("Code")
                        if error_code == "404":
                            logger.info(f"Bucket {self.bucket_name} does not exist, creating it")
                            await client.create_bucket(Bucket=self.bucket_name)
                            logger.info(f"Created bucket {self.bucket_name}")
                        else:
                            logger.error(
                                f"Error checking bucket {self.bucket_name}: {str(exc)}"
                            )
                            raise
                self._initialized = True
            except Exception as exc:
                logger.error(f"Error initializing S3 handler: {str(exc)}")
                raise

    async def upload_bytes(self, data, object_key):
        if not self._initialized:
            await self.initialize()
        async with await self._get_s3_client() as client:
            await client.put_object(Bucket=self.bucket_name, Key=object_key, Body=data)
            logger.info(f"Successfully uploaded data to {object_key}")
            return True

    async def upload_string(self, string_data, object_key):
        if not self._initialized:
            await self.initialize()
        async with await self._get_s3_client() as client:
            await client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=string_data,
            )
            logger.info(f"Successfully uploaded string data to {object_key}")
            return True

    async def download_bytes(self, object_key):
        if not self._initialized:
            await self.initialize()
        async with await self._get_s3_client() as client:
            response = await client.get_object(Bucket=self.bucket_name, Key=object_key)
            async with response["Body"] as stream:
                data = await stream.read()
            return data

    async def download_string(self, object_key):
        data = await self.download_bytes(object_key)
        return data.decode("utf-8") if data else None

    async def delete_object(self, object_key):
        if not self._initialized:
            await self.initialize()
        async with await self._get_s3_client() as client:
            await client.delete_object(Bucket=self.bucket_name, Key=object_key)
            logger.info(f"Successfully deleted object {object_key}")
            return True

    async def list_objects(self, prefix=""):
        if not self._initialized:
            await self.initialize()
        async with await self._get_s3_client() as client:
            response = await client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
            )
        objects = []
        if "Contents" in response:
            objects = [item["Key"] for item in response["Contents"]]
        return objects

    async def delete_prefix(self, prefix: str):
        if not prefix:
            return True
        if not self._initialized:
            await self.initialize()
        try:
            async with await self._get_s3_client() as client:
                continuation_token = None
                deleted_any = False
                while True:
                    list_kwargs = {
                        "Bucket": self.bucket_name,
                        "Prefix": prefix,
                    }
                    if continuation_token:
                        list_kwargs["ContinuationToken"] = continuation_token
                    response = await client.list_objects_v2(**list_kwargs)
                    contents = response.get("Contents", [])
                    if not contents:
                        break
                    delete_payload = {
                        "Bucket": self.bucket_name,
                        "Delete": {
                            "Objects": [{"Key": item["Key"]} for item in contents],
                            "Quiet": True,
                        },
                    }
                    await client.delete_objects(**delete_payload)
                    deleted_any = True
                    continuation_token = response.get("NextContinuationToken")
                    if not response.get("IsTruncated") or not continuation_token:
                        break
                if deleted_any:
                    logger.info(f"Deleted objects with prefix {prefix}")
                else:
                    logger.info(f"No objects found for prefix {prefix}")
                return True
        except ClientError as exc:
            logger.error(f"Error deleting objects with prefix {prefix}: {str(exc)}")
            return False
        except Exception as exc:
            logger.error(f"Unexpected error deleting objects with prefix {prefix}: {str(exc)}")
            return False

    async def upload_file(self, local_file_path, object_key):
        if not self._initialized:
            await self.initialize()
        try:
            with open(local_file_path, "rb") as file:
                data = file.read()
            async with await self._get_s3_client() as client:
                await client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Body=data,
                )
            logger.info(
                f"Successfully uploaded file {local_file_path} to {object_key}"
            )
            return True
        except Exception as exc:
            logger.error(
                f"Error uploading file {local_file_path} to {object_key}: {str(exc)}"
            )
            return False

    async def download_file(self, object_key, local_file_path):
        if not self._initialized:
            await self.initialize()
        try:
            async with await self._get_s3_client() as client:
                response = await client.get_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                )
                async with response["Body"] as stream:
                    data = await stream.read()
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            with open(local_file_path, "wb") as file:
                file.write(data)
            logger.info(f"Successfully downloaded {object_key} to {local_file_path}")
            return True
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code == "NoSuchKey":
                logger.info(f"Object {object_key} does not exist in S3")
                return False
            logger.error(
                f"Error downloading {object_key} to {local_file_path}: {str(exc)}"
            )
            return False
        except Exception as exc:
            logger.error(
                f"Error downloading {object_key} to {local_file_path}: {str(exc)}"
            )
            return False

    async def object_exists(self, object_key):
        if not self._initialized:
            await self.initialize()
        try:
            async with await self._get_s3_client() as client:
                await client.head_object(Bucket=self.bucket_name, Key=object_key)
            return True
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code == "404":
                return False
            logger.error(f"Error checking if {object_key} exists: {str(exc)}")
            return False
        except Exception as exc:
            logger.error(f"Error checking if {object_key} exists: {str(exc)}")
            return False

def _get_global_s3_session():
    global _global_s3_session
    if _global_s3_session is None:
        _global_s3_session = get_session()
    return _global_s3_session

async def get_global_s3_handler():
    global _global_s3_handler
    async with _s3_handler_lock:
        if _global_s3_handler is None:
            _global_s3_handler = S3Handler()
            await _global_s3_handler.initialize()
        return _global_s3_handler

async def cleanup_global_s3_handler():
    global _global_s3_handler, _global_s3_session
    _global_s3_handler = None
    _global_s3_session = None

def build_document_s3_base_path(org_id: str, user_id: str, source: str, filename: str) -> str:
    base_filename = os.path.splitext(filename or "")[0]
    sanitized = "".join(c if c.isalnum() or c in ["-", "_"] else "_" for c in base_filename)
    if not sanitized:
        sanitized = "document"
    return f"{org_id}/{user_id}/{source}/{sanitized}"
