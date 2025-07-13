import os
import logging
import asyncio
from aiobotocore.session import get_session
from botocore.exceptions import ClientError
from config import settings

logger = logging.getLogger(__name__)

class S3Handler:
    def __init__(self):
        self.endpoint_url = settings.MINIO_ENDPOINT_URL
        self.bucket_name = settings.S3_BUCKET_NAME
        self.aws_access_key_id = settings.AWS_ACCESS_KEY_ID
        self.aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY
        self.region_name = settings.AWS_REGION
        
        logger.info(f"Initializing S3 handler with endpoint: {self.endpoint_url}, bucket: {self.bucket_name}")
        
        self.session = get_session()
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def _get_s3_client(self):
        return self.session.create_client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name
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
                    except ClientError as e:
                        error_code = e.response.get('Error', {}).get('Code')
                        if error_code == '404':
                            logger.info(f"Bucket {self.bucket_name} does not exist, creating it")
                            await client.create_bucket(Bucket=self.bucket_name)
                            logger.info(f"Created bucket {self.bucket_name}")
                        else:
                            logger.error(f"Error checking bucket {self.bucket_name}: {str(e)}")
                            raise
                
                self._initialized = True
            except Exception as e:
                logger.error(f"Error initializing S3 handler: {str(e)}")
                raise
    
    async def upload_bytes(self, data, object_key):
        if not self._initialized:
            await self.initialize()
        
        async with await self._get_s3_client() as client:
            await client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=data
            )
            logger.info(f"Successfully uploaded data to {object_key}")
            return True
    
    async def upload_string(self, string_data, object_key):
        if not self._initialized:
            await self.initialize()
        
        async with await self._get_s3_client() as client:
            await client.put_object(
                Bucket=self.bucket_name,
                Key=object_key,
                Body=string_data
            )
            logger.info(f"Successfully uploaded string data to {object_key}")
            return True
    
    async def download_bytes(self, object_key):
        if not self._initialized:
            await self.initialize()
        
        async with await self._get_s3_client() as client:
            response = await client.get_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            async with response['Body'] as stream:
                data = await stream.read()
            return data
    
    async def download_string(self, object_key):
        data = await self.download_bytes(object_key)
        return data.decode('utf-8') if data else None
    
    async def delete_object(self, object_key):
        """Delete an object from S3"""
        if not self._initialized:
            await self.initialize()
        
        async with await self._get_s3_client() as client:
            await client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            logger.info(f"Successfully deleted object {object_key}")
            return True
    
    async def list_objects(self, prefix=""):
        if not self._initialized:
            await self.initialize()
        
        async with await self._get_s3_client() as client:
            response = await client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
        
        objects = []
        if 'Contents' in response:
            objects = [obj['Key'] for obj in response['Contents']]
        return objects

    async def upload_file(self, local_file_path, object_key):
        if not self._initialized:
            await self.initialize()
        
        try:
            with open(local_file_path, 'rb') as file:
                data = file.read()
            
            async with await self._get_s3_client() as client:
                await client.put_object(
                    Bucket=self.bucket_name,
                    Key=object_key,
                    Body=data
                )
            logger.info(f"Successfully uploaded file {local_file_path} to {object_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading file {local_file_path} to {object_key}: {str(e)}")
            return False
    
    async def download_file(self, object_key, local_file_path):
        if not self._initialized:
            await self.initialize()
        
        try:
            async with await self._get_s3_client() as client:
                response = await client.get_object(
                    Bucket=self.bucket_name,
                    Key=object_key
                )
                async with response['Body'] as stream:
                    data = await stream.read()
            
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            with open(local_file_path, 'wb') as file:
                file.write(data)
            
            logger.info(f"Successfully downloaded {object_key} to {local_file_path}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == 'NoSuchKey':
                logger.info(f"Object {object_key} does not exist in S3")
                return False
            else:
                logger.error(f"Error downloading {object_key} to {local_file_path}: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Error downloading {object_key} to {local_file_path}: {str(e)}")
            return False
    
    async def object_exists(self, object_key):
        if not self._initialized:
            await self.initialize()
        
        try:
            async with await self._get_s3_client() as client:
                await client.head_object(
                    Bucket=self.bucket_name,
                    Key=object_key
                )
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code')
            if error_code == '404':
                return False
            else:
                logger.error(f"Error checking if {object_key} exists: {str(e)}")
                return False
        except Exception as e:
            logger.error(f"Error checking if {object_key} exists: {str(e)}")
            return False
