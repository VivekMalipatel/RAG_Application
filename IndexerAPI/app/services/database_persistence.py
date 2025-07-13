import os
import logging
import asyncio
import shutil
from typing import Optional
from core.storage.s3_handler import S3Handler
from config import settings

logger = logging.getLogger(__name__)

_global_db_persistence: Optional['DatabasePersistence'] = None

class DatabasePersistence:
    def __init__(self):
        self.s3_handler = S3Handler()
        self.s3_prefix = f"{settings.S3_BACKUP_PREFIX}database/"
        self.db_path = self._extract_db_path()
        
    def _extract_db_path(self) -> str:
        if settings.DB_URL and settings.DB_URL.startswith('sqlite'):
            db_path = settings.DB_URL.replace('sqlite:///', '').replace('sqlite+aiosqlite:///', '')
            if not db_path.startswith('/'):
                db_path = os.path.abspath(db_path)
            return db_path
        return "app.db"
        
    async def backup_to_s3(self):
        try:
            if not os.path.exists(self.db_path):
                logger.warning(f"Database file {self.db_path} does not exist, skipping backup")
                return False
                
            s3_key = f"{self.s3_prefix}app.db"
            success = await self.s3_handler.upload_file(self.db_path, s3_key)
            if success:
                logger.info("Successfully backed up database to S3")
            else:
                logger.error("Failed to backup database to S3")
            return success
            
        except Exception as e:
            logger.error(f"Error backing up database to S3: {str(e)}")
            return False
    
    async def restore_from_s3(self):
        try:
            if os.path.exists(self.db_path):
                backup_path = f"{self.db_path}.backup"
                shutil.copy2(self.db_path, backup_path)
                logger.info(f"Created backup of existing database at {backup_path}")
            
            s3_key = f"{self.s3_prefix}app.db"
            exists = await self.s3_handler.object_exists(s3_key)
            if not exists:
                logger.info("No database backup found in S3")
                return False
            
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            success = await self.s3_handler.download_file(s3_key, self.db_path)
            if success:
                logger.info("Successfully restored database from S3")
            else:
                logger.error("Failed to restore database from S3")
            return success
            
        except Exception as e:
            logger.error(f"Error restoring database from S3: {str(e)}")
            return False
    
    async def schedule_periodic_backup(self, interval_minutes: int = 30):
        while True:
            await asyncio.sleep(interval_minutes * 60)
            await self.backup_to_s3()
            
    @property 
    def s3_key(self):
        return f"{self.s3_prefix}app.db"

def get_global_db_persistence() -> DatabasePersistence:
    global _global_db_persistence
    if _global_db_persistence is None:
        _global_db_persistence = DatabasePersistence()
    return _global_db_persistence
