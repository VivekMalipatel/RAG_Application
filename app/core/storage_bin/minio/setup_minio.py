import asyncio
import logging
import aiohttp
from app.config import settings

class MinIOSetup:
    """Handles MinIO initialization, webhook configuration, and bucket event notifications."""

    def __init__(self):
        self.minio_alias = "omnirag-minio"
        self.endpoint = f"http://{settings.MINIO_ENDPOINT}"
        self.access_key = settings.MINIO_ROOT_USER
        self.secret_key = settings.MINIO_ROOT_PASSWORD
        self.webhook_url = settings.MINIO_WEBHOOK_PATH
        self.webhook_secret = settings.MINIO_WEBHOOK_SECRET
        self.bucket_name = settings.MINIO_BUCKET_NAME

    async def wait_for_minio(self, max_retries=5, delay=3):
        """Waits until MinIO is fully ready to accept requests."""
        minio_health_url = f"{self.endpoint}/minio/health/live"
        async with aiohttp.ClientSession() as session:
            for attempt in range(max_retries):
                try:
                    async with session.get(minio_health_url) as response:
                        if response.status == 200:
                            logging.info("MinIO is ready")
                            return True
                except Exception as e:
                    logging.warning(f"MinIO not ready (attempt {attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(delay)
        
        logging.error("MinIO did not start in time.")
        return False

    async def run_command(self, command, capture_output=False):
        """Executes a shell command asynchronously and logs the result."""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE if capture_output else None,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logging.error(f"Command failed: {' '.join(command)} - {stderr.decode().strip()}")
                return None

            logging.info(f"Executed command: {' '.join(command)}")
            return stdout.decode().strip() if capture_output and stdout else None

        except Exception as e:
            logging.error(f"Error executing command: {' '.join(command)} - {e}")
            return None

    async def setup_minio_alias(self):
        """Sets up MinIO alias if not already set."""
        await self.run_command([
            "mc", "alias", "set", self.minio_alias, self.endpoint, self.access_key, self.secret_key
        ])
        logging.info("MinIO alias configured")

    async def configure_webhook(self):
        """Configures MinIO webhook settings."""
        await self.run_command([
            "mc", "admin", "config", "set", self.minio_alias, "notify_webhook:FastAPI",
            f"endpoint={self.webhook_url}",
            f"auth_token={self.webhook_secret}",
            "enable=on",
        ])
        await self.run_command(["mc", "admin", "service", "restart", self.minio_alias])
        logging.info("MinIO webhook configured and service restarted")

    async def event_rule_exists(self):
        """Checks if the bucket event rule already exists."""
        existing_rules = await self.run_command([
            "mc", "event", "list", f"{self.minio_alias}/{self.bucket_name}"
        ], capture_output=True)

        if existing_rules and "arn:minio:sqs::FastAPI:webhook" in existing_rules:
            logging.info("Event notification rule already exists. Skipping rule addition.")
            return True
        
        return False

    async def enable_bucket_notifications(self):
        """Enables event notifications for the configured MinIO bucket if not already set."""

        await asyncio.sleep(0.5)
        if await self.event_rule_exists():
            return  # Skip if the rule is already in place

        await self.run_command([
            "mc", "event", "add", f"{self.minio_alias}/{self.bucket_name}",
            "arn:minio:sqs::FastAPI:webhook",
            "--event", "put,delete"
        ])
        logging.info(f"Event notifications enabled for bucket: {self.bucket_name}")

    async def setup_minio(self):
        """Runs all MinIO setup steps sequentially in async mode."""
        if not await self.wait_for_minio():
            return
    
        await self.setup_minio_alias()
        await self.configure_webhook()
        await self.enable_bucket_notifications()


if __name__ == "__main__":
    minio_setup = MinIOSetup()
    asyncio.run(minio_setup.setup_minio())