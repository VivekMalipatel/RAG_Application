import asyncio
import logging
import subprocess
import aiohttp
from app.config import settings

class MinIOSetup:
    """Handles MinIO initialization, webhook configuration, and bucket event notifications."""

    def __init__(self):
        self.minio_alias = "myminio"
        self.endpoint = f"http://{settings.MINIO_ENDPOINT}"
        self.access_key = settings.MINIO_ACCESS_KEY
        self.secret_key = settings.MINIO_SECRET_KEY
        self.webhook_url = settings.MINIO_WEBHOOK_PATH
        self.webhook_secret = settings.MINIO_WEBHOOK_SECRET
        self.bucket_name = settings.MINIO_BUCKET_NAME

    async def wait_for_minio(self):
        """Waits until MinIO is ready to accept requests."""
        minio_health_url = f"{self.endpoint}/minio/health/live"
        async with aiohttp.ClientSession() as session:
            for attempt in range(10):  # Retry up to 10 times
                try:
                    async with session.get(minio_health_url) as response:
                        if response.status == 200:
                            logging.info("MinIO is ready")
                            return True
                except Exception as e:
                    logging.warning(f"MinIO not ready yet (attempt {attempt+1}/10): {e}")
                    await asyncio.sleep(3)
        logging.error("MinIO did not start in time.")
        return False

    def run_command(self, command, capture_output=False):
        """Executes a shell command and logs the result."""
        try:
            result = subprocess.run(command, check=True, text=True, capture_output=capture_output)
            logging.info(f"Executed command: {' '.join(command)}")
            return result.stdout.strip() if capture_output else None
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed: {' '.join(command)} - {e}")
            return None

    def setup_minio_alias(self):
        """Sets up MinIO alias if not already set."""
        self.run_command([
            "mc", "alias", "set", self.minio_alias, self.endpoint, self.access_key, self.secret_key
        ])
        logging.info("MinIO alias configured")

    def configure_webhook(self):
        """Configures MinIO webhook settings."""
        self.run_command([
            "mc", "admin", "config", "set", self.minio_alias, "notify_webhook:FastAPI",
            f"endpoint={self.webhook_url}",
            f"auth_token={self.webhook_secret}",
            "enable=on",
        ])
        self.run_command(["mc", "admin", "service", "restart", self.minio_alias])
        logging.info("MinIO webhook configured and service restarted")

    def event_rule_exists(self):
        """Checks if the bucket event rule already exists."""
        existing_rules = self.run_command([
            "mc", "event", "list", f"{self.minio_alias}/{self.bucket_name}"
        ], capture_output=True)

        if existing_rules and "arn:minio:sqs::FastAPI:webhook" in existing_rules:
            logging.info("Event notification rule already exists. Skipping rule addition.")
            return True
        
        return False

    def enable_bucket_notifications(self):
        """Enables event notifications for the configured MinIO bucket if not already set."""
        if self.event_rule_exists():
            return  # Skip if the rule is already in place

        self.run_command([
            "mc", "event", "add", f"{self.minio_alias}/{self.bucket_name}",
            "arn:minio:sqs::FastAPI:webhook",
            "--event", "put,delete"
        ])
        logging.info(f"Event notifications enabled for bucket: {self.bucket_name}")

    async def setup_minio(self):
        """Runs all MinIO setup steps sequentially."""
        if not await self.wait_for_minio():
            return
    
        self.setup_minio_alias()
        self.configure_webhook()
        self.enable_bucket_notifications()


# Entry point to execute MinIO setup
if __name__ == "__main__":
    minio_setup = MinIOSetup()
    asyncio.run(minio_setup.setup_minio())