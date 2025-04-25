import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DB_URL: str = os.getenv("DB_URL", "sqlite:///./app.db")
    # Add more config as needed

settings = Settings()