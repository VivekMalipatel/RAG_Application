import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from passlib.context import CryptContext

from backend.config import get_settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return _pwd_context.verify(password, hashed)


def generate_session_token() -> tuple[str, datetime]:
    settings = get_settings()
    token = secrets.token_urlsafe(settings.session_token_bytes)
    expiry = datetime.now(timezone.utc) + timedelta(seconds=settings.session_ttl_seconds)
    return token, expiry


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()