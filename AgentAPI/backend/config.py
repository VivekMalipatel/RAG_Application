import os
from functools import lru_cache

from pydantic import BaseModel
from sqlalchemy.engine import URL, make_url


def _parse_database_url(raw_url: str) -> URL:
    url = raw_url
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgresql://") and "+" not in url.split("://", 1)[0]:
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    parsed = make_url(url)
    query = dict(parsed.query)
    if "sslmode" in query:
        query.pop("sslmode")
        parsed = parsed.set(query=query)
    return parsed


class BackendSettings(BaseModel):
    postgres_url: str | None = None
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str | None = None
    postgres_password: str | None = None
    postgres_db: str | None = None
    min_pool_size: int = 5
    max_pool_size: int = 20
    pool_timeout: int = 60
    pool_recycle: int = 300
    echo: bool = False
    session_ttl_seconds: int = 86400
    session_token_bytes: int = 32
    admin_email: str | None = None
    admin_password: str | None = None
    admin_full_name: str | None = None

    @property
    def database_url(self) -> str:
        if self.postgres_url:
            parsed = _parse_database_url(self.postgres_url)
            if parsed.host:
                self.postgres_host = parsed.host
            if parsed.port is not None:
                self.postgres_port = int(parsed.port)
            if parsed.username:
                self.postgres_user = parsed.username
            if parsed.password:
                self.postgres_password = parsed.password
            if parsed.database:
                self.postgres_db = parsed.database
            return parsed.render_as_string(hide_password=False)
        if not self.postgres_user or not self.postgres_password or not self.postgres_db:
            raise ValueError("Database credentials are not fully configured")
        url = URL.create(
            drivername="postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )
        return url.render_as_string(hide_password=False)


@lru_cache(maxsize=1)
def get_settings() -> BackendSettings:
    postgres_url = os.getenv("BACKEND_DB_URL")
    if postgres_url is not None:
        postgres_url = postgres_url.strip() or None

    postgres_host = os.getenv("BACKEND_DB_HOST")
    if postgres_host is None or postgres_host.strip() == "":
        postgres_host = "localhost"
    else:
        postgres_host = postgres_host.strip()

    postgres_port = os.getenv("BACKEND_DB_PORT")
    postgres_user = os.getenv("BACKEND_DB_USER")
    postgres_password = os.getenv("BACKEND_DB_PASSWORD")
    postgres_db = os.getenv("BACKEND_DB_NAME")

    min_pool_size = os.getenv("BACKEND_DB_MIN_POOL_SIZE")
    max_pool_size = os.getenv("BACKEND_DB_MAX_POOL_SIZE")
    pool_timeout = os.getenv("BACKEND_DB_POOL_TIMEOUT")
    pool_recycle = os.getenv("BACKEND_DB_POOL_RECYCLE")
    echo = os.getenv("BACKEND_DB_ECHO")

    session_ttl = os.getenv("BACKEND_SESSION_TTL_SECONDS")
    session_token_bytes = os.getenv("BACKEND_SESSION_TOKEN_BYTES")

    admin_email = os.getenv("BACKEND_ADMIN_EMAIL")
    admin_password = os.getenv("BACKEND_ADMIN_PASSWORD")
    admin_full_name = os.getenv("BACKEND_ADMIN_FULL_NAME")

    parsed_url: URL | None = None
    if postgres_url:
        parsed_url = _parse_database_url(postgres_url)
        if parsed_url.host:
            postgres_host = parsed_url.host
        if parsed_url.port is not None:
            postgres_port = str(parsed_url.port)
        if not postgres_user and parsed_url.username:
            postgres_user = parsed_url.username
        if not postgres_password and parsed_url.password:
            postgres_password = parsed_url.password
        if not postgres_db and parsed_url.database:
            postgres_db = parsed_url.database

    return BackendSettings(
        postgres_url=postgres_url,
        postgres_host=postgres_host,
        postgres_port=int((postgres_port or "5432").strip()),
        postgres_user=(postgres_user or "").strip() or None,
        postgres_password=(postgres_password or "").strip() or None,
        postgres_db=(postgres_db or "").strip() or None,
        min_pool_size=int((min_pool_size or "5").strip()),
        max_pool_size=int((max_pool_size or "20").strip()),
        pool_timeout=int((pool_timeout or "60").strip()),
        pool_recycle=int((pool_recycle or "300").strip()),
        echo=(echo or "false").strip().lower() in {"1", "true", "yes"},
        session_ttl_seconds=int((session_ttl or "86400").strip()),
        session_token_bytes=int((session_token_bytes or "32").strip()),
        admin_email=(admin_email or "").strip() or None,
        admin_password=(admin_password or "").strip() or None,
        admin_full_name=(admin_full_name or "").strip() or None,
    )
