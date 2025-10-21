import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Iterable, Awaitable

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .config import BackendSettings, get_settings
from backend.services import users as user_service

logger = logging.getLogger(__name__)

_async_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def _max_overflow(min_size: int, max_size: int) -> int:
    if max_size <= min_size:
        return 0
    return max_size - min_size


async def init_engine() -> None:
    global _async_engine, _session_factory
    settings = get_settings()
    if not settings.postgres_user or not settings.postgres_password or not settings.postgres_db:
        logger.warning("Database credentials missing; backend engine not initialized")
        return
    try:
        logger.info("Initializing backend engine with host %r", settings.postgres_host)
        engine = create_async_engine(
            settings.database_url,
            pool_size=settings.min_pool_size,
            max_overflow=_max_overflow(settings.min_pool_size, settings.max_pool_size),
            pool_timeout=settings.pool_timeout,
            pool_recycle=settings.pool_recycle,
            pool_pre_ping=True,
            pool_reset_on_return="commit",
            echo=settings.echo,
            future=True,
        )
        session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )
        await _test_connection(engine)
        await _apply_schema(engine)
        _async_engine = engine
        _session_factory = session_factory
        await _ensure_admin_user(session_factory, settings)
        logger.info("Backend database engine initialized")
    except Exception as exc:
        logger.error("Failed to initialize backend database engine", exc_info=exc)
        raise


async def _test_connection(engine: AsyncEngine) -> None:
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.error("Backend database connection test failed", exc_info=exc)
        raise


async def _apply_schema(engine: AsyncEngine) -> None:
    from .models import Base

    # TODO: replace with migration workflow when ready
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def _ensure_admin_user(
    session_factory: async_sessionmaker[AsyncSession],
    settings: BackendSettings,
) -> None:
    if not settings.admin_email or not settings.admin_password:
        return
    async with session_factory() as session:
        try:
            created = await user_service.ensure_admin_user(
                session,
                settings.admin_email,
                settings.admin_password,
                settings.admin_full_name,
            )
            await session.commit()
            if created:
                logger.info("Default admin user provisioned")
        except Exception as exc:
            await session.rollback()
            logger.error("Failed to ensure default admin user", exc_info=exc)


def get_engine() -> AsyncEngine:
    if _async_engine is None:
        raise RuntimeError("Backend database engine not initialized")
    return _async_engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    if _session_factory is None:
        raise RuntimeError("Backend session factory not initialized")
    return _session_factory


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    factory = get_session_factory()
    session: AsyncSession | None = None
    try:
        session = factory()
        yield session
    except Exception:
        if session is not None:
            await session.rollback()
        raise
    finally:
        if session is not None:
            await session.close()


@asynccontextmanager
async def get_read_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_session() as session:
        yield session


@asynccontextmanager
async def get_write_session() -> AsyncGenerator[AsyncSession, None]:
    async with get_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def run_concurrent(tasks: Iterable[Awaitable[Any]]) -> list[Any]:
    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    results = list(gathered)
    for item in results:
        if isinstance(item, Exception):
            logger.error(
                "Concurrent backend task failed: %s",
                item,
                exc_info=(type(item), item, item.__traceback__),
            )
    return results


async def shutdown_engine() -> None:
    global _async_engine, _session_factory
    if _async_engine is not None:
        try:
            await _async_engine.dispose()
        except Exception as exc:
            logger.error("Error while disposing backend database engine", exc_info=exc)
        finally:
            _async_engine = None
            _session_factory = None
