from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db import get_read_session
from backend.models import User, UserSession
from backend.security import hash_token
from backend.services import users as user_service


@dataclass
class AuthenticatedUser:
    user: User
    session: UserSession


async def _get_read_db() -> AsyncIterator[AsyncSession]:
    async with get_read_session() as session:
        yield session


async def require_auth(
    authorization: str | None = Header(default=None, alias="Authorization"),
    session: AsyncSession = Depends(_get_read_db),
) -> AuthenticatedUser:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    raw_token = authorization.split(" ", 1)[1].strip()
    if not raw_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    token_hash_value = hash_token(raw_token)
    record = await user_service.get_user_with_token(session, token_hash_value)
    if record is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    if record.expires_at and record.expires_at <= datetime.now(timezone.utc):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")
    return AuthenticatedUser(user=record.user, session=record)


async def require_admin(
    auth: AuthenticatedUser = Depends(require_auth),
) -> AuthenticatedUser:
    role_names = {role.name for role in auth.user.roles}
    if "admin" not in role_names:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return auth