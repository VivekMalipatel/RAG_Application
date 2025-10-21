import uuid
from datetime import datetime, timezone

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from backend.security import generate_session_token, hash_password, hash_token, verify_password
from backend.schemas import UserCreate, UserRead
from backend.models import Role, User, UserSession

ROLE_DESCRIPTIONS = {
    "admin": "Administrator access",
    "user": "Standard user access",
}


async def ensure_roles(session: AsyncSession, role_names: list[str]) -> list[Role]:
    normalized = [name.lower() for name in role_names]
    stmt = select(Role).where(Role.name.in_(normalized))
    existing = await session.execute(stmt)
    by_name: dict[str, Role] = {role.name: role for role in existing.scalars()}
    created: list[Role] = []
    for name in normalized:
        if name not in by_name:
            role = Role(name=name, description=ROLE_DESCRIPTIONS.get(name))
            session.add(role)
            created.append(role)
            by_name[name] = role
    if created:
        await session.flush()
    return [by_name[name] for name in dict.fromkeys(normalized)]


async def ensure_default_roles(session: AsyncSession) -> None:
    await ensure_roles(session, list(ROLE_DESCRIPTIONS.keys()))


async def create_user(session: AsyncSession, payload: UserCreate, role_names: list[str] | None = None) -> User:
    await ensure_default_roles(session)
    email = payload.email.lower()
    existing = await session.execute(select(User).where(User.email == email))
    if existing.scalar_one_or_none() is not None:
        raise ValueError("User with this email already exists")
    user = User(
        email=email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
    )
    session.add(user)
    roles = await ensure_roles(session, role_names or ["user"])
    user.roles = roles
    await session.flush()
    await session.refresh(user)
    return user


async def list_users(session: AsyncSession) -> list[User]:
    stmt: Select[tuple[User]] = select(User).options(joinedload(User.roles)).order_by(User.created_at)
    result = await session.execute(stmt)
    return result.scalars().unique().all()


async def set_user_roles(session: AsyncSession, user: User, role_names: list[str]) -> User:
    if not role_names:
        role_names = ["user"]
    roles = await ensure_roles(session, role_names)
    user.roles = roles
    await session.flush()
    await session.refresh(user)
    return user


async def authenticate_user(session: AsyncSession, email: str, password: str) -> tuple[User, str, datetime]:
    stmt = select(User).options(joinedload(User.roles)).where(User.email == email.lower())
    result = await session.execute(stmt)
    user = result.unique().scalar_one_or_none()
    if user is None or not verify_password(password, user.hashed_password) or not user.is_active:
        raise ValueError("Invalid credentials")
    token, expires_at = generate_session_token()
    session_record = UserSession(
        user_id=user.id,
        token_hash=hash_token(token),
        expires_at=expires_at,
        is_active=True,
    )
    session.add(session_record)
    await session.flush()
    return user, token, expires_at


async def revoke_session(session: AsyncSession, session_record: UserSession) -> None:
    if not session_record.is_active:
        return
    session_record.is_active = False
    session_record.revoked_at = datetime.now(timezone.utc)
    await session.flush()


async def get_session_by_id(session: AsyncSession, session_id: uuid.UUID) -> UserSession | None:
    stmt = (
        select(UserSession)
        .options(joinedload(UserSession.user).joinedload(User.roles))
        .where(UserSession.id == session_id)
    )
    result = await session.execute(stmt)
    return result.unique().scalar_one_or_none()


async def get_user_by_id(session: AsyncSession, user_id: uuid.UUID) -> User | None:
    stmt = select(User).options(joinedload(User.roles)).where(User.id == user_id)
    result = await session.execute(stmt)
    return result.unique().scalar_one_or_none()


async def get_user_with_token(session: AsyncSession, token_hash_value: str) -> UserSession | None:
    stmt = (
        select(UserSession)
        .options(joinedload(UserSession.user).joinedload(User.roles))
        .where(UserSession.token_hash == token_hash_value, UserSession.is_active.is_(True))
    )
    result = await session.execute(stmt)
    return result.unique().scalar_one_or_none()


async def ensure_admin_user(
    session: AsyncSession,
    email: str,
    password: str,
    full_name: str | None = None,
) -> bool:
    normalized = email.lower()
    stmt = select(User).options(joinedload(User.roles)).where(User.email == normalized)
    result = await session.execute(stmt)
    user = result.unique().scalar_one_or_none()
    if user is not None:
        if not any(role.name == "admin" for role in user.roles):
            required = await ensure_roles(session, ["admin"])
            current = list(user.roles)
            for role in required:
                if role not in current:
                    current.append(role)
            user.roles = current
            await session.flush()
        return False
    payload = UserCreate(email=email, password=password, full_name=full_name)
    await create_user(session, payload, role_names=["admin", "user"])
    return True


def to_user_read(user: User) -> UserRead:
    return UserRead(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        roles=[role.name for role in user.roles],
        is_active=user.is_active,
        created_at=user.created_at,
    )