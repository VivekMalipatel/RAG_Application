import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from ..dependencies import AuthenticatedUser, require_admin
from ..db import get_read_session, get_write_session
from ..schemas import AdminCreateUserRequest, AssignRolesRequest, UserRead
from ..services import users as user_service

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=list[UserRead])
async def list_users(_: AuthenticatedUser = Depends(require_admin)) -> list[UserRead]:
    async with get_read_session() as session:
        users = await user_service.list_users(session)
        return [user_service.to_user_read(user) for user in users]


@router.post("/users", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: AdminCreateUserRequest,
    _: AuthenticatedUser = Depends(require_admin),
) -> UserRead:
    async with get_write_session() as session:
        try:
            user = await user_service.create_user(session, payload, role_names=payload.roles)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return user_service.to_user_read(user)


@router.put("/users/{user_id}/roles", response_model=UserRead)
async def update_user_roles(
    user_id: uuid.UUID,
    payload: AssignRolesRequest,
    _: AuthenticatedUser = Depends(require_admin),
) -> UserRead:
    async with get_write_session() as session:
        user = await user_service.get_user_by_id(session, user_id)
        if user is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
        updated = await user_service.set_user_roles(session, user, payload.roles)
        return user_service.to_user_read(updated)
