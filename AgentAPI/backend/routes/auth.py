from fastapi import APIRouter, Depends, HTTPException, Response, status

from backend.dependencies import AuthenticatedUser, require_auth
from backend.schemas import LoginRequest, LoginResponse, UserCreate, UserRead
from backend.services import users as user_service
from backend.db import get_write_session

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def register_user(payload: UserCreate) -> UserRead:
    async with get_write_session() as session:
        try:
            user = await user_service.create_user(session, payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return user_service.to_user_read(user)


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest) -> LoginResponse:
    async with get_write_session() as session:
        try:
            user, token, expires_at = await user_service.authenticate_user(session, payload.email, payload.password)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
        return LoginResponse(token=token, expires_at=expires_at, user=user_service.to_user_read(user))


@router.get("/me", response_model=UserRead)
async def read_me(auth: AuthenticatedUser = Depends(require_auth)) -> UserRead:
    return user_service.to_user_read(auth.user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(auth: AuthenticatedUser = Depends(require_auth)) -> Response:
    async with get_write_session() as session:
        record = await user_service.get_session_by_id(session, auth.session.id)
        if record is not None:
            await user_service.revoke_session(session, record)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
