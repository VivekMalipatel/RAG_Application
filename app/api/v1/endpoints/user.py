from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ...db.base import get_db
from ...models.user import User
from ...schemas.user import UserCreate, UserResponse, UserEmailUpdate, UserListResponse, DeleteUserResponse
from ...core.crud import CRUDBase
from ...core.security import get_password_hash, create_access_token
router = APIRouter()

def process_user_create(data: dict) -> dict:
    # Convert password to hashed_password
    if "password" in data:
        data["hashed_password"] = get_password_hash(data.pop("password"))
    return data

user_crud = (
    CRUDBase[User, UserCreate, UserResponse](User)
    .set_create_handler(process_user_create)
)

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check existing user
    existing_user = db.query(User).filter(
        (User.email == user.email) | (User.username == user.username)
    ).first()
    
    if existing_user:
        if existing_user.email == user.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user
    db_user = user_crud.create(db=db, obj_in=user)
    
    # Generate token
    access_token = create_access_token(data={"sub": db_user.email})
    
    # Return response with required fields
    return UserResponse(
        userId=db_user.id,
        username=db_user.username,
        email=db_user.email,
        is_active=db_user.is_active,
        created_at=db_user.created_at,
        access_token=access_token,
        token_type="bearer"
    )

@router.get("/", response_model=list[UserListResponse])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = user_crud.get_multi(db=db, skip=skip, limit=limit)
    return [
        UserListResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            created_at=user.created_at
        ) for user in users
    ]

@router.get("/{user_id}", response_model=UserResponse)
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = user_crud.get(db=db, id=user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/{user_identifier}", response_model=UserResponse)
def update_user(
    user_identifier: str,
    user_update: UserEmailUpdate,
    db: Session = Depends(get_db)
):
    try:
        user_id = int(user_identifier)
        current_user = user_crud.get(db=db, id=user_id)
    except ValueError:
        current_user = db.query(User).filter(User.username == user_identifier).first()
    
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if email is already taken
    existing_email = db.query(User).filter(
        User.email == user_update.email,
        User.id != current_user.id
    ).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    updated_user = user_crud.update(
        db=db,
        db_obj=current_user,
        obj_in={"email": user_update.email}
    )
    
    access_token = create_access_token(data={"sub": updated_user.email})
    
    return UserResponse(
        userId=updated_user.id,
        username=updated_user.username,
        email=updated_user.email,
        is_active=updated_user.is_active,
        created_at=updated_user.created_at,
        access_token=access_token,
        token_type="bearer"
    )

@router.delete("/{user_identifier}", response_model=DeleteUserResponse)
def delete_user(user_identifier: str, db: Session = Depends(get_db)):
    try:
        user_id = int(user_identifier)
        user = user_crud.get(db=db, id=user_id)
    except ValueError:
        user = db.query(User).filter(User.username == user_identifier).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    username = user.username    
    user_crud.delete(db=db, id=user.id)
    
    return DeleteUserResponse(
        message="User deleted successfully",
        username=username
    )