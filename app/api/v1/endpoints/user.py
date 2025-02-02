from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from ...db.base import get_db
from ...models.user import User
from ...schemas.user import UserCreate, User as UserSchema
from app.api.v1.core.security import get_password_hash
from ...schemas.user import UserCreate, User as UserSchema, UserResponse
from datetime import datetime, timezone
from ...schemas.user import TokenResponse, SignInRequest
from app.api.v1.core.security import get_password_hash, verify_password, create_access_token

router = APIRouter()

@router.post("/", response_model=UserSchema)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(
        email=user.email,
        username=user.username,
        hashed_password=get_password_hash(user.password)
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/", response_model=List[UserSchema])
def read_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@router.get("/{user_id}", response_model=UserSchema)
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.post("/signup", response_model=UserResponse)
async def signup(user_in: UserCreate, db: Session = Depends(get_db)):
    # Input validation
    if len(user_in.password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters"
        )

    # Check existing user
    existing_user = (
        db.query(User)
        .filter(
            (User.username == user_in.username) | (User.email == user_in.email)
        )
        .first()
    )
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Username or email already registered"
        )
    
    # Create new user
    db_user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=get_password_hash(user_in.password),
        is_active=True,
        created_at=datetime.now(timezone.utc)
    )
    
    try:
        # Save user to database
        db.add(db_user)
        db.commit()
        db.refresh(db_user)

        # Generate access token
        access_token = create_access_token(data={"sub": db_user.email})
        
        # Return response
        return UserResponse(
            userId=db_user.id, 
            username=db_user.username,
            email=db_user.email,
            is_active=db_user.is_active,
            created_at=db_user.created_at,
            access_token=access_token,
            token_type="bearer"
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/signin", response_model=TokenResponse)
async def signin(request: SignInRequest, db: Session = Depends(get_db)):
    # Find user
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        raise HTTPException(status_code=403, detail="User not found")
    
    # Verify password
    if not verify_password(request.password, user.hashed_password):  # Changed from password to hashed_password
        raise HTTPException(status_code=403, detail="Invalid password")
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
        
    # Create token
    token = create_access_token({
        "sub": str(user.id),
        "email": user.email,
        "username": user.username
    })
    
    return {"token": token}