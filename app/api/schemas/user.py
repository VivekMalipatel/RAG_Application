from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    email: EmailStr
    username: str


class UserUpdate(UserBase):
    password: Optional[str] = None

class UserInDBBase(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]

class Config:
    from_attributes = True

class User(UserInDBBase):
    pass

class SignInRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    token: str


class Token(BaseModel):
    access_token: str
    token_type: str

class UserBase(BaseModel):
    username: str
    email: str

class UserCreate(UserBase):
    password: str

class UserData(UserBase):
    id: int
    is_active: bool
    created_at: datetime

class UserResponse(BaseModel):
    userId: int  # Required field as per error
    username: str
    email: str
    is_active: bool
    created_at: datetime
    access_token: str
    token_type: str