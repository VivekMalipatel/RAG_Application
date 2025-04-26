from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from typing import Optional

from sqlalchemy.orm import Session

from db.session import get_db
from db.models import ApiKey
from config import settings

api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)
bearer_token_header = APIKeyHeader(name=settings.BEARER_TOKEN_HEADER, auto_error=False)

def get_api_key(
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Security(api_key_header),
    authorization: Optional[str] = Security(bearer_token_header),
) -> ApiKey:
    api_key_value = None
    
    if x_api_key:
        api_key_value = x_api_key
    
    elif authorization and settings.USE_BEARER_TOKEN:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            api_key_value = parts[1]
    
    if not api_key_value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if api_key_value in settings.API_KEYS:
        api_key = ApiKey(
            id=0,
            key=api_key_value,
            name="Default test key",
            is_active=True
        )
        return api_key
        
    api_key = db.query(ApiKey).filter(ApiKey.key == api_key_value, ApiKey.is_active == True).first()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key