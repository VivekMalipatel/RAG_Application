from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from typing import Optional

from sqlalchemy.orm import Session

from db.session import get_db
from db.models import ApiKey
from config import settings

# Define both API Key header and Bearer token security schemes
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)
bearer_token_header = APIKeyHeader(name=settings.BEARER_TOKEN_HEADER, auto_error=False)

def get_api_key(
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Security(api_key_header),
    authorization: Optional[str] = Security(bearer_token_header),
) -> ApiKey:
    """
    Validate API key from either X-Api-Key header or Authorization: Bearer token
    """
    api_key_value = None
    
    # Check for API key in X-Api-Key header
    if x_api_key:
        api_key_value = x_api_key
    
    # If not found and Bearer token format is enabled, check Authorization header
    elif authorization and settings.USE_BEARER_TOKEN:
        # Extract token from "Bearer {token}" format
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            api_key_value = parts[1]
    
    if not api_key_value:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Check hardcoded API keys first (for development/testing)
    if api_key_value in settings.API_KEYS:
        # Create a temporary API key object
        api_key = ApiKey(
            id=0,  # Dummy ID
            key=api_key_value,
            name="Default test key",
            is_active=True
        )
        return api_key
        
    # Look up the key in the database
    api_key = db.query(ApiKey).filter(ApiKey.key == api_key_value, ApiKey.is_active == True).first()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key