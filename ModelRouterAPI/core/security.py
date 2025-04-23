from fastapi import HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.orm import Session
from starlette.status import HTTP_401_UNAUTHORIZED, HTTP_403_FORBIDDEN

from config import settings
from db.session import get_db
from db.models import ApiKey

api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)

async def get_api_key(
    api_key_header: str = Security(api_key_header),
    db: Session = Depends(get_db),
) -> ApiKey:
    """
    Validate the API key from the request header and return the API key record.
    """
    # First check if key is in default settings (useful for development)
    if api_key_header in settings.API_KEYS:
        # For default keys, create a dummy API key object
        api_key = ApiKey(
            key=api_key_header,
            user_id="default",
            name="Default Key",
            is_active=True
        )
        return api_key
    
    # Otherwise check the database
    if api_key_header:
        api_key = db.query(ApiKey).filter(ApiKey.key == api_key_header).first()
        if api_key and api_key.is_active:
            return api_key
    
    # If no valid API key found, raise an exception
    raise HTTPException(
        status_code=HTTP_401_UNAUTHORIZED,
        detail="Invalid API key or key not provided",
        headers={"WWW-Authenticate": f"{settings.API_KEY_HEADER} access token"},
    )