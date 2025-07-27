# core/middleware/auth.py
import jwt
import hashlib
import hmac
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
import json
from functools import wraps

class MCPAuthMiddleware:
    def __init__(self, 
                 secret_key: str = None,
                 token_expiry_hours: int = 24,
                 enable_api_keys: bool = True,
                 enable_jwt: bool = True):
        
        self.secret_key = secret_key or os.getenv("MCP_SECRET_KEY", "your-secret-key-change-this")
        self.token_expiry_hours = token_expiry_hours
        self.enable_api_keys = enable_api_keys
        self.enable_jwt = enable_jwt
        
        # Load API keys from environment or file
        self.api_keys = self._load_api_keys()
        
        # Rate limiting storage (in production, use Redis)
        self.rate_limit_store = {}
        
    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load API keys from environment variables and hardcoded defaults"""
        api_keys = {}
        default_keys = {
            "dev-key-123": {
                "name": "Development Key",
                "permissions": ["read", "write", "admin"],
                "created": datetime.now().isoformat(),
                "source": "default"
            },
            "demo-read-456": {
                "name": "Demo Read Only",
                "permissions": ["read"],
                "created": datetime.now().isoformat(),
                "source": "default"
            },
            "demo-write-789": {
                "name": "Demo Write Access",
                "permissions": ["read", "write"],
                "created": datetime.now().isoformat(),
                "source": "default"
            }
        }
        
        api_keys.update(default_keys)
        
        admin_key = os.getenv("MCP_ADMIN_KEY")
        if admin_key:
            api_keys[admin_key] = {
                "name": "Admin User (from env)",
                "permissions": ["read", "write", "admin"],
                "created": datetime.now().isoformat(),
                "source": "environment"
            }
        
        readonly_key = os.getenv("MCP_READONLY_KEY") 
        if readonly_key:
            api_keys[readonly_key] = {
                "name": "Read Only User (from env)",
                "permissions": ["read"],
                "created": datetime.now().isoformat(),
                "source": "environment"
            }
        
        additional_keys = os.getenv("MCP_ADDITIONAL_KEYS")
        if additional_keys:
            try:
                extra_keys = json.loads(additional_keys)
                for key, info in extra_keys.items():
                    api_keys[key] = {
                        "name": info.get("name", "Additional Key"),
                        "permissions": info.get("permissions", ["read"]),
                        "created": info.get("created", datetime.now().isoformat()),
                        "source": "environment_additional"
                    }
            except json.JSONDecodeError:
                pass
        
        return api_keys
    
    def generate_jwt_token(self, user_id: str, permissions: list = None) -> str:
        """Generate a JWT token for a user"""
        if not self.enable_jwt:
            raise ValueError("JWT authentication is disabled")
        
        payload = {
            "user_id": user_id,
            "permissions": permissions or ["read"],
            "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            "iat": datetime.utcnow(),
            "iss": "mcp-server"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        if not self.enable_jwt:
            return None
        
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=["HS256"],
                options={"verify_exp": True}
            )
            return payload
        except jwt.ExpiredSignatureError:
            return {"error": "Token has expired"}
        except jwt.InvalidTokenError:
            return {"error": "Invalid token"}
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key"""
        if not self.enable_api_keys:
            return None
        
        if api_key in self.api_keys:
            return {
                "api_key": api_key,
                "user_info": self.api_keys[api_key],
                "permissions": self.api_keys[api_key].get("permissions", ["read"])
            }
        return {"error": "Invalid API key"}
    
    def check_rate_limit(self, identifier: str, max_requests: int = 100, window_minutes: int = 15) -> bool:
        """Simple rate limiting check"""
        now = time.time()
        window_start = now - (window_minutes * 60)
        
        if identifier not in self.rate_limit_store:
            self.rate_limit_store[identifier] = []
        
        # Clean old requests
        self.rate_limit_store[identifier] = [
            req_time for req_time in self.rate_limit_store[identifier] 
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.rate_limit_store[identifier]) >= max_requests:
            return False
        
        # Add current request
        self.rate_limit_store[identifier].append(now)
        return True
    
    def authenticate_request(self, headers: Dict[str, str], required_permission: str = "read") -> Dict[str, Any]:
        """Main authentication method"""
        auth_header = headers.get("Authorization", "")
        api_key_header = headers.get("X-API-Key", "")
        
        # Try JWT authentication first
        if auth_header.startswith("Bearer ") and self.enable_jwt:
            token = auth_header[7:]  # Remove "Bearer " prefix
            result = self.verify_jwt_token(token)
            
            if result and "error" not in result:
                user_permissions = result.get("permissions", [])
                if required_permission in user_permissions or "admin" in user_permissions:
                    
                    # Rate limiting
                    user_id = result.get("user_id", "unknown")
                    if not self.check_rate_limit(f"jwt_{user_id}"):
                        return {"error": "Rate limit exceeded", "status": 429}
                    
                    return {
                        "authenticated": True,
                        "auth_type": "jwt",
                        "user_id": result.get("user_id"),
                        "permissions": user_permissions
                    }
                else:
                    return {"error": f"Insufficient permissions. Required: {required_permission}", "status": 403}
            else:
                return {"error": result.get("error", "Authentication failed"), "status": 401}
        
        # Try API Key authentication
        elif api_key_header and self.enable_api_keys:
            result = self.verify_api_key(api_key_header)
            
            if result and "error" not in result:
                user_permissions = result.get("permissions", [])
                if required_permission in user_permissions or "admin" in user_permissions:
                    
                    # Rate limiting  
                    if not self.check_rate_limit(f"api_{api_key_header[:8]}"):
                        return {"error": "Rate limit exceeded", "status": 429}
                    
                    return {
                        "authenticated": True,
                        "auth_type": "api_key", 
                        "user_info": result.get("user_info"),
                        "permissions": user_permissions
                    }
                else:
                    return {"error": f"Insufficient permissions. Required: {required_permission}", "status": 403}
            else:
                return {"error": result.get("error", "Authentication failed"), "status": 401}
        
        # No valid authentication found
        return {"error": "Authentication required. Provide Bearer token or X-API-Key header", "status": 401}

    def create_auth_decorator(self, required_permission: str = "read"):
        """Create a decorator for tool authentication"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # In FastMCP, we need to get headers from the request context
                # This is a simplified approach - you may need to adapt based on FastMCP's actual implementation
                
                # For now, we'll add auth info to the function's metadata
                if not hasattr(wrapper, '_auth_required'):
                    wrapper._auth_required = True
                    wrapper._required_permission = required_permission
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator
