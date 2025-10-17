from typing import Any, Dict, Optional

from ..api_client import BaseAPIService, HTTPMethod


class ProfileService(BaseAPIService):
    async def list_by_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        endpoint = f"profiles?user_id={user_id}"
        success, response = await self.client.make_request(
            endpoint=endpoint,
            method=HTTPMethod.GET.value,
        )
        if success:
            return response if isinstance(response, dict) else None
        return None


__all__ = ["ProfileService"]
