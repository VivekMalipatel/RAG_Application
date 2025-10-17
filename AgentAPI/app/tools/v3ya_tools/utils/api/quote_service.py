from typing import Any, Dict, Optional
from ..api_client import BaseAPIService, HTTPMethod

class QuoteService(BaseAPIService):
    async def get_by_id(self, quote_id: str) -> Optional[Dict[str, Any]]:
        endpoint = f"quotes/{quote_id}"
        success, response = await self.client.make_request(
            endpoint=endpoint,
            method=HTTPMethod.GET.value,
        )
        if success:
            self.logger.info(f"Retrieved quote {quote_id}")
            return response if isinstance(response, dict) else None
        self.logger.error(f"Failed to get quote {quote_id}: {response}")
        return None

    async def list_by_user(self, user_id: str, page: int = 1, limit: int = 10) -> Optional[Dict[str, Any]]:
        endpoint = f"quotes?user_id={user_id}&page={page}&limit={limit}"
        success, response = await self.client.make_request(
            endpoint=endpoint,
            method=HTTPMethod.GET.value,
        )
        if success:
            self.logger.info(f"Retrieved quotes for user {user_id}")
            return response if isinstance(response, dict) else None
        self.logger.error(f"Failed to list quotes for user {user_id}: {response}")
        return None


__all__ = ["QuoteService"]
