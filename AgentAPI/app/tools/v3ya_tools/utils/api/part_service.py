from typing import Any, Dict, Optional
from ..api_client import BaseAPIService, HTTPMethod

class PartService(BaseAPIService):
    async def get_by_id(self, part_id: str) -> Optional[Dict[str, Any]]:
        endpoint = f"parts/{part_id}"
        success, response = await self.client.make_request(
            endpoint=endpoint,
            method=HTTPMethod.GET.value,
        )
        if success:
            self.logger.info(f"Retrieved part {part_id}")
            return response if isinstance(response, dict) else None
        self.logger.error(f"Failed to get part {part_id}: {response}")
        return None


__all__ = ["PartService"]
