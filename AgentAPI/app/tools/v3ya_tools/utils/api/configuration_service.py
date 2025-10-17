from typing import Any, Dict, Optional
from ..api_client import BaseAPIService, HTTPMethod

class ConfigurationService(BaseAPIService):
    async def create(self, config_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        endpoint = "configurations"
        success, response = await self.client.make_request(
            endpoint=endpoint,
            method=HTTPMethod.POST.value,
            payload=config_data,
        )
        if success:
            self.logger.info("Created configuration")
            return response if isinstance(response, dict) else None
        self.logger.error(f"Failed to create configuration: {response}")
        return None

    async def get_all(
        self,
        page: int = 1,
        limit: int = 10,
        part_id: Optional[str] = None,
        technology_id: Optional[str] = None,
        material_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        params = f"?page={page}&limit={limit}"
        if part_id:
            params += f"&part_id={part_id}"
        if technology_id:
            params += f"&technology_id={technology_id}"
        if material_id:
            params += f"&material_id={material_id}"
        endpoint = f"configurations{params}"
        success, response = await self.client.make_request(
            endpoint=endpoint,
            method=HTTPMethod.GET.value,
        )
        if success:
            self.logger.info("Retrieved configurations")
            return response if isinstance(response, dict) else None
        self.logger.error(f"Failed to get configurations: {response}")
        return None

    async def get_by_id(self, config_id: str) -> Optional[Dict[str, Any]]:
        endpoint = f"configurations/{config_id}"
        success, response = await self.client.make_request(
            endpoint=endpoint,
            method=HTTPMethod.GET.value,
        )
        if success:
            self.logger.info(f"Retrieved configuration {config_id}")
            return response if isinstance(response, dict) else None
        self.logger.error(f"Failed to get configuration {config_id}: {response}")
        return None

    async def update(self, config_id: str, config_data: Dict[str, Any]) -> bool:
        endpoint = f"configurations/{config_id}"
        success, response = await self.client.make_request(
            endpoint=endpoint,
            method=HTTPMethod.PUT.value,
            payload=config_data,
        )
        if success:
            self.logger.info(f"Updated configuration {config_id}")
            return True
        self.logger.error(f"Failed to update configuration {config_id}: {response}")
        return False

    async def delete(self, config_id: str) -> bool:
        endpoint = f"configurations/{config_id}"
        success, response = await self.client.make_request(
            endpoint=endpoint,
            method=HTTPMethod.DELETE.value,
        )
        if success:
            self.logger.info(f"Deleted configuration {config_id}")
            return True
        self.logger.error(f"Failed to delete configuration {config_id}: {response}")
        return False

    async def update_by_part_id(self, part_id: str, config_data: Dict[str, Any]) -> bool:
        endpoint = f"configurations/part/{part_id}"
        success, response = await self.client.make_request(
            endpoint=endpoint,
            method=HTTPMethod.PUT.value,
            payload=config_data,
        )
        if success:
            self.logger.info(f"Updated configuration for part {part_id}")
            return True
        self.logger.error(f"Failed to update configuration for part {part_id}: {response}")
        return False


__all__ = ["ConfigurationService"]
