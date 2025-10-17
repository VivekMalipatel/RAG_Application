import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, Optional, Union

import aiohttp


class HTTPMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class MiddlewareAwareAPIClient:
    def __init__(self, base_url: str, auth_config: Optional[Dict[str, Any]] = None):
        self.base_url = base_url.rstrip("/")
        self.auth_config = auth_config or {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        self._auth_token: Optional[str] = None
        self._token_expires_at: Optional[float] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Python-Worker/1.0",
                "Accept": "application/json",
            }
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=20),
            )
        return self.session

    async def _get_client_credentials_token(self) -> Optional[str]:
        if (
            self._auth_token
            and self._token_expires_at
            and datetime.now(timezone.utc).timestamp() < self._token_expires_at - 60
        ):
            return self._auth_token
        try:
            token_url = self.auth_config["token_url"]
            client_id = self.auth_config["client_id"]
            client_secret = self.auth_config["client_secret"]
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    token_url,
                    json={"client_id": client_id, "client_secret": client_secret},
                ) as response:
                    data = await response.json()
                    self._auth_token = data.get("access_token")
                    expires_in = data.get("expires_in", 3600)
                    self._token_expires_at = (
                        datetime.now(timezone.utc).timestamp() + expires_in
                    )
                    return self._auth_token
        except Exception as error:
            self.logger.error(f"Error getting client credentials token: {error}")
            return None

    async def _handle_authentication(self, headers: Dict[str, str]) -> Dict[str, str]:
        auth_type = self.auth_config.get("type", "none")

        if auth_type == "client_credentials":
            token = await self._get_client_credentials_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"

        return headers

    async def _handle_middleware_headers(
        self,
        headers: Dict[str, str],
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        headers["X-Request-ID"] = f"python-worker-{int(time.time())}-{hash(endpoint) % 10000}"
        headers["X-Source"] = "python-gif-worker"
        headers["X-Timestamp"] = datetime.now(timezone.utc).isoformat()

        if payload is not None:
            content = json.dumps(payload).encode("utf-8")
            headers["Content-Length"] = str(len(content))

        if "client_id" in self.auth_config:
            headers["X-Client-ID"] = self.auth_config["client_id"]

        if self.auth_config.get("cors_headers"):
            headers["Origin"] = self.auth_config.get("origin", self.base_url)

        return headers

    async def make_request(
        self,
        endpoint: str,
        method: str = "POST",
        payload: Optional[Dict[str, Any]] = None,
        query_params: Optional[Dict[str, str]] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        retry_count: int = 3,
    ) -> tuple[bool, Union[Dict[str, Any], str]]:
        for attempt in range(retry_count + 1):
            try:
                session = await self._get_session()
                url = f"{self.base_url}/{endpoint.lstrip('/')}"

                headers = dict(session.headers)

                if custom_headers:
                    headers.update(custom_headers)

                headers = await self._handle_authentication(headers)
                headers = await self._handle_middleware_headers(headers, endpoint, payload)

                request_kwargs: Dict[str, Any] = {
                    "url": url,
                    "headers": headers,
                    "params": query_params,
                }

                if method.upper() in {"POST", "PUT", "PATCH"} and payload is not None:
                    request_kwargs["json"] = payload

                async with session.request(method.upper(), **request_kwargs) as response:
                    response_text = await response.text()

                    try:
                        response_data: Union[Dict[str, Any], str] = json.loads(response_text) if response_text else {}
                    except json.JSONDecodeError:
                        response_data = response_text

                    if response.status == 401 and attempt < retry_count and self.auth_config.get("type") == "client_credentials":
                        self._auth_token = None
                        self._token_expires_at = None
                        await asyncio.sleep(1)
                        continue

                    if response.status == 429 and attempt < retry_count:
                        retry_after = response.headers.get("Retry-After", "60")
                        wait_time = min(int(retry_after), 300)
                        await asyncio.sleep(wait_time)
                        continue

                    if 200 <= response.status < 300:
                        return True, response_data

                    if 500 <= response.status < 600 and attempt < retry_count:
                        await asyncio.sleep(2 ** attempt)
                        continue

                    return False, response_data

            except aiohttp.ClientError as error:
                if attempt < retry_count:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return False, str(error)
            except Exception as error:
                return False, str(error)

        return False, "Max retries exceeded"

    async def close(self) -> None:
        if self.session and not self.session.closed:
            await self.session.close()

    async def __aenter__(self) -> "MiddlewareAwareAPIClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()


class BaseAPIService:
    def __init__(
        self,
        base_url: str,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        auth_config: Optional[Dict[str, Any]] = None

        if client_id and client_secret:
            auth_config = {
                "type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
                "token_url": f"{base_url.rstrip('/')}/auth/token",
            }

        self.client = MiddlewareAwareAPIClient(base_url, auth_config)
        self.logger = logging.getLogger(__name__)
        self._cached_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._client_id = client_id
        self._client_secret = client_secret

    async def request(
        self,
        endpoint: str,
        method: Union[str, HTTPMethod] = HTTPMethod.GET,
        payload: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, Union[Dict[str, Any], str]]:
        headers = await self._get_auth_headers()
        normalized_method = method.value if isinstance(method, HTTPMethod) else str(method).upper()
        return await self.client.make_request(
            endpoint=endpoint,
            method=normalized_method,
            payload=payload,
            custom_headers=headers or None,
        )

    async def _get_auth_headers(self) -> Dict[str, str]:
        if self._client_id and self._client_secret:
            token = await self._get_valid_token()
            return {"Authorization": f"Bearer {token}"}
        return {}

    async def _get_valid_token(self) -> str:
        now = datetime.now()
        if not self._client_id or not self._client_secret:
            raise RuntimeError("Client credentials are not configured")

        if self._cached_token and self._token_expires_at and now < self._token_expires_at:
            return self._cached_token

        token_data = await self._request_token()
        if token_data and "access_token" in token_data:
            self._cached_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self._token_expires_at = now + timedelta(seconds=expires_in - 60)
            return self._cached_token

        raise RuntimeError("Failed to obtain access token")

    async def _request_token(self) -> Optional[Dict[str, Any]]:
        if not self._client_id or not self._client_secret:
            self.logger.error("Client credentials are required for token requests")
            return None

        payload = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }

        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                url = f"{self.client.base_url.rstrip('/')}/auth/token"
                async with session.post(url, json=payload) as response:
                    response_text = await response.text()
                    if response.status == 200:
                        try:
                            return json.loads(response_text)
                        except json.JSONDecodeError:
                            self.logger.error(f"Invalid JSON response: {response_text}")
                            return None
                    self.logger.error(f"Token request failed: {response.status} - {response_text}")
                    return None
            except Exception as error:
                self.logger.error(f"Error requesting token: {error}")
                return None

    async def __aenter__(self) -> "BaseAPIService":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def close(self) -> None:
        await self.client.close()


__all__ = [
    "HTTPMethod",
    "BaseAPIService",
]
