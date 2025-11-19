import json as js
from typing import Any, Literal

from cybotrade.http import Client, Response


class HttpClient:
    def __init__(
        self,
        url: str,
        headers: dict[str, str] = {"Content-Type": "application/json"},
    ):
        self.url = url
        self.client = Client()
        self.default_headers = headers

    async def get(
        self, json: dict[str, Any] | None = None, headers: dict[str, str] = {}
    ) -> Response:
        return await self.client.request(
            method="GET",
            url=self.url,
            body=None if json is None else js.dumps(json),
            headers={**self.default_headers, **headers},
        )

    async def post(
        self, json: dict[str, Any] | None = None, headers: dict[str, str] = {}
    ) -> Response:
        return await self.client.request(
            method="POST",
            url=self.url,
            body=None if json is None else js.dumps(json),
            headers={**self.default_headers, **headers},
        )

    async def put(
        self, json: dict[str, Any] | None = None, headers: dict[str, str] = {}
    ) -> Response:
        return await self.client.request(
            method="PUT",
            url=self.url,
            body=None if json is None else js.dumps(json),
            headers={**self.default_headers, **headers},
        )

    async def patch(
        self, json: dict[str, Any] | None = None, headers: dict[str, str] = {}
    ) -> Response:
        return await self.client.request(
            method="PATCH",
            url=self.url,
            body=None if json is None else js.dumps(json),
            headers={**self.default_headers, **headers},
        )

    async def delete(
        self, json: dict[str, Any] | None = None, headers: dict[str, str] = {}
    ) -> Response:
        return await self.client.request(
            method="DELETE",
            url=self.url,
            body=None if json is None else js.dumps(json),
            headers={**self.default_headers, **headers},
        )

    async def request(
        self,
        method: Literal[
            "CONNECT",
            "DELETE",
            "GET",
            "HEAD",
            "OPTIONS",
            "PATCH",
            "POST",
            "PUT",
            "TRACE",
        ],
        url: str,
        body: str | None,
        headers: dict[str, str] = {},
    ) -> Response:
        return await self.client.request(
            method=method,
            url=url,
            body=body,
            headers={**self.default_headers, **headers},
        )
