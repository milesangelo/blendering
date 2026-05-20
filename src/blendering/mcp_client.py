"""Async wrapper around the MCP Python SDK (stdio transport)."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import MCPConfig
from .utils.logging import get_logger

log = get_logger("blendering.mcp")


@dataclass
class ToolSpec:
    """OpenAI-compatible tool spec for an MCP tool."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
                or {"type": "object", "properties": {}, "additionalProperties": True},
            },
        }


@dataclass
class ToolResult:
    text: str
    image_bytes: bytes | None = None
    is_error: bool = False


class BlenderMCP:
    """Thin async wrapper around an MCP ClientSession."""

    def __init__(self, session: ClientSession):
        self._session = session
        self._tools: list[ToolSpec] = []

    async def initialize(self) -> None:
        await self._session.initialize()
        listed = await self._session.list_tools()
        self._tools = [
            ToolSpec(
                name=t.name,
                description=t.description or "",
                parameters=(t.inputSchema or {}),
            )
            for t in listed.tools
        ]

    @property
    def tools(self) -> list[ToolSpec]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        log.debug("mcp call %s args=%s", name, arguments)
        result = await self._session.call_tool(name, arguments=arguments or {})
        text_parts: list[str] = []
        image_bytes: bytes | None = None
        for item in result.content:
            kind = getattr(item, "type", None)
            if kind == "text":
                text_parts.append(getattr(item, "text", ""))
            elif kind == "image":
                data = getattr(item, "data", "")
                if data:
                    image_bytes = base64.b64decode(data)
        return ToolResult(
            text="\n".join(text_parts),
            image_bytes=image_bytes,
            is_error=bool(getattr(result, "isError", False)),
        )

    async def get_screenshot(self, max_size: int = 1024) -> bytes | None:
        """Convenience helper: call get_viewport_screenshot and return PNG bytes."""
        for tool_name in ("get_viewport_screenshot", "get_screenshot"):
            if any(t.name == tool_name for t in self._tools):
                res = await self.call_tool(tool_name, {"max_size": max_size})
                return res.image_bytes
        return None


@asynccontextmanager
async def mcp_client(cfg: MCPConfig) -> AsyncIterator[BlenderMCP]:
    """Spawn the configured MCP server and yield a ready-to-use BlenderMCP."""
    params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env or None)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        client = BlenderMCP(session)
        await client.initialize()
        yield client
