"""Quick smoke test: connect to the configured MCP server and print its tools."""

from __future__ import annotations

import asyncio
import sys

from blendering.config import load_settings
from blendering.mcp_client import mcp_client


async def main() -> int:
    s = load_settings()
    print(f"Launching MCP: {s.mcp.command} {' '.join(s.mcp.args)}")
    try:
        async with mcp_client(s.mcp) as mcp:
            print(f"Connected. {len(mcp.tools)} tools:")
            for t in mcp.tools:
                print(f"  - {t.name}: {t.description.splitlines()[0] if t.description else ''}")
            print("\nAttempting get_viewport_screenshot…")
            try:
                img = await asyncio.wait_for(mcp.get_screenshot(max_size=512), timeout=15)
                if img:
                    print(f"  ok — {len(img)} bytes")
                else:
                    print("  (no screenshot tool found or returned no image)")
            except TimeoutError:
                print("  TIMEOUT — is Blender open and the add-on Connected?")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc!r}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
