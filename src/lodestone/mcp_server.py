"""MCP server exposing the discovery tools (A1).

The server is bound to ONE partner at start-up, from ``LODESTONE_PARTNER_TOKEN`` (or, for local
demos, ``LODESTONE_PARTNER``). No tool takes a partner argument, so an agent connected to partner
A's server has no way to even ask for partner B's data.

Run:  uv run lodestone mcp            (stdio transport; point any MCP client at it)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from lodestone.store import AccessDenied, Database
from lodestone.tools import TOOLS, Toolbox


def scope_from_env():
    db = Database()
    token = os.environ.get("LODESTONE_PARTNER_TOKEN")
    if token:
        partner_id = db.platform().partner_for_token(token)
    else:
        partner_id = os.environ.get("LODESTONE_PARTNER")
        if not partner_id:
            raise AccessDenied("set LODESTONE_PARTNER_TOKEN (or LODESTONE_PARTNER for local demos)")
    return db.scope(partner_id, actor="agent")


def build_server(box: Toolbox) -> Server:
    tools = [
        types.Tool(name=name, description=desc, inputSchema=model.model_json_schema())
        for name, (model, _, desc) in TOOLS.items()
    ]

    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        # Tools are CPU-bound (GP fits); run off the event loop.
        out = await asyncio.to_thread(box.call, params.name, params.arguments or {})
        return types.CallToolResult(
            content=[types.TextContent(text=json.dumps(out, default=str))],
            isError="error" in out,
        )

    return Server(
        "lodestone",
        instructions=(
            f"Closed-loop thin-film discovery tools for partner workspace {box.scope.partner_id}. "
            "Lab data comes from NREL's HTEM DB; campaigns replay real measurements."
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def _serve() -> None:
    server = build_server(Toolbox(scope_from_env()))
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    try:
        asyncio.run(_serve())
    except AccessDenied as e:
        print(f"lodestone MCP: {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
