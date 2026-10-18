import asyncio
import json
import sys

from lodestone.tools import Toolbox


def test_tool_guardrails(seeded):
    db, ids, _ = seeded
    box = Toolbox(db.scope("partner-a"))
    assert "error" in box.call("check_stability", {"material_ids": ["mp-99999999"]})
    assert "error" in box.call("check_stability", {"material_ids": ["hafnia"]})
    assert "error" in box.call("search_materials", {"contains_all": ["Qq"]})
    assert "error" in box.call("nonexistent_tool", {})
    assert "error" in box.call("set_target", {"excluded_elements": ["O"], "required_elements": ["O"]})
    out = box.call("search_materials", {"formula": "HfO2"})
    assert out["materials"][0]["material_id"] == "mp-352"
    assert "k_total" not in json.dumps(out) and "band_gap" not in out["materials"][0]  # hidden truth stays hidden
    assert sum(not c.ok for c in box.calls) == 5


def test_mcp_server_over_stdio(seeded):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    _, ids, path = seeded

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "lodestone.mcp_server"],
                                       env={"LODESTONE_PARTNER_TOKEN": "tok-a", "LODESTONE_DB": path})
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = [t.name for t in (await s.list_tools()).tools]
                own = await s.call_tool("get_program_status", {"program_id": ids["partner-a"]["program"]})
                other = await s.call_tool("get_program_status", {"program_id": ids["partner-b"]["program"]})
                return names, own, other

    names, own, other = asyncio.run(go())
    assert {"search_materials", "check_stability", "propose_batch", "submit_experiments", "get_results"} <= set(names)
    assert not own.is_error and other.is_error
