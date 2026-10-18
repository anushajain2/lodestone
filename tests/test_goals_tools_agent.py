import asyncio
import json
import sys
from types import SimpleNamespace as NS

import anthropic
import pytest

from lodestone.goals import parse_goal
from lodestone.schemas import CampaignSpec


@pytest.mark.parametrize("text,check", [
    ("most conductive zinc tin nitride, 40 experiments", lambda s: s.system == "N-Sn-Zn" and s.objectives[0].property == "conductivity" and s.budget == 40),
    ("transparent conductor in cobalt nickel zinc oxide: maximize conductivity with band gap above 3 eV", lambda s: s.kind == "constrained" and s.constraints[0].value == 3.0),
    ("Zn-Sn-N with a band gap between 1.4 and 1.6 eV, batches of 4", lambda s: s.kind == "window" and s.window.low == 1.4 and s.batch_size == 4),
    ("Co-Ni-O-Zn: highest transmittance with band gap below 3.5 eV, 30 runs", lambda s: s.objectives[0].property == "visible_transmittance" and s.constraints[0].op == "<="),
    ("zinc tin nitride with the widest band gap", lambda s: s.objectives[0].property == "band_gap" and s.objectives[0].direction == "max"),
])
def test_goal_reader(text, check):
    spec, _ = parse_goal(text)
    assert check(spec), spec


def test_spec_validation():
    with pytest.raises(ValueError):
        CampaignSpec(name="x", system="N-Sn-Zn")  # no goal
    with pytest.raises(ValueError):
        CampaignSpec(name="x", system="N-Sn-Qq", objectives=[{"property": "conductivity"}])
    with pytest.raises(ValueError):
        CampaignSpec(name="x", system="N-Sn-Zn", window={"property": "band_gap", "low": 2, "high": 1})


def test_tool_guardrails(seeded):
    from lodestone.tools import Toolbox

    db, ids, _ = seeded
    box = Toolbox(db.scope("partner-a"))
    assert "error" in box.call("get_record", {"record_id": "rec-htem-999999999"})
    assert "error" in box.call("get_record", {"record_id": "hafnia"})
    assert "error" in box.call("get_status", {"campaign_id": "cmp-0000000000"})
    assert "error" in box.call("nope", {})
    assert "error" in box.call("create_campaign", {"goal": "most conductive Ga-Mg-O-Zn"})
    out = box.call("find_records", {"system": "N-Sn-Zn", "limit": 2})
    assert out["returned"] == 2 and out["records"][0]["htem_sample_id"]
    systems = box.call("list_systems", {})
    assert {s["system"] for s in systems["available_from_htem"]} >= {"N-Sn-Zn", "Co-Ni-O-Zn"}


def test_mcp_server_over_stdio(seeded):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    _, ids, path = seeded

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "lodestone.mcp_server"], env={"LODESTONE_PARTNER_TOKEN": "tok-a", "LODESTONE_DB": path})
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = [t.name for t in (await s.list_tools()).tools]
                own = await s.call_tool("get_status", {"campaign_id": ids["partner-a"]["campaign"]})
                other = await s.call_tool("get_status", {"campaign_id": ids["partner-b"]["campaign"]})
                return names, own, other

    names, own, other = asyncio.run(go())
    assert {"list_systems", "import_library", "create_campaign", "run_round", "get_status", "explain_choice", "compare_with_baselines"} <= set(names)
    assert not own.is_error and other.is_error


class FakeMessages:
    def __init__(self, script):
        self.script, self.calls = script, []

    async def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})
        return self.script[len(self.calls) - 1]


def _u():
    return NS(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0, cache_read_input_tokens=500)


def test_claude_loop_over_mcp_and_invented_numbers(seeded, monkeypatch):
    from lodestone.agent import run_claude

    _, ids, path = seeded
    script = [
        NS(stop_reason="tool_use", usage=_u(), content=[
            NS(type="tool_use", id="t1", name="find_records", input={"system": "N-Sn-Zn", "limit": 1}),
            NS(type="tool_use", id="t2", name="get_record", input={"record_id": "rec-htem-424242424"}),
        ]),
        NS(stop_reason="end_turn", usage=_u(), content=[NS(type="text", text="PLACEHOLDER")]),
    ]
    fake = FakeMessages(script)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: NS(beta=NS(messages=fake)))

    # make the final answer quote the real top value and one made-up number
    orig_create = fake.create

    async def create(**kw):
        out = await orig_create(**kw)
        if out.stop_reason == "end_turn":
            tool_text = kw["messages"][-1]["content"][0]["content"]
            top = json.loads(tool_text)["records"][0]
            out.content[0].text = (f"Best is {top['record_id']} at {top['measurements']['conductivity']} S/cm, about 12345.6 in other units.\n"
                                   f"```json\n{json.dumps({'answer_ids': [top['record_id']], 'answer_value': top['measurements']['conductivity'], 'answer_text': 'x', 'campaign_id': None})}\n```")
        return out

    fake.create = create
    run = asyncio.run(run_claude("Find the best", token="tok-a", db_path=path, model="claude-opus-5-5"))
    assert run.error is None
    assert [e.ok for e in run.tool_events] == [True, False]
    assert run.invented_ids() == []
    assert run.invented_numbers() == ["12345.6"]
    req = fake.calls[0]
    assert req["model"] == "claude-opus-5-5" and req["fallbacks"] == "default"
    assert fake.calls[1]["messages"][-1]["content"][1]["is_error"] is True
