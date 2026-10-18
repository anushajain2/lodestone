"""The Claude loop, exercised end to end over the real MCP server with a fake Anthropic client."""

import asyncio
from types import SimpleNamespace as NS

import anthropic

from lodestone.agent import run_claude


class FakeMessages:
    def __init__(self, script):
        self.script, self.calls = script, []

    async def create(self, **kw):
        self.calls.append({**kw, "messages": list(kw["messages"])})  # snapshot: the loop keeps appending
        return self.script[len(self.calls) - 1]


def _usage():
    return NS(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0, cache_read_input_tokens=500)


def test_claude_loop_over_mcp(seeded, monkeypatch):
    _, _, path = seeded
    script = [
        NS(stop_reason="tool_use", usage=_usage(), content=[
            NS(type="text", text="Looking up hafnia."),
            NS(type="tool_use", id="t1", name="search_materials", input={"formula": "HfO2"}),
            NS(type="tool_use", id="t2", name="check_stability", input={"material_ids": ["mp-424242424"]}),
        ]),
        NS(stop_reason="end_turn", usage=_usage(), content=[
            NS(type="text", text='HfO2 is mp-352.\n```json\n{"answer_material_ids": ["mp-352"], "shortlist": [], "program_id": null}\n```'),
        ]),
    ]
    fake = FakeMessages(script)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda *a, **k: NS(beta=NS(messages=fake)))
    run = asyncio.run(run_claude("Find HfO2", token="tok-a", db_path=path, model="claude-opus-5-5"))
    assert run.error is None
    assert [e.name for e in run.tool_events] == ["search_materials", "check_stability"]
    assert [e.ok for e in run.tool_events] == [True, False]  # the invented id is rejected by the tool
    assert run.final_json["answer_material_ids"] == ["mp-352"] and run.invented_ids() == []
    req = fake.calls[0]
    assert req["model"] == "claude-opus-5-5" and req["fallbacks"] == "default"
    assert {t["name"] for t in req["tools"]} >= {"propose_batch", "submit_experiments", "get_results"}
    second = fake.calls[1]["messages"][-1]["content"]
    assert second[1]["is_error"] is True and second[1]["tool_use_id"] == "t2"
    assert run.metrics()["cost_usd"] > 0


def test_substitution_generates_unverified_candidates():
    from lodestone.substitution import propose_substitutions

    out = propose_substitutions("LaOF", excluded=["Pb"])
    assert out and all(c["formula"] != "LaOF" for c in out)
    assert any(c["swap"].startswith("La→") for c in out)
