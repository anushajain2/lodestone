"""The Claude agent (A2) and the plain-English goal reader (T2).

The agent connects to the Lodestone MCP server over stdio (the same server any other MCP client
would use), lists its tools, and runs a manual tool-use loop with Claude until it answers. Every
run records what the evaluation needs: tool calls and how many were valid, tokens, cost, wall time,
ids the agent invented, and **numbers in its answer that no tool returned**.

A deterministic scripted policy drives the same MCP tools without an LLM. It is the reference policy
for the evaluation harness and keeps the demo runnable with no API key; it is labelled as such
everywhere it appears.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from lodestone import config

# $ per million tokens (input, output, cache read).
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20), "claude-sonnet-5-5": (2.0, 10.0, 0.20), "claude-haiku-5-5": (0.10, 0.50, 0.01)}

SYSTEM_PROMPT = """You are the discovery agent for Lodestone, a platform that helps materials labs find better thin films with fewer experiments. You work for one lab (the partner) and act only through the tools you are given; they are scoped to that partner's private workspace.

The data: the partner's "lab" is NREL's High-Throughput Experimental Materials Database (HTEM DB): real sputtered and laser-deposited thin films, each with a composition, synthesis conditions (deposition temperature, pressure) and measurements (four-point-probe conductivity in S/cm, direct band gap in eV from optical spectra, visible transmittance). Campaigns run as a lab replay: the optimizer picks samples and the replay returns their real measured values, so it can only choose among samples that were actually made.

How to work:
- To answer a question about the partner's data, use find_records / get_record (import a library first if the system isn't in the workspace).
- To run a campaign, call create_campaign (plain-English goal is fine), then run_round until the campaign status is complete, then get_status. Use explain_choice and compare_with_baselines when asked why or whether the optimizer helped.
- Every number and id you state must come from a tool result in this conversation. Never estimate, round loosely, or recall values from memory. If a tool returns an error, read it and fix the call.
- Failed experiments are normal (films too resistive to measure, simulated lab mishaps); keep going.
- Be concise and say which HTEM sample id each reported value comes from.

Finish with a short answer, then a fenced ```json block with exactly this shape:
{"answer_ids": ["rec-..." or HTEM sample ids], "answer_value": number or null, "answer_text": "...", "campaign_id": "cmp-..." or null}
answer_ids lists the record ids that answer the request (or [] if none); answer_value is the single number the request asks for, if any."""


@dataclass
class ToolEvent:
    name: str
    input: dict
    ok: bool
    output_preview: str
    seconds: float
    output: str = ""


@dataclass
class AgentRun:
    task: str
    policy: str
    model: str | None
    final_text: str = ""
    final_json: dict | None = None
    tool_events: list[ToolEvent] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    seconds: float = 0.0
    stop_reason: str | None = None
    error: str | None = None

    @property
    def n_tool_calls(self) -> int:
        return len(self.tool_events)

    @property
    def n_valid_tool_calls(self) -> int:
        return sum(e.ok for e in self.tool_events)

    @property
    def cost_usd(self) -> float:
        if not self.model or self.model not in PRICES:
            return 0.0
        pi, po, pc = PRICES[self.model]
        return (self.input_tokens * pi + self.output_tokens * po + self.cache_read_tokens * pc) / 1e6

    def tool_text(self) -> str:
        return " ".join(e.output for e in self.tool_events) + " " + self.task

    def invented_numbers(self) -> list[str]:
        """Numbers in the answer (text + JSON) that appear in no tool result or in the task itself."""
        answer = self.final_text.split("```json")[0]
        if self.final_json and self.final_json.get("answer_value") is not None:
            answer += f" {self.final_json['answer_value']}"
        seen = self.tool_text()
        nums_seen = {_norm(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", seen)}
        out = []
        for x in re.findall(r"(?<![\w.\-])-?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?", answer):
            n = _norm(x.replace(",", ""))
            if n in nums_seen or _close_to_any(n, nums_seen) or _trivial(n):
                continue
            out.append(x)
        return out

    def invented_ids(self) -> list[str]:
        text = self.final_text + json.dumps(self.final_json or {})
        ids = set(re.findall(r"\brec-[a-z]+-[A-Za-z0-9_\-]+\b", text)) | set(re.findall(r"\bcmp-[0-9a-f]{10}\b", text))
        seen = self.tool_text()
        return sorted(i for i in ids if i not in seen)

    def metrics(self) -> dict:
        return {
            "policy": self.policy, "model": self.model,
            "tool_calls": self.n_tool_calls, "valid_tool_calls": self.n_valid_tool_calls,
            "valid_tool_call_rate": (self.n_valid_tool_calls / self.n_tool_calls) if self.n_tool_calls else None,
            "invented_ids": self.invented_ids(), "invented_numbers": self.invented_numbers(),
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 4), "seconds": round(self.seconds, 1),
            "stop_reason": self.stop_reason, "error": self.error,
        }

    def transcript(self) -> list[dict]:
        return [{k: v for k, v in asdict(e).items() if k != "output"} for e in self.tool_events] + [{"final_text": self.final_text}]


def _norm(x: str) -> str:
    try:
        v = float(x)
    except ValueError:
        return x
    return f"{v:.4g}"


def _close_to_any(n: str, seen: set[str]) -> bool:
    """Allow the same number re-rounded (e.g. 2139.0 reported as 2139 or 2.14e3)."""
    try:
        v = float(n)
    except ValueError:
        return False
    for s in seen:
        try:
            w = float(s)
        except ValueError:
            continue
        if w != 0 and abs(v - w) / abs(w) < 0.01:
            return True
    return False


def _trivial(n: str) -> bool:
    try:
        v = float(n)
    except ValueError:
        return False
    return v in (0, 1, 2, 3, 4, 5, 10, 100) or v in (1.0,)


def extract_json(text: str) -> dict | None:
    for b in reversed(re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            continue
    return None


# ------------------------------------------------------------------------------- MCP connection


class MCPTools:
    """An MCP client session to the partner-bound Lodestone server."""

    def __init__(self, partner_id: str | None = None, token: str | None = None, db_path: str | None = None):
        env = {k: v for k, v in os.environ.items() if k.startswith(("PATH", "HOME", "LODESTONE", "PYTHON"))}
        if token:
            env["LODESTONE_PARTNER_TOKEN"] = token
        elif partner_id:
            env["LODESTONE_PARTNER"] = partner_id
        env["LODESTONE_DB"] = str(db_path or config.db_path())
        self.env = env
        self._stack = None

    async def __aenter__(self):
        from contextlib import AsyncExitStack

        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        self._stack = AsyncExitStack()
        params = StdioServerParameters(command=sys.executable, args=["-m", "lodestone.mcp_server"], env=self.env)
        read, write = await self._stack.enter_async_context(stdio_client(params, errlog=open(os.devnull, "w")))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        self.tools = (await self.session.list_tools()).tools
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    def anthropic_tools(self) -> list[dict]:
        return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema} for t in self.tools]

    async def call(self, name: str, args: dict) -> tuple[bool, str]:
        res = await self.session.call_tool(name, args, read_timeout_seconds=600)
        return (not res.is_error), "".join(getattr(c, "text", "") for c in res.content)


# ----------------------------------------------------------------------------------- Claude loop


async def run_claude(task: str, partner_id: str | None = None, token: str | None = None, model: str | None = None,
                     max_turns: int = 80, effort: str = "medium", db_path: str | None = None, on_event=None) -> AgentRun:
    import anthropic

    model = model or config.ANTHROPIC_MODEL
    run = AgentRun(task=task, policy="claude", model=model)
    client = anthropic.AsyncAnthropic()
    t0 = time.time()
    try:
        async with MCPTools(partner_id, token, db_path) as mcp:
            tools = mcp.anthropic_tools()
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}  # cache the tools + system prefix
            messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
            for _ in range(max_turns):
                resp = await client.beta.messages.create(
                    model=model, max_tokens=16000, system=SYSTEM_PROMPT, tools=tools, messages=messages,
                    output_config={"effort": effort}, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                )
                run.input_tokens += resp.usage.input_tokens + (resp.usage.cache_creation_input_tokens or 0)
                run.cache_read_tokens += resp.usage.cache_read_input_tokens or 0
                run.output_tokens += resp.usage.output_tokens
                run.stop_reason = resp.stop_reason
                if resp.stop_reason == "refusal":
                    run.error = f"refused: {getattr(resp.stop_details, 'category', None)}"
                    break
                messages.append({"role": "assistant", "content": resp.content})
                if resp.stop_reason == "pause_turn":
                    continue
                uses = [b for b in resp.content if b.type == "tool_use"]
                if not uses:
                    run.final_text = "".join(b.text for b in resp.content if b.type == "text")
                    break
                results = []
                for b in uses:
                    t1 = time.time()
                    try:
                        ok, text = await mcp.call(b.name, b.input if isinstance(b.input, dict) else {})
                    except Exception as e:  # transport failure: tell the model, keep going
                        ok, text = False, json.dumps({"error": f"tool transport failed: {e}"})
                    ev = ToolEvent(b.name, b.input, ok, text[:400], time.time() - t1, text)
                    run.tool_events.append(ev)
                    if on_event:
                        on_event(ev)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": text, "is_error": not ok})
                messages.append({"role": "user", "content": results})
            else:
                run.error = f"stopped after {max_turns} turns"
    except anthropic.AuthenticationError:
        run.error = "no valid Anthropic credentials (set ANTHROPIC_API_KEY)"
    except anthropic.APIStatusError as e:
        run.error = f"API error {e.status_code}: {e.message}"
    except anthropic.APIConnectionError:
        run.error = "could not reach the Anthropic API"
    run.seconds = time.time() - t0
    run.final_json = extract_json(run.final_text)
    return run


def has_anthropic_credentials() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_PROFILE"):
        return True
    return os.path.isdir(os.path.expanduser("~/.config/anthropic"))


# ------------------------------------------------------------------------------ scripted policy


async def run_scripted(intent: dict, task: str, partner_id: str | None = None, token: str | None = None, db_path: str | None = None, on_event=None) -> AgentRun:
    """Reference policy: no LLM; follows the task's structured intent with the same MCP tools.

    It shows the harness and tools work end to end. It is NOT evidence about Claude's behaviour.
    """
    run = AgentRun(task=task, policy="scripted-reference", model=None)
    t0 = time.time()
    async with MCPTools(partner_id, token, db_path) as mcp:

        async def call(name, args):
            t1 = time.time()
            ok, text = await mcp.call(name, args)
            ev = ToolEvent(name, args, ok, text[:400], time.time() - t1, text)
            run.tool_events.append(ev)
            if on_event:
                on_event(ev)
            return ok, json.loads(text)

        kind = intent.get("kind")
        ans_ids: list[str] = []
        value = None
        text = ""
        cid = None
        if kind == "lookup":
            ok, out = await call("find_records", intent["query"])
            if ok and out["records"]:
                top = out["records"][0]
                ans_ids = [top["record_id"]]
                field_ = intent.get("answer_field")
                value = top["deposition_temp_c"] if field_ == "deposition_temp_c" else top["measurements"].get(field_ or intent["query"].get("sort_by", "conductivity"))
                text = f"{top['record_id']} (HTEM sample {top['htem_sample_id']}): {value}"
            elif ok:
                text = "No records match."
        elif kind == "count":
            ok, out = await call("find_records", {**intent["query"], "limit": 1})
            value = out.get("matches") if ok else None
            text = f"{value} records match."
        elif kind == "provenance":
            ok, out = await call("get_record", {"record_id": intent["record_id"]})
            if ok:
                value = out["provenance"]["source_library_id"]
                ans_ids = [intent["record_id"]]
                text = f"Library {value}, sample {out['provenance']['source_sample_id']}"
        elif kind == "campaign":
            ok, out = await call("create_campaign", intent["create"])
            if ok:
                cid = out["campaign_id"]
                while True:
                    ok, r = await call("run_round", {"campaign_id": cid})
                    if not ok or r["campaign_status"] != "active":
                        break
                ok, st = await call("get_status", {"campaign_id": cid})
                b = st.get("best") or {}
                ans_ids = [b["record_id"]] if b.get("record_id") else []
                value = b.get(intent.get("answer_field", "conductivity"))
                text = f"Best: {b.get('label')} ({b.get('htem_sample_id')}) = {value}"
                if intent.get("compare"):
                    await call("compare_with_baselines", {"campaign_id": cid, "seeds": 2})
        elif kind == "guardrail":
            ok, out = await call(intent["tool"], intent["args"])
            text = out.get("error", "")
        run.final_json = {"answer_ids": ans_ids, "answer_value": value, "answer_text": text, "campaign_id": cid}
        run.final_text = f"Scripted reference policy ({kind}). {text}\n```json\n{json.dumps(run.final_json)}\n```"
        run.stop_reason = "end_turn"
    run.seconds = time.time() - t0
    return run


# ------------------------------------------------------------------------ T2: reading a goal


class GoalDraft(BaseModel):
    """What Claude extracts from a plain-English goal; validated into a CampaignSpec afterwards."""

    system: str = Field(description="Element system as dash-joined symbols, e.g. 'N-Sn-Zn', 'In-O-Sn-Zn', 'Co-Ni-O-Zn'")
    maximize: str | None = Field(None, description="conductivity | band_gap | visible_transmittance | sheet_resistance, or null")
    minimize: str | None = None
    band_gap_min_eV: float | None = None
    band_gap_max_eV: float | None = None
    window_low_eV: float | None = Field(None, description="Band-gap window lower edge, for 'hit a range' goals")
    window_high_eV: float | None = None
    budget: int | None = None
    batch_size: int | None = None
    assumptions: list[str] = Field(default_factory=list)


def parse_goal_with_claude(text: str, model: str | None = None):
    """Plain English → CampaignSpec using Claude structured outputs. Returns (spec, notes)."""
    import anthropic

    from lodestone.schemas import CampaignSpec

    resp = anthropic.Anthropic().messages.parse(
        model=model or config.ANTHROPIC_MODEL, max_tokens=4000, output_config={"effort": "low"},
        system=("Turn a thin-film lab's plain-English goal into structured fields. 'Transparent' means band gap ≥ 3 eV unless a number is given. "
                "Only fill a field when the text states or clearly implies it; record inferences in assumptions."),
        messages=[{"role": "user", "content": text}], output_format=GoalDraft,
    )
    d: GoalDraft = resp.parsed_output
    fields: dict = {"name": f"{d.system} campaign", "system": d.system, "plain_text": text, "objectives": [], "constraints": []}
    if d.window_low_eV is not None and d.window_high_eV is not None:
        fields["window"] = {"property": "band_gap", "low": d.window_low_eV, "high": d.window_high_eV}
    else:
        if d.maximize:
            fields["objectives"].append({"property": d.maximize, "direction": "max"})
        if d.minimize:
            fields["objectives"].append({"property": d.minimize, "direction": "min"})
        if d.band_gap_min_eV is not None:
            fields["constraints"].append({"property": "band_gap", "op": ">=", "value": d.band_gap_min_eV})
        if d.band_gap_max_eV is not None:
            fields["constraints"].append({"property": "band_gap", "op": "<=", "value": d.band_gap_max_eV})
    for k in ("budget", "batch_size"):
        if getattr(d, k):
            fields[k] = getattr(d, k)
    return CampaignSpec(**fields), ["read by Claude"] + d.assumptions


def run_sync(coro):
    return asyncio.run(coro)
