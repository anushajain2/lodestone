"""Command line: `uv run lodestone --help`."""

from __future__ import annotations

import json
import warnings

import typer

warnings.filterwarnings("ignore")

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Lodestone: closed-loop discovery on your lab's thin-film data.")
data_app = typer.Typer(no_args_is_help=True, help="Lab data: pull from HTEM DB, import into a workspace.")
campaign_app = typer.Typer(no_args_is_help=True, help="Run campaigns.")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluations.")
agent_app = typer.Typer(no_args_is_help=True, help="The discovery agent.")
for name, sub in (("data", data_app), ("campaign", campaign_app), ("eval", eval_app), ("agent", agent_app)):
    app.add_typer(sub, name=name)


# ------------------------------------------------------------------------------------------ data


@data_app.command("pull")
def data_pull(system: list[str] = typer.Option(None, help="System(s) to pull, e.g. N-Sn-Zn. Default: the demo systems.")):
    """Pull libraries from the HTEM DB API and refresh the local snapshot (needs network)."""
    from lodestone.connectors import htem

    htem.list_libraries(refresh=True)
    for s in system or list(htem.DEMO_SYSTEMS):
        htem.pull_system(s)


@data_app.command("systems")
def data_systems():
    """Systems available in the shipped HTEM snapshot."""
    from lodestone.connectors import htem

    for s in htem.available_systems():
        libs = htem.load_snapshot(s)
        typer.echo(f"{s:12s} {len(libs):3d} libraries, {sum(len(l.get('samples') or []) for l in libs):5d} samples  {htem.DEMO_SYSTEMS.get(s, '')}")


@data_app.command("import")
def data_import(
    partner: str = typer.Option("partner-a"),
    library: int = typer.Option(None, help="One HTEM library id"),
    system: str = typer.Option(None, help="Every library of a system"),
    csv: str = typer.Option(None, help="A lab CSV export (needs --system unless the mapping names it)"),
    mapping: str = typer.Option(None, help="Column mapping YAML/JSON for --csv (default: the Helix demo layout)"),
):
    """Import lab data into a partner workspace and print the import report."""
    from lodestone.connectors.mapping import load_mapping
    from lodestone.onboarding import import_htem_library, import_htem_system, import_partner_csv
    from lodestone.store import Database

    scope = Database().scope(partner, "cli")
    if csv:
        reps = import_partner_csv(scope, open(csv, encoding="utf-8-sig").read(), system, csv, load_mapping(mapping) if mapping else None)
    elif library:
        reps = [import_htem_library(scope, library)]
    else:
        reps = import_htem_system(scope, system)
    for r in reps:
        typer.echo(f"library {r.source_library_id}: received {r.received}, accepted {r.accepted}, rejected {r.rejected}  {r.reasons()}")


@data_app.command("check")
def data_check(
    csv: str = typer.Argument(..., help="A lab CSV export"),
    mapping: str = typer.Option(None, help="Column mapping YAML/JSON (default: the Helix demo layout)"),
    system: str = typer.Option(None, help="Element system, if the mapping doesn't name it"),
):
    """Dry-run an import: which columns are used, what would be accepted, rejected and why. Nothing is saved."""
    from lodestone.connectors.adapters import LabCSVAdapter, Reject
    from lodestone.connectors.mapping import helix_mapping, load_mapping

    import csv as csvlib

    m = load_mapping(mapping) if mapping else helix_mapping()
    text = open(csv, encoding="utf-8-sig").read()
    adapter = LabCSVAdapter(m)
    try:
        cols, comp, ignored = adapter.plan(next(csvlib.reader(text.splitlines()), []))
        records, reps = adapter.adapt_csv(text, system)
    except Reject as e:
        typer.echo(f"cannot import {csv}: {e}", err=True)
        raise typer.Exit(1)
    typer.echo(f"mapping: {m.name}")
    for f, c in cols.items():
        typer.echo(f"  {f:24s} <- {c.header}" + (f"  [{c.unit_text}]" if c.unit_text else "") + (f"  ERROR: {c.error}" if c.error else ""))
    for el, h in comp.items():
        typer.echo(f"  {'composition ' + el:24s} <- {h}")
    typer.echo(f"  unused columns: {ignored or 'none'}")
    for r in reps:
        typer.echo(f"library {r.source_library_id}: received {r.received}, accepted {r.accepted}, rejected {r.rejected}  {r.reasons()}")
    typer.echo(f"{len(records)} samples would be imported; measured: {dict(sorted({p: sum(r.measurements.get(p, 0) for r in reps) for p in {p for r in reps for p in r.measurements}}.items()))}")


@app.command("replay-report")
def replay_report(
    csv: str = typer.Argument(..., help="Your lab's CSV export"),
    goal: str = typer.Option(..., help='e.g. "maximize conductivity with band gap >= 3 eV"'),
    mapping: str = typer.Option(None, help="Column mapping YAML/JSON (default: the Helix demo layout); see data/templates/"),
    cost_per_experiment: float = typer.Option(1500.0, help="What one experiment (one row) costs you"),
    currency: str = typer.Option("$"),
    system: str = typer.Option(None, help="Element system, if the mapping doesn't name it"),
    seeds: int = typer.Option(30, help="Replays per method; fewer than 20 gives an interval too wide to quote"),
    budget: int = typer.Option(None, help="Experiments per replay (default: from the goal, else min(60, half the samples))"),
    failure_rate: float = typer.Option(0.05, help="Simulated random lab failures on top of real missing measurements"),
    out: str = typer.Option(None, help="Output HTML (default: <export>-replay-report.html next to the export)"),
):
    """Replay your past experiments: how many could Bayesian optimization have skipped? Writes one HTML page."""
    from pathlib import Path

    from lodestone.replay_report import ReplayError, write_replay_report

    target = out or str(Path(csv).with_name(Path(csv).stem + "-replay-report.html"))
    typer.echo(f"replaying {seeds} seeds x 3 methods (the Bayesian runs take a few seconds each)...")
    try:
        path, r = write_replay_report(target, csv_path=csv, goal=goal, mapping_path=mapping, system=system, cost_per_experiment=cost_per_experiment,
                                      currency=currency, seeds=seeds, budget=budget, failure_rate=failure_rate, verbose=True)
    except (ReplayError, ValueError) as e:
        typer.echo(f"no report: {e}", err=True)
        raise typer.Exit(1)
    imp = r["import"]
    typer.echo(f"imported {imp['accepted']} of {imp['received']} rows ({imp['rejected']} rejected, {imp['dropped']} values dropped)")
    typer.echo(f"goal: {r['goal_summary']}")
    typer.echo(f"[{r['verdict']}] {r['verdict_text']}")
    for n in r["verdict_notes"]:
        typer.echo(f"  note: {n}")
    typer.echo(f"wrote {path}")


# -------------------------------------------------------------------------------------- campaigns


@app.command()
def seed(quick: bool = typer.Option(False, help="Fewer rounds, for a fast start")):
    """Reset the database and create the two demo partners, their data and campaigns."""
    from lodestone.demo import seed as do_seed

    for pid, sts in do_seed(full=not quick).items():
        for st in sts:
            b = st["best"] or {}
            typer.echo(f"{pid}: {st['name']}: {st['budget_used']}/{st['budget']} experiments, best {b.get('text')} "
                       f"(#{b.get('rank_in_data')} of {b.get('of')} samples, HTEM sample {b.get('htem_sample_id')})")


@campaign_app.command("run")
def campaign_run(
    goal: str = typer.Argument(..., help='e.g. "most conductive zinc tin nitride, 40 experiments"'),
    partner: str = typer.Option("partner-a"),
    strategy: str = typer.Option("bayes"),
    seed_: int = typer.Option(0, "--seed"),
):
    """Create a campaign from plain English and run it to the end of its budget."""
    from lodestone.campaign import CampaignService
    from lodestone.goals import parse_goal
    from lodestone.store import Database

    scope = Database().scope(partner, "cli")
    spec, notes = parse_goal(goal, list(scope.systems()))
    spec = spec.model_copy(update={"strategy": strategy, "seed": seed_})
    typer.echo(f"Goal: {spec.summary()}  {notes}")
    svc = CampaignService(scope)
    c = svc.create(spec)
    st = svc.run(c["campaign_id"], on_round=lambda i: typer.echo(f"round {i['round']}: {len(i['arrived'])} results, {len(i['failed'])} failed"))
    typer.echo(json.dumps(st, indent=2, default=str))


@campaign_app.command("compare")
def campaign_compare(campaign_id: str, partner: str = typer.Option("partner-a"), seeds: int = typer.Option(5)):
    """Replay a campaign's goal with random and grid search on the same data."""
    from lodestone.campaign import CampaignService
    from lodestone.store import Database

    out = CampaignService(Database().scope(partner, "cli")).compare_with_baselines(campaign_id, seeds)
    out.pop("curves")
    typer.echo(json.dumps(out, indent=2))


@campaign_app.command("export")
def campaign_export(campaign_id: str, partner: str = typer.Option("partner-a"), out: str = typer.Option("")):
    """Export a campaign as FAIR-style JSON."""
    from lodestone.export import export_campaign
    from lodestone.store import Database

    text = json.dumps(export_campaign(Database().scope(partner, "cli"), campaign_id), indent=2, default=str)
    if out:
        open(out, "w").write(text)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)


@app.command()
def bench(seeds: int = typer.Option(20), robustness_seeds: int = typer.Option(10)):
    """Benchmark (E1): Bayesian optimization vs random and grid search on real HTEM data."""
    from lodestone.loop.benchmark import BenchConfig, run_benchmark

    out = run_benchmark(BenchConfig(seeds=seeds, robustness_seeds=robustness_seeds))
    for k, v in out["campaigns"].items():
        typer.echo(k + ": " + ", ".join(f"{m} {s['experiments_to_top_median']:.0f}" for m, s in v["methods"].items()))


# ------------------------------------------------------------------------------------------ agent


@agent_app.command("run")
def agent_run(
    task: str = typer.Argument("Run a campaign to find the most conductive zinc tin nitride with 40 experiments, then tell me the best sample and whether Bayesian optimization beat random search."),
    partner: str = typer.Option("partner-a"),
    policy: str = typer.Option("claude", help="claude or scripted"),
    model: str = typer.Option(""),
):
    """Run the agent against a partner's MCP server."""
    from lodestone.agent import has_anthropic_credentials, run_claude, run_scripted, run_sync
    from lodestone.store import Database

    def show(ev):
        typer.echo(f"  → {ev.name}({json.dumps(ev.input)[:120]}) {'ok' if ev.ok else 'ERROR'}")

    if policy == "claude":
        if not has_anthropic_credentials():
            typer.echo("No Anthropic credentials: set ANTHROPIC_API_KEY (or use --policy scripted).", err=True)
            raise typer.Exit(2)
        run = run_sync(run_claude(task, partner_id=partner, model=model or None, on_event=show))
    else:
        run = run_sync(run_scripted({"kind": "campaign", "create": {"goal": task}, "compare": True}, task, partner_id=partner, on_event=show))
    typer.echo("\n" + run.final_text)
    typer.echo(json.dumps(run.metrics(), indent=2))
    Database().scope(partner, "agent").save_agent_run((run.final_json or {}).get("campaign_id"), run.transcript(), run.metrics())


# ------------------------------------------------------------------------------------------ evals


@eval_app.command("agent")
def eval_agent(policy: str = typer.Option("scripted"), model: str = typer.Option(""), only: str = typer.Option(""), parse_with: str = typer.Option("rules")):
    """Agent test set (E2), scored automatically."""
    from lodestone.evals.agent_eval import run_agent_eval

    typer.echo(json.dumps(run_agent_eval(policy, model or None, only.split(",") if only else None, parse_with), indent=2))


@eval_app.command("onboarding")
def eval_onboarding():
    """Import reports for every snapshot library plus a set of deliberately broken records."""
    from lodestone.evals.onboarding_eval import run_onboarding_eval

    typer.echo(json.dumps(run_onboarding_eval(), indent=2)[:4000])


@eval_app.command("provenance")
def eval_provenance(n: int = typer.Option(20), live: bool = typer.Option(False, help="Also re-fetch each sample from the HTEM API")):
    """Spot-check random results against their source records (and the live API with --live)."""
    from lodestone.evals.onboarding_eval import run_provenance_check

    typer.echo(json.dumps(run_provenance_check(n, live), indent=2))


@eval_app.command("calibration")
def eval_calibration(reps: int = typer.Option(20), seeds: int = typer.Option(20), loop: bool = typer.Option(True, help="Also rerun the Bayesian campaigns with each uncertainty variant")):
    """Is the GP's uncertainty honest on held-out samples, and does fixing it help the loop?"""
    from lodestone.loop.calibration import save, save_bench

    save(reps=reps)
    if loop:
        save_bench(seeds=seeds)
    typer.echo("wrote reports/calibration/")


@eval_app.command("report")
def eval_report():
    """Write reports/EVALUATION.md from the latest results."""
    from lodestone.report import write_report

    typer.echo(f"wrote {write_report()}")


# ------------------------------------------------------------------------------------------ serve


@app.command()
def api(port: int = 8000):
    """Serve the REST API (FastAPI)."""
    import uvicorn

    uvicorn.run("lodestone.api:app", host="127.0.0.1", port=port)


@app.command()
def mcp():
    """Serve the MCP tools over stdio for one partner (LODESTONE_PARTNER_TOKEN)."""
    from lodestone.mcp_server import main

    main()


@app.command()
def dashboard(port: int = 8501):
    """Open the Streamlit dashboard."""
    import subprocess
    import sys

    from lodestone import config

    subprocess.run([sys.executable, "-m", "streamlit", "run", str(config.ROOT / "dashboard" / "app.py"), "--server.port", str(port), "--server.headless", "true"])


if __name__ == "__main__":
    app()
