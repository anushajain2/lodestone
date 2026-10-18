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
    csv: str = typer.Option(None, help="A partner lab CSV export (needs --system)"),
):
    """Import lab data into a partner workspace and print the import report."""
    from lodestone.onboarding import import_htem_library, import_htem_system, import_partner_csv
    from lodestone.store import Database

    scope = Database().scope(partner, "cli")
    if csv:
        reps = import_partner_csv(scope, open(csv).read(), system, csv)
    elif library:
        reps = [import_htem_library(scope, library)]
    else:
        reps = import_htem_system(scope, system)
    for r in reps:
        typer.echo(f"library {r.source_library_id}: received {r.received}, accepted {r.accepted}, rejected {r.rejected}  {r.reasons()}")


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
def bench(seeds: int = typer.Option(20), robustness_seeds: int = typer.Option(10),
          level: str = typer.Option("sample", help="sample: one experiment = one spot (the main benchmark). library: one experiment = one deposition run"),
          variant: str = typer.Option("", help="Library level only: a sensitivity variant (budget-5, budget-15, seed-runs-1) instead of the main 10-run design")):
    """Benchmark (E1): Bayesian optimization vs random and grid search on real HTEM data."""
    from lodestone.loop.benchmark import LIBRARY_VARIANTS, BenchConfig, run_benchmark, run_library_benchmark, run_library_variant

    if level == "library":
        if variant and variant not in LIBRARY_VARIANTS:
            raise typer.BadParameter(f"variant must be one of {', '.join(LIBRARY_VARIANTS)}")
        out = run_library_variant(variant, seeds=seeds) if variant else run_library_benchmark(seeds=seeds)
        for k, v in out["campaigns"].items():
            typer.echo(k + ": runs to first top-5% sample " + ", ".join(f"{m} {s['experiments_to_first_top5_median']:.0f}" for m, s in v["methods"].items()))
        for k, v in out["skipped"].items():
            typer.echo(f"{k}: skipped ({v['reason']})")
        return
    if level != "sample":
        raise typer.BadParameter("level must be 'sample' or 'library'")
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
