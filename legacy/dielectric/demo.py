"""Seed the two demo partners and their programs (W1)."""

from __future__ import annotations

from lodestone import config
from lodestone.program import ProgramService
from lodestone.store import Database
from lodestone.targets import parse_target_text

PARTNERS = {
    "partner-a": ("Northwind Semiconductor Equipment (demo)", "demo-token-a"),
    "partner-b": ("Helix Specialty Chemicals (demo)", "demo-token-b"),
}


def reset_db() -> Database:
    path = config.db_path()
    for suffix in ("", "-wal", "-shm"):
        p = path.with_name(path.name + suffix)
        if p.exists():
            p.unlink()
    return Database(path)


def seed(full: bool = True, verbose: bool = True) -> dict:
    db = reset_db()
    plat = db.platform()
    for pid, (name, token) in PARTNERS.items():
        plat.create_partner(name, pid, token=token)

    out = {}
    # Partner A: the demo-script target, run to completion.
    a = ProgramService(db.scope("partner-a", "demo-seed"))
    spec, _ = parse_target_text("k above 15, gap above 4 eV, no lead, 60 experiments")
    t = a.create_target(spec)
    p = a.create_program(t["target_id"], name="Gate dielectric screen", strategy="bayes", seed=1)
    a.run(p["program_id"], max_rounds=100 if full else 3, on_round=_printer("A", verbose))
    out["partner-a"] = a.status(p["program_id"])

    # Partner B: a different chemistry and target, part-way through.
    b = ProgramService(db.scope("partner-b", "demo-seed"))
    spec_b, _ = parse_target_text("fluorides with kappa > 8 and a band gap over 6 eV, ternary at most, 40 experiments")
    tb = b.create_target(spec_b)
    pb = b.create_program(tb["target_id"], name="Wide-gap fluoride capacitor layer", strategy="bayes_cost", seed=2)
    for _ in range(4 if full else 2):
        info = b.run_round(pb["program_id"])
        _printer("B", verbose)(info)
    out["partner-b"] = b.status(pb["program_id"])
    db.close()
    return out


def _printer(tag: str, verbose: bool):
    def f(info):
        if verbose:
            print(
                f"[{tag}] round {info['round']:2d} day {info['day']:5.1f}: {len(info['arrived'])} results, "
                f"{len(info['failed'])} failed, {len(info['rejected'])} rejected, {len(info['late'])} late",
                flush=True,
            )

    return f
