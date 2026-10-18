"""Seed the two demo partners: their imported lab data and campaigns (W1, D4)."""

from __future__ import annotations

from lodestone import config
from lodestone.campaign import CampaignService
from lodestone.connectors import htem
from lodestone.connectors.adapters import HTEMAdapter, to_partner_csv
from lodestone.goals import parse_goal
from lodestone.onboarding import import_htem_library, import_partner_csv
from lodestone.store import Database

PARTNERS = {
    "partner-a": ("Northwind Thin Films (demo)", "demo-token-a"),
    "partner-b": ("Helix Materials Lab (demo)", "demo-token-b"),
}
PARTNER_CSV = config.DATA_DIR / "partner_lab" / "helix_second_lab_export.csv"


def reset_db() -> Database:
    path = config.db_path()
    for suffix in ("", "-wal", "-shm"):
        p = path.with_name(path.name + suffix)
        if p.exists():
            p.unlink()
    return Database(path)


def second_lab_csv(libraries: list[dict]) -> str:
    """Helix's second lab: the same kind of films, exported in its own format, with a few bad rows."""
    recs = []
    for lib in libraries:
        recs += HTEMAdapter().adapt_library(lib, set())[0]
    lines = to_partner_csv(recs).splitlines()
    header, rows = lines[0], lines[1:]
    bad = []
    if len(rows) > 6:
        r = rows[3].split(",")
        r[6] = "-" + r[6]  # negative resistivity
        bad.append(",".join(r))
        r = rows[5].split(",")
        r[3] = r[3].replace(";", "/")  # unreadable composition
        bad.append(",".join(r))
        bad.append(rows[0])  # duplicate row
        r = rows[7].split(",")
        r[7] = f"{float(r[7]) / 1000:.3f}" if r[7] else "0.3"  # edge written in µm instead of nm → impossible gap
        bad.append(",".join(r))
    return "\n".join([header] + rows + bad) + "\n"


def seed(full: bool = True, verbose: bool = True) -> dict:
    db = reset_db()
    plat = db.platform()
    for pid, (name, token) in PARTNERS.items():
        plat.create_partner(name, pid, token=token)
    say = print if verbose else (lambda *a, **k: None)
    out = {}

    # Northwind: zinc tin nitride and IZTO from the HTEM connector.
    a = db.scope("partner-a", "demo-seed")
    for system in ("N-Sn-Zn", "In-O-Sn-Zn"):
        reps = [import_htem_library(a, lib["id"]) for lib in htem.load_snapshot(system)]
        say(f"[Northwind] imported {system}: {sum(r.accepted for r in reps)} accepted, {sum(r.rejected for r in reps)} rejected from {len(reps)} libraries")
    sa = CampaignService(a)
    spec, _ = parse_goal("most conductive zinc tin nitride, 60 experiments")
    spec = spec.model_copy(update={"name": "Most conductive zinc tin nitride", "delay_rounds": 1, "seed": 1})
    c1 = sa.create(spec)
    sa.run(c1["campaign_id"], max_rounds=200 if full else 3, on_round=_printer("Northwind σ", verbose))
    spec, _ = parse_goal("zinc tin nitride with a band gap between 1.4 and 1.6 eV, 60 experiments")
    spec = spec.model_copy(update={"name": "ZnSnN₂ band gap in the 1.4–1.6 eV solar window", "seed": 2})
    c2 = sa.create(spec)
    for _ in range(6 if full else 2):
        _printer("Northwind gap", verbose)(sa.run_round(c2["campaign_id"]))
    out["partner-a"] = [sa.status(c1["campaign_id"]), sa.status(c2["campaign_id"])]

    # Helix: Co-Ni-Zn-O from two labs: HTEM connector + a second lab's CSV export.
    b = db.scope("partner-b", "demo-seed")
    libs = [lib for lib in htem.load_snapshot("Co-Ni-O-Zn") if lib.get("samples")]
    via_csv, via_api = libs[:4], libs[4:]
    reps = [import_htem_library(b, lib["id"]) for lib in via_api]
    say(f"[Helix] imported Co-Ni-O-Zn via HTEM: {sum(r.accepted for r in reps)} accepted from {len(reps)} libraries")
    PARTNER_CSV.parent.mkdir(parents=True, exist_ok=True)
    PARTNER_CSV.write_text(second_lab_csv(via_csv))
    reps = import_partner_csv(b, PARTNER_CSV.read_text(), "Co-Ni-O-Zn", PARTNER_CSV.name)
    say(f"[Helix] imported second-lab CSV: {sum(r.accepted for r in reps)} accepted, {sum(r.rejected for r in reps)} rejected")
    sb = CampaignService(b)
    spec, _ = parse_goal("transparent conductor in cobalt nickel zinc oxide: maximize conductivity with band gap above 3 eV, 50 experiments")
    spec = spec.model_copy(update={"name": "Transparent p-type conductor (gap ≥ 3 eV)", "failure_rate": 0.1, "seed": 3})
    c3 = sb.create(spec)
    sb.run(c3["campaign_id"], max_rounds=200 if full else 3, on_round=_printer("Helix", verbose))
    out["partner-b"] = [sb.status(c3["campaign_id"])]
    db.close()
    return out


def _printer(tag: str, verbose: bool):
    def f(info):
        if verbose:
            print(f"[{tag}] round {info['round']:2d}: {len(info['arrived'])} results, {len(info['failed'])} failed, {info['still_queued']} still in the lab", flush=True)

    return f
