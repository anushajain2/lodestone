"""Connector for NREL's High-Throughput Experimental Materials Database (HTEM DB) (D1).

The "partner lab" in this prototype. HTEM DB holds real thin films made by sputtering and pulsed
laser deposition, each a composition-graded library of 44 measured spots with its synthesis
conditions. The API moved when NREL was renamed the National Laboratory of the Rockies (Dec 2025):
it now lives at https://htem-api.nlr.gov/api (the old htem-api.nrel.gov no longer resolves).

What this module does
* list every public library (``/api/sample_library``),
* fetch a library's detail and its samples (``/api/sample_library/{id}``, ``/api/sample/{id}``),
* keep only scalar fields: spectra are dropped, and so are people's names and e-mail addresses,
* cache raw records under ``data/raw/htem`` (gitignored) and write a compact per-system snapshot to
  ``data/htem/<system>.json.gz`` that ships with the repo, so a fresh clone works offline.

The records stay in HTEM's own shape here. Turning them into the platform's standard experiment
record is the adapter's job (:mod:`lodestone.connectors.adapters`), which is where validation and
rejection happen.

Data: Zakutayev et al., "An open experimental database for exploring inorganic materials",
Scientific Data 5, 180053 (2018). Cite the dataset: DOI 10.7799/1407128.
"""

from __future__ import annotations

import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from lodestone import config

API = "https://htem-api.nlr.gov/api"
RAW_DIR = config.DATA_DIR / "raw" / "htem"
SNAPSHOT_DIR = config.DATA_DIR / "htem"

LIBRARY_FIELDS = [
    "id", "num", "pdac", "quality", "sample_date", "elements", "xrf_elements", "xrf_compounds",
    "deposition_compounds", "deposition_power", "deposition_gases", "deposition_gas_flow_sccm",
    "deposition_sample_time_min", "deposition_base_pressure_mtorr", "deposition_growth_pressure_mtorr",
    "deposition_target_pulses", "deposition_rep_rate", "deposition_energy", "deposition_cycles",
    "deposition_ts_distance", "deposition_initial_temp_c", "deposition_substrate_material",
    "has_xrd", "has_xrf", "has_ele", "has_opt", "data_access", "sample_ids",
]
SAMPLE_FIELDS = [
    "id", "sample_library_id", "position", "thickness", "xyz_mm", "xrf_elements", "xrf_compounds",
    "xrf_concentration", "fpm_sheet_resistance", "fpm_resistivity", "fpm_conductivity",
    "fpm_standard_deviation", "absolute_temp_c", "opt_average_vis_trans", "opt_direct_bandgap", "peak_count",
]

# Systems used in the demo campaigns: plenty of samples with both electrical and optical data.
DEMO_SYSTEMS = {
    "In-O-Sn-Zn": "Indium zinc tin oxide: a transparent conducting oxide (composition-driven, room-temperature growth)",
    "N-Sn-Zn": "Zinc tin nitride (ZnSnN₂ family): an earth-abundant semiconductor (deposition temperature matters)",
    "Co-Ni-O-Zn": "Cobalt nickel zinc oxide: p-type transparent oxide candidates (grown at 400–700 °C)",
}


def system_key(elements: list[str] | None) -> str:
    return "-".join(sorted(elements or []))


def _client() -> httpx.Client:
    return httpx.Client(timeout=60.0, headers={"User-Agent": "lodestone/0.2 (research prototype)"})


def _get(client: httpx.Client, path: str, retries: int = 4):
    for attempt in range(retries):
        try:
            r = client.get(f"{API}/{path}")
            if r.status_code == 200:
                return r.json()
            if r.status_code in (400, 404):
                return None
        except httpx.HTTPError:
            pass
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"HTEM API request failed after {retries} attempts: {path}")


def list_libraries(refresh: bool = False) -> list[dict]:
    """Every public sample library (summary fields)."""
    path = RAW_DIR / "libraries.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    with _client() as c:
        libs = _get(c, "sample_library") or []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(libs))
    return libs


def libraries_for_system(system: str, require=("has_ele", "has_opt")) -> list[dict]:
    return [
        lib for lib in list_libraries()
        if system_key(lib.get("elements")) == system and all(lib.get(r) for r in require)
    ]


def fetch_library(library_id: int, workers: int = 8) -> dict:
    """A library with its samples (scalar fields only), cached on disk."""
    path = RAW_DIR / "libraries" / f"{library_id}.json.gz"
    if path.exists():
        return json.loads(gzip.decompress(path.read_bytes()))
    with _client() as c:
        detail = _get(c, f"sample_library/{library_id}")
        if detail is None:
            raise LookupError(f"HTEM library {library_id} not found")
        lib = {k: detail.get(k) for k in LIBRARY_FIELDS}

        def one(sid):
            s = _get(c, f"sample/{sid}")
            return None if s is None else {k: s.get(k) for k in SAMPLE_FIELDS}

        with ThreadPoolExecutor(workers) as ex:
            lib["samples"] = [s for s in ex.map(one, detail.get("sample_ids") or []) if s is not None]
    lib["source"] = {"api": f"{API}/sample_library/{library_id}", "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(lib).encode()))
    return lib


def pull_system(system: str, max_libraries: int | None = None, verbose: bool = True) -> list[dict]:
    """Fetch every library of a system that has electrical and optical data; write the snapshot."""
    libs = libraries_for_system(system)
    if max_libraries:
        libs = libs[:max_libraries]
    out = []
    for i, lib in enumerate(libs, 1):
        t0 = time.time()
        full = fetch_library(lib["id"])
        out.append(full)
        if verbose:
            print(f"[{system}] library {lib['id']} ({i}/{len(libs)}): {len(full['samples'])} samples, {time.time() - t0:.1f}s", flush=True)
    save_snapshot(system, out)
    return out


def snapshot_path(system: str) -> Path:
    return SNAPSHOT_DIR / f"{system}.json.gz"


def save_snapshot(system: str, libraries: list[dict]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path(system).write_bytes(gzip.compress(json.dumps(libraries, separators=(",", ":")).encode()))


def load_snapshot(system: str) -> list[dict]:
    path = snapshot_path(system)
    if not path.exists():
        raise FileNotFoundError(f"No snapshot for {system}. Run `uv run lodestone data pull --system {system}`.")
    return json.loads(gzip.decompress(path.read_bytes()))


def available_systems() -> list[str]:
    return sorted(p.name.removesuffix(".json.gz") for p in SNAPSHOT_DIR.glob("*.json.gz"))


def sample_url(sample_id: int) -> str:
    return f"{API}/sample/{sample_id}"


def library_url(library_id: int) -> str:
    return f"{API}/sample_library/{library_id}"
