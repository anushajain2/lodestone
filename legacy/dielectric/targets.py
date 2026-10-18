"""Target specs: filtering the pool by a partner's constraints (C2) and reading plain English (T2).

The plain-English reader here is deterministic and handles the common phrasings. The Claude agent
offers a second reader (``lodestone.agent.parse_target_with_claude``); either way the result is a
validated :class:`TargetSpec` that the partner confirms before anything runs.
"""

from __future__ import annotations

import re

import pandas as pd
from pymatgen.core.periodic_table import Element

from lodestone import config
from lodestone.schemas import TargetSpec

# --------------------------------------------------------------------------------------- filtering


def filter_pool(catalog: pd.DataFrame, spec: TargetSpec) -> pd.DataFrame:
    """Materials a partner allows. Uses only public columns (composition and stability)."""
    df = catalog
    stable = df["e_hull"].notna() & (df["e_hull"] <= spec.e_hull_max_eV_per_atom + 1e-12)
    excluded = set(spec.excluded_elements)
    required = set(spec.required_elements)
    ok_elements = df["elements"].apply(lambda els: not (excluded & set(els)) and required <= set(els))
    mask = stable & ok_elements
    if spec.max_elements:
        mask &= df["n_elements"] <= spec.max_elements
    return df[mask]


def explain_exclusions(catalog: pd.DataFrame, spec: TargetSpec) -> dict[str, int]:
    """How many materials each constraint removes (shown on the target page)."""
    unstable = ~(catalog["e_hull"].notna() & (catalog["e_hull"] <= spec.e_hull_max_eV_per_atom + 1e-12))
    excluded = catalog["elements"].apply(lambda els: bool(set(spec.excluded_elements) & set(els)))
    missing_req = catalog["elements"].apply(lambda els: not set(spec.required_elements) <= set(els))
    too_many = catalog["n_elements"] > spec.max_elements if spec.max_elements else catalog["n_elements"] < 0
    return {
        "catalog": int(len(catalog)),
        "unstable_or_unknown_stability": int(unstable.sum()),
        "contains_excluded_element": int(excluded.sum()),
        "missing_required_element": int(missing_req.sum()),
        "too_many_elements": int(too_many.sum()),
        "allowed": int(len(filter_pool(catalog, spec))),
    }


# ----------------------------------------------------------------------------- plain-English reader

_NAME_TO_SYMBOL = {el.long_name.lower(): el.symbol for el in Element}
_NAME_TO_SYMBOL.update({"sulphur": "S", "aluminum": "Al", "caesium": "Cs"})
_GROUPS = {
    "toxic": config.TOXIC_ELEMENTS,
    "toxic elements": config.TOXIC_ELEMENTS,
    "radioactive": config.RADIOACTIVE_ELEMENTS,
    "radioactive elements": config.RADIOACTIVE_ELEMENTS,
    "scarce": config.SCARCE_ELEMENTS,
    "scarce elements": config.SCARCE_ELEMENTS,
    "critical": config.SCARCE_ELEMENTS,
    "rare earths": config.RARE_EARTHS,
    "rare-earths": config.RARE_EARTHS,
    "rare earth": config.RARE_EARTHS,
    "rare-earth": config.RARE_EARTHS,
    "lanthanides": config.RARE_EARTHS,
    "halogens": ["F", "Cl", "Br", "I"],
    "halides": ["F", "Cl", "Br", "I"],
}
_ANION_WORDS = {
    "oxide": "O",
    "oxides": "O",
    "nitride": "N",
    "nitrides": "N",
    "fluoride": "F",
    "fluorides": "F",
    "chloride": "Cl",
    "chlorides": "Cl",
    "sulfide": "S",
    "sulfides": "S",
    "selenide": "Se",
    "selenides": "Se",
}
_NUM = r"(\d+(?:\.\d+)?)"
_GT = r"(?:above|over|greater than|more than|higher than|at least|exceeding|>=|≥|>|min(?:imum)?(?: of)?)"
_LT = r"(?:below|under|less than|lower than|at most|<=|≤|<|max(?:imum)?(?: of)?)"
_K_WORDS = r"(?:k|κ|kappa|dielectric constant|dielectric|permittivity|epsilon|ε)"
_GAP_WORDS = r"(?:band ?gap|gap|eg)"


def _element(token: str) -> list[str]:
    t = token.strip().strip(".,;").lower()
    if t in _GROUPS:
        return list(_GROUPS[t])
    if t in _NAME_TO_SYMBOL:
        return [_NAME_TO_SYMBOL[t]]
    sym = token.strip().strip(".,;")
    if Element.is_valid_symbol(sym):
        return [sym]
    if Element.is_valid_symbol(sym.capitalize()) and len(sym) <= 2 and sym.lower() not in {"no", "in", "as", "at", "be", "he", "i", "s", "o", "b"}:
        return [sym.capitalize()]
    return []


def parse_target_text(text: str) -> tuple[TargetSpec, list[str]]:
    """Read a plain-English target. Returns the spec and notes on what was understood or assumed."""
    s = text.strip()
    low = s.lower()
    notes: list[str] = []
    fields: dict = {"plain_text": s}

    def find(pattern: str):
        return re.search(pattern, low)

    # k thresholds
    if m := find(rf"{_K_WORDS}\s*(?:of\s*)?{_GT}\s*{_NUM}") or find(rf"{_K_WORDS}\s*(?:>|≥)\s*{_NUM}"):
        fields["k_min"] = float(m.group(1))
    if m := find(rf"{_K_WORDS}\s*(?:of\s*)?{_LT}\s*{_NUM}"):
        fields["k_max"] = float(m.group(1))
    if m := find(rf"{_K_WORDS}\s*(?:between|from)\s*{_NUM}\s*(?:and|to|-|–)\s*{_NUM}"):
        fields["k_min"], fields["k_max"] = float(m.group(1)), float(m.group(2))
    # gap thresholds
    if m := find(rf"{_GAP_WORDS}\s*(?:of\s*)?{_GT}\s*{_NUM}\s*(?:ev)?"):
        fields["gap_min_eV"] = float(m.group(1))
    if m := find(rf"{_GAP_WORDS}\s*(?:of\s*)?{_LT}\s*{_NUM}\s*(?:ev)?"):
        fields["gap_max_eV"] = float(m.group(1))
    if m := find(rf"{_GAP_WORDS}\s*(?:between|from)\s*{_NUM}\s*(?:and|to|-|–)\s*{_NUM}"):
        fields["gap_min_eV"], fields["gap_max_eV"] = float(m.group(1)), float(m.group(2))
    if "wide gap" in low or "wide band gap" in low or "wide-bandgap" in low or "wide bandgap" in low:
        if "gap_min_eV" not in fields:
            fields["gap_min_eV"] = 4.0
            notes.append("'wide band gap' read as gap > 4 eV")
    if ("high-k" in low or "high k" in low or "high κ" in low) and "k_min" not in fields:
        fields["k_min"] = 15.0
        notes.append("'high-k' read as k > 15")
    # stability
    if m := find(rf"(?:hull|e_hull|ehull|stability)[^\d]{{0,30}}{_NUM}\s*(?:mev)"):
        fields["e_hull_max_eV_per_atom"] = float(m.group(1)) / 1000.0
    elif m := find(rf"(?:hull|e_hull|ehull)[^\d]{{0,30}}{_NUM}"):
        fields["e_hull_max_eV_per_atom"] = float(m.group(1))
    elif "strictly stable" in low or "on the hull" in low or "ground state" in low:
        fields["e_hull_max_eV_per_atom"] = 0.0
    # budget / batch
    if m := find(rf"(?:budget|up to|within)\s*(?:of\s*)?(\d+)\s*(?:experiments|tests|samples)"):
        fields["budget"] = int(m.group(1))
    elif m := find(r"(\d+)\s*(?:experiments|tests|samples)"):
        fields["budget"] = int(m.group(1))
    if m := find(r"(?:batch(?:es)? of|batch size)\s*(\d+)") or find(r"(\d+)\s*(?:per round|a round|per batch)"):
        fields["batch_size"] = int(m.group(1))
    if m := find(r"(?:at most|no more than|max(?:imum)?|up to)\s*(\d)\s*elements") or find(r"(binary|ternary|quaternary)"):
        word = m.group(1)
        fields["max_elements"] = {"binary": 2, "ternary": 3, "quaternary": 4}.get(word, int(word) if word.isdigit() else None)

    # exclusions: "no lead", "without Pb or Cd", "lead-free", "exclude toxic elements"
    excluded = list(config.TOXIC_ELEMENTS)
    extra: list[str] = []
    for m in re.finditer(r"\b(?:no|without|exclude|excluding|avoid|avoiding|free of)\s+([a-z0-9 ,\-]+?)(?=$|[.;)]|\bbut\b|\bwith\b|\bk\b|\bgap\b|\band (?:k|gap|band)\b|\babove\b|\bbelow\b)", low):
        chunk = m.group(1)
        for part in re.split(r",|\bor\b|\band\b|/", chunk):
            part = re.sub(r"^(?:no|without|avoid(?:ing)?|exclud(?:e|ing)|free of)\s+", "", part.strip())
            part = re.sub(r"\s+(?:compounds?|elements?|metals?)$", "", part)
            if not part:
                continue
            got = _element(part) or _element(part.rstrip("s"))
            if not got and part in _GROUPS:
                got = _GROUPS[part]
            extra += got
    for m in re.finditer(r"\b([a-z]+)[- ]free\b", low):
        extra += _element(m.group(1))
    for e in extra:
        if e not in excluded:
            excluded.append(e)
    if extra:
        notes.append("excluding " + ", ".join(sorted(set(extra))) + " (plus the default toxic list)")
    else:
        notes.append("default exclusions applied: " + ", ".join(config.TOXIC_ELEMENTS))
    if "allow lead" in low or "lead is fine" in low:
        excluded = [e for e in excluded if e != "Pb"]

    # required anion: "oxides", "nitride"
    required: list[str] = []
    for word, sym in _ANION_WORDS.items():
        if re.search(rf"\b{word}\b", low) and not re.search(rf"\b(?:no|non|not)[- ]{word}", low):
            if sym not in required:
                required.append(sym)
    if len(required) > 1:
        notes.append(f"several chemistries mentioned ({', '.join(required)}); not restricting to one")
        required = []
    if m := find(r"(?:contain(?:ing|s)?|with|including)\s+([a-z]+)\b"):
        req = _element(m.group(1))
        if len(req) == 1 and req[0] not in excluded and req[0] not in required and m.group(1) not in {"k", "gap", "band"}:
            required.append(req[0])
    if required:
        fields["required_elements"] = required
        notes.append("must contain " + ", ".join(required))
    fields["excluded_elements"] = [e for e in excluded if e not in required]

    if "k_min" not in fields and "gap_min_eV" not in fields:
        notes.append("no numeric thresholds found: the loop will push the whole k–gap trade-off front")
    name_bits = []
    if "k_min" in fields:
        name_bits.append(f"k>{fields['k_min']:g}")
    if "gap_min_eV" in fields:
        name_bits.append(f"Eg>{fields['gap_min_eV']:g} eV")
    fields["name"] = ("Dielectric target: " + ", ".join(name_bits)) if name_bits else "Dielectric trade-off front"
    spec = TargetSpec(**fields)
    if spec.k_min is None and spec.gap_min_eV is None:
        pass
    return spec, notes
