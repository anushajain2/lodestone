"""Plain-English goal → campaign spec (T2).

A deterministic reader that handles the common phrasings. Claude offers a second reader
(:func:`lodestone.agent.parse_goal_with_claude`). Either way the result is a validated
:class:`~lodestone.schemas.CampaignSpec` that the partner confirms before anything runs.
"""

from __future__ import annotations

import re

from lodestone.schemas import CampaignSpec, SYMBOLS

SYSTEM_ALIASES = {
    "In-O-Sn-Zn": ["izto", "indium zinc tin oxide", "zinc indium tin oxide", "in-zn-sn-o", "in-sn-zn-o", "zito"],
    "N-Sn-Zn": ["zinc tin nitride", "znsnn2", "znsnn", "zn-sn-n", "zsn"],
    "Co-Ni-O-Zn": ["cobalt nickel zinc oxide", "co-ni-zn-o", "co-ni-o-zn", "cnzo", "zn-co-ni-o"],
}
PROPERTY_WORDS = {
    "conductivity": r"conduct(?:ive|ivity|ing)|sigma|σ",
    "band_gap": r"band ?gap|\bgap\b|\beg\b",
    "visible_transmittance": r"transmittance|transparen(?:t|cy)",
    "sheet_resistance": r"sheet resistance",
}
_NUM = r"(\d+(?:\.\d+)?)"


def find_system(text: str, known: list[str] | None = None) -> str | None:
    low = text.lower()
    for system, names in SYSTEM_ALIASES.items():
        if any(n in low for n in names):
            return system
    m = re.search(r"\b([A-Z][a-z]?(?:\s*-\s*[A-Z][a-z]?){1,4})\b", text)
    if m:
        els = [e.strip() for e in m.group(1).split("-")]
        if all(e in SYMBOLS for e in els):
            return "-".join(sorted(els))
    if known and len(known) == 1:
        return known[0]
    return None


def parse_goal(text: str, known_systems: list[str] | None = None) -> tuple[CampaignSpec, list[str]]:
    low = text.lower()
    notes: list[str] = []
    system = find_system(text, known_systems)
    if system is None:
        raise ValueError("which material system? e.g. 'zinc tin nitride', 'In-O-Sn-Zn' or 'cobalt nickel zinc oxide'")
    fields: dict = {"system": system, "plain_text": text.strip()}

    # window: "band gap between 1.4 and 1.6 eV", "gap of 1.4–1.6 eV", "gap in the 1.4-1.6 eV range"
    window = re.search(rf"(?:band ?gap|gap)[^\d]{{0,25}}?(?:between|from|of|in(?: the)?|within)?\s*{_NUM}\s*(?:ev)?\s*(?:and|to|-|–)\s*{_NUM}\s*ev", low)
    if window:
        lo, hi = float(window.group(1)), float(window.group(2))
        fields["window"] = {"property": "band_gap", "low": min(lo, hi), "high": max(lo, hi)}
        notes.append(f"target window: band gap {min(lo, hi):g}–{max(lo, hi):g} eV")
    else:
        objectives = []
        for prop, pat in PROPERTY_WORDS.items():
            filler = r"(?:(?!with\b|and\b|but\b|while\b|of\b)\w+\s+){0,2}"
            if re.search(rf"(?:most|highest|maxim\w*|best|high|more|widest|largest|biggest)\s+{filler}(?:{pat})", low) or re.search(rf"(?:{pat})\w*\s+(?:as high as possible|maximi[sz]ed|as well as possible)", low) or (prop == "conductivity" and re.search(r"conducts? as well as possible", low)):
                if prop == "band_gap" and re.search(rf"(?:band ?gap|gap)\s*(?:above|over|below|under|>=?|<=?|≥|≤|at least|at most)\s*{_NUM}", low) and not re.search(r"(?:most|highest|maxim\w*|widest)\s+(?:band ?gap|gap)", low):
                    continue  # a threshold, not an objective
                objectives.append({"property": prop, "direction": "max"})
            elif re.search(rf"(?:lowest|minimi\w*|least|narrowest|smallest)\s+(?:(?!with\b|and\b)\w+\s+){{0,2}}(?:{pat})", low):
                objectives.append({"property": prop, "direction": "min"})
        if not objectives:
            objectives = [{"property": "conductivity", "direction": "max"}]
            notes.append("no objective named: assuming 'maximize conductivity'")
        fields["objectives"] = objectives[:2]

        constraints = []
        if m := re.search(rf"(?:band ?gap|gap|eg)\s*(?:above|over|of at least|at least|>=|≥|>|greater than|more than)\s*{_NUM}", low):
            constraints.append({"property": "band_gap", "op": ">=", "value": float(m.group(1))})
        if m := re.search(rf"(?:band ?gap|gap|eg)\s*(?:below|under|at most|<=|≤|<|less than)\s*{_NUM}", low):
            constraints.append({"property": "band_gap", "op": "<=", "value": float(m.group(1))})
        if m := re.search(rf"transmittance\s*(?:above|over|at least|>=|≥|>)\s*{_NUM}\s*(%?)", low):
            v = float(m.group(1))
            constraints.append({"property": "visible_transmittance", "op": ">=", "value": v / 100 if (m.group(2) or v > 1) else v})
        if re.search(r"\btransparent\b", low) and not any(c["property"] == "band_gap" for c in constraints) and not any(o["property"] == "visible_transmittance" for o in objectives):
            constraints.append({"property": "band_gap", "op": ">=", "value": 3.0})
            notes.append("'transparent' read as band gap ≥ 3 eV (visible light passes through)")
        if constraints and len(fields["objectives"]) == 2:
            fields["objectives"] = fields["objectives"][:1]
            notes.append("kept one objective because constraints were given")
        fields["constraints"] = constraints
        for c in constraints:
            notes.append(f"constraint: {c['property'].replace('_', ' ')} {c['op']} {c['value']:g}")

    if m := re.search(r"(?:budget of|up to|within|in)?\s*(\d+)\s*(?:experiments|samples|runs|tests)", low):
        fields["budget"] = int(m.group(1))
    if m := re.search(r"(?:batch(?:es)? of|batch size)\s*(\d+)", low) or re.search(r"(\d+)\s*(?:per round|a round|per batch|at a time)", low):
        fields["batch_size"] = int(m.group(1))
    if re.search(r"random forest|\brf\b", low):
        fields["surrogate"] = "rf"
    if re.search(r"\bgrid\b", low):
        fields["strategy"] = "grid"
    elif re.search(r"\brandom(?:ly)? (?:search|baseline|picks?)\b", low):
        fields["strategy"] = "random"
    if "budget" in fields and fields["budget"] < 10:
        fields["initial"] = max(1, fields["budget"] // 2)

    obj_text = (
        f"band gap {fields['window']['low']:g}–{fields['window']['high']:g} eV" if "window" in fields
        else " + ".join(("max " if o["direction"] == "max" else "min ") + o["property"].replace("_", " ") for o in fields["objectives"])
    )
    fields["name"] = f"{system}: {obj_text}"[:120]
    return CampaignSpec(**fields), notes
