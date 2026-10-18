"""Lodestone dashboard.  Run:  uv run lodestone dashboard"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

from lodestone import config
from lodestone.agent import has_anthropic_credentials
from lodestone.campaign import CampaignError, CampaignService
from lodestone.connectors import htem
from lodestone.goals import parse_goal
from lodestone.loop.problem import untransform
from lodestone.onboarding import OnboardingError, import_htem_library, import_partner_csv
from lodestone.schemas import PROPERTIES, CampaignSpec
from lodestone.store import Database, NotFound

st.set_page_config(page_title="Lodestone", page_icon="🧪", layout="wide")
ACCENT = "#2f6fdf"
COLORS = {"campaign": "#2f6fdf", "random": "#8a8f98", "grid": "#d08a2c", "bayes-gp": "#2f6fdf", "bayes-rf": "#18a07a"}
LABELS = {"random": "Random", "grid": "Grid sweep", "bayes-gp": "Bayesian (GP)", "bayes-rf": "Bayesian (random forest)"}
PARTNER_COLORS = {"partner-a": "#2f6fdf", "partner-b": "#b4469b"}

st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; max-width: 1400px;}
.mf-badge {display:inline-block;padding:2px 10px;border-radius:999px;font-size:0.8rem;font-weight:600;color:white;}
.mf-note {color:#6b7280;font-size:0.85rem;}
.mf-why {font-size:0.86rem;line-height:1.35;}
.mf-save {background:#ecfdf5;border:1px solid #a7f3d0;border-radius:10px;padding:12px 16px;}
div[data-testid="stMetricValue"] {font-size:1.5rem;}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def db() -> Database:
    return Database()


def scope_for(pid: str):
    return db().scope(pid, actor="dashboard")


platform = db().platform()
partners = platform.list_partners()
if not partners:
    st.title("Lodestone")
    st.warning("No partners yet. Run `uv run lodestone seed` to create the demo labs, then reload.")
    st.stop()

# ----------------------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🧪 Lodestone")
    st.caption("Your lab's experiments in, an AI-driven discovery loop out.")
    names = {p["partner_id"]: p["name"] for p in partners}
    pid = st.selectbox("Lab workspace", list(names), format_func=names.get, key="partner")
    st.markdown(f'<span class="mf-badge" style="background:{PARTNER_COLORS.get(pid, ACCENT)}">signed in as {pid}</span>', unsafe_allow_html=True)
    scope = scope_for(pid)
    svc = CampaignService(scope)
    libraries = scope.list_libraries()
    campaigns = scope.list_campaigns()
    systems = scope.systems()
    rows_seen = libraries + campaigns
    foreign = sum(r["partner_id"] != pid for r in rows_seen)
    st.markdown("**What this workspace can see**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Libraries", len(libraries))
    c2.metric("Samples", sum(systems.values()))
    c3.metric("Campaigns", len(campaigns))
    st.markdown(
        f"<span class='mf-note'>Rows from other labs visible: <b style='color:{'#18a07a' if foreign == 0 else '#c0392b'}'>{foreign}</b>. "
        "Every query goes through one lab-scoped store.</span>", unsafe_allow_html=True)
    st.divider()
    if campaigns:
        cnames = {c["campaign_id"]: f"{c['name']} · {c['status']}" for c in campaigns}
        cid = st.selectbox("Campaign", list(cnames), format_func=cnames.get, key=f"cmp-{pid}")
    else:
        cid = None
        st.info("No campaigns yet: set a goal on the Goal tab.")
    st.divider()
    st.caption("Lab data: NREL HTEM DB (real thin films). The lab replay returns real past measurements, so the loop chooses among samples that were actually made.")

st.markdown(f"## {names[pid]}")
tabs = st.tabs(["① Lab data", "② Goal", "③ Campaign", "④ Evidence", "⑤ Agent", "⑥ Audit & isolation"])


def fmtv(prop: str, v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    unit = PROPERTIES[prop]["unit"]
    if prop == "conductivity":
        return f"{v:,.0f} {unit}" if v >= 100 else f"{v:.3g} {unit}"
    if unit == "fraction":
        return f"{100 * v:.0f}%"
    return f"{v:.2f} {unit}"


# ------------------------------------------------------------------------------------ ① lab data
with tabs[0]:
    st.markdown("#### Your lab's data in: validated, in one format, every number traceable")
    left, right = st.columns([1, 1.35])
    with left:
        with st.expander("Import from the HTEM DB connector", expanded=not libraries):
            avail = htem.available_systems()
            sys_pick = st.selectbox("Material system", avail, format_func=lambda s: f"{s} · {htem.DEMO_SYSTEMS.get(s, '')[:60]}")
            snap = htem.load_snapshot(sys_pick)
            have = {lib["source_library_id"] for lib in libraries}
            opts = [lib for lib in snap if str(lib["id"]) not in have]
            lib_pick = st.multiselect("Libraries", [lib["id"] for lib in opts],
                                      format_func=lambda i: next(f"{i} · {len(l.get('samples') or [])} samples · {l.get('deposition_initial_temp_c')} °C" for l in opts if l["id"] == i))
            if st.button("Import selected", type="primary", disabled=not lib_pick):
                for i in lib_pick:
                    try:
                        rep = import_htem_library(scope, i)
                        st.toast(f"Library {i}: {rep.accepted} accepted, {rep.rejected} rejected")
                    except OnboardingError as e:
                        st.error(str(e))
                st.rerun()
        with st.expander("Upload another lab's CSV export"):
            st.caption("Columns like `SampleID, Library, Composition [at.%], Substrate T [K], Rho [mOhm*cm], Optical edge [nm], T_vis [%], Thickness [nm], Date`. Units are read from the headers.")
            up = st.file_uploader("CSV file", type=["csv"])
            up_sys = st.selectbox("System", sorted(set(htem.available_systems()) | set(systems)), key="csv_sys")
            if up is not None and st.button("Import CSV"):
                try:
                    reps = import_partner_csv(scope, up.getvalue().decode(), up_sys, up.name)
                    st.toast(f"{sum(r.accepted for r in reps)} accepted, {sum(r.rejected for r in reps)} rejected")
                    st.rerun()
                except OnboardingError as e:
                    st.error(str(e))
        if libraries:
            lib_df = pd.DataFrame([{"library": l["library_ref"], "system": l["system"], "source": l["source"], "accepted": l["accepted"], "rejected": l["rejected"],
                                    "temp °C": (l.get("synthesis") or {}).get("deposition_temp_c")} for l in libraries])
            st.markdown("**Imported libraries**")
            st.dataframe(lib_df, hide_index=True, use_container_width=True, height=280)
            tot_acc, tot_rej = lib_df["accepted"].sum(), lib_df["rejected"].sum()
            st.caption(f"{tot_acc:,} records accepted, {tot_rej:,} rejected with a reason, across {len(lib_df)} libraries.")
    with right:
        if libraries:
            ref = st.selectbox("Import report for", [l["library_ref"] for l in libraries], format_func=lambda r: f"{r} ({next(l['system'] for l in libraries if l['library_ref'] == r)})")
            rep = scope.get_library(ref)["report"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Received", rep["received"])
            m2.metric("Accepted", rep["accepted"])
            m3.metric("Rejected", rep["rejected"])
            if rep["issues"]:
                st.dataframe(pd.DataFrame(rep["issues"])[["source_sample_id", "level", "field", "reason"]], hide_index=True, use_container_width=True, height=180)
            else:
                st.success("Every record passed validation.")
            recs = scope.list_records(library_ref=ref)
            if recs:
                st.markdown("**Records** (pick one to see where every number came from)")
                rdf = pd.DataFrame([{"record": r["record_id"], "composition": " ".join(f"{k}{v:.2f}" for k, v in r["composition"].items()),
                                     "σ (S/cm)": r["measurements"].get("conductivity", {}).get("value"), "gap (eV)": r["measurements"].get("band_gap", {}).get("value"),
                                     "T_vis": r["measurements"].get("visible_transmittance", {}).get("value"), "missing": ", ".join(r["missing"])} for r in recs])
                st.dataframe(rdf, hide_index=True, use_container_width=True, height=200)
                pick = st.selectbox("Provenance for", [r["record_id"] for r in recs])
                rec = scope.get_record(pick)
                p = rec["provenance"]
                st.markdown(f"Source **{p['source']}** · sample `{p['source_sample_id']}` · library `{p['source_library_id']}` · spot {p.get('position')} · adapter `{p['adapter']}`"
                            + (f" · [open in HTEM API]({p['url']})" if p.get("url") else ""))
                a, b = st.columns(2)
                a.json({"composition": rec["composition"], "measurements": rec["measurements"], "missing": rec["missing"]}, expanded=True)
                b.json({"synthesis": rec["synthesis"], "provenance": p}, expanded=False)


# ------------------------------------------------------------------------------------------ ② goal
with tabs[1]:
    if not systems:
        st.info("Import some lab data first (Lab data tab).")
    else:
        left, right = st.columns([1, 1.1])
        with left:
            st.markdown("#### Say what you want")
            text = st.text_area("Goal in plain English", st.session_state.get("goal_text", "most conductive zinc tin nitride, 40 experiments"), height=90,
                                help="e.g. 'transparent conductor: maximize conductivity with band gap above 3 eV' or 'band gap between 1.4 and 1.6 eV'")
            use_claude = st.toggle("Read with Claude", value=False, disabled=not has_anthropic_credentials(), help="Needs ANTHROPIC_API_KEY; otherwise a deterministic reader is used.")
            if st.button("Read goal", type="primary"):
                try:
                    if use_claude:
                        from lodestone.agent import parse_goal_with_claude

                        spec, notes = parse_goal_with_claude(text)
                    else:
                        spec, notes = parse_goal(text, list(systems))
                    st.session_state.update({"draft": spec.model_dump(), "draft_notes": notes, "goal_text": text})
                except Exception as e:
                    st.error(f"Couldn't read that: {e}")
            for n in st.session_state.get("draft_notes", []):
                st.markdown(f"- {n}")
        with right:
            st.markdown("#### Campaign spec (edit, then confirm)")
            d = st.session_state.get("draft") or {"name": "New campaign", "system": next(iter(systems)), "objectives": [{"property": "conductivity", "direction": "max"}], "constraints": [], "window": None, "budget": 40, "batch_size": 5, "strategy": "bayes", "surrogate": "gp", "failure_rate": 0.05, "delay_rounds": 0, "cost_per_experiment": 1500.0}
            with st.form("spec"):
                name = st.text_input("Name", d["name"])
                sys_opts = sorted(systems)
                system = st.selectbox("Material system", sys_opts, index=sys_opts.index(d["system"]) if d["system"] in sys_opts else 0)
                kind = st.radio("Goal type", ["maximize / minimize", "hit a band-gap window"], index=1 if d.get("window") else 0, horizontal=True)
                c1, c2 = st.columns(2)
                obj = d["objectives"][0] if d.get("objectives") else {"property": "conductivity", "direction": "max"}
                props = ["conductivity", "band_gap", "visible_transmittance", "sheet_resistance"]
                prop = c1.selectbox("Property", props, index=props.index(obj["property"]), format_func=lambda p: PROPERTIES[p]["label"])
                direction = c2.selectbox("Direction", ["max", "min"], index=0 if obj["direction"] == "max" else 1)
                cons = {c["op"]: c["value"] for c in d.get("constraints", []) if c["property"] == "band_gap"}
                c3, c4 = st.columns(2)
                gmin = c3.number_input("Band gap at least (eV, 0 = none)", 0.0, 8.0, float(cons.get(">=", 0.0)), step=0.1)
                gmax = c4.number_input("Band gap at most (eV, 0 = none)", 0.0, 8.0, float(cons.get("<=", 0.0)), step=0.1)
                w = d.get("window") or {"low": 1.4, "high": 1.6}
                c5, c6 = st.columns(2)
                wlo = c5.number_input("Window low (eV)", 0.1, 8.0, float(w["low"]), step=0.05)
                whi = c6.number_input("Window high (eV)", 0.1, 8.0, float(w["high"]), step=0.05)
                c7, c8, c9 = st.columns(3)
                budget = c7.number_input("Budget (experiments)", 5, 500, int(d["budget"]))
                batch = c8.number_input("Per round", 1, 20, int(d["batch_size"]))
                cost = c9.number_input("Cost per experiment ($)", 0.0, 1e6, float(d.get("cost_per_experiment", 1500.0)), step=100.0)
                c10, c11, c12, c13 = st.columns(4)
                strategy = c10.selectbox("Strategy", ["bayes", "random", "grid"], index=["bayes", "random", "grid"].index(d.get("strategy", "bayes")))
                surrogate = c11.selectbox("Model", ["gp", "rf"], index=0 if d.get("surrogate", "gp") == "gp" else 1, format_func={"gp": "Gaussian process", "rf": "Random forest"}.get)
                fail = c12.slider("Lab failure rate", 0.0, 0.5, float(d.get("failure_rate", 0.05)), 0.05)
                delay = c13.number_input("Delay (rounds)", 0, 5, int(d.get("delay_rounds", 0)))
                go_ = st.form_submit_button("Confirm & start campaign", type="primary")
            if go_:
                try:
                    fields = {"name": name, "system": system, "budget": budget, "batch_size": batch, "strategy": strategy, "surrogate": surrogate,
                              "failure_rate": fail, "delay_rounds": delay, "cost_per_experiment": cost, "plain_text": st.session_state.get("goal_text"),
                              "seed": int(np.random.default_rng().integers(1000)), "initial": min(10, budget)}
                    if kind.startswith("hit"):
                        fields["window"] = {"property": "band_gap", "low": wlo, "high": whi}
                    else:
                        fields["objectives"] = [{"property": prop, "direction": direction}]
                        fields["constraints"] = ([{"property": "band_gap", "op": ">=", "value": gmin}] if gmin else []) + ([{"property": "band_gap", "op": "<=", "value": gmax}] if gmax else [])
                    c = svc.create(CampaignSpec(**fields))
                    st.session_state[f"cmp-{pid}"] = c["campaign_id"]
                    st.toast("Campaign created")
                    st.rerun()
                except (CampaignError, ValueError) as e:
                    st.error(str(e))


# -------------------------------------------------------------------------------------- ③ campaign
@st.cache_data(show_spinner="Replaying random and grid search on the same data…")
def comparison(pid_: str, cid_: str, seeds: int) -> dict:
    return CampaignService(scope_for(pid_)).compare_with_baselines(cid_, seeds=seeds)


with tabs[2]:
    if not cid:
        st.info("No campaign selected.")
    else:
        spec = svc.spec(cid)
        status = svc.status(cid)
        best = status["best"]
        st.caption(f"Goal: {status['target']}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Round", status["round"])
        m2.metric("Experiments", f"{status['budget_used']}/{status['budget']}")
        m3.metric("Failed", status["experiments_by_status"].get("failed", 0))
        m4.metric("Best so far", best["text"] if best else "—")
        m5.metric("Best's rank in the data", f"#{best['rank_in_data']} of {best['of']}" if best else "—", help="Hindsight from the replay: how the best sample found ranks among every sample that was ever made")
        b1, b2, b3 = st.columns([1, 1, 2])
        active = status["status"] == "active"
        if b1.button("Run one round", disabled=not active, use_container_width=True, type="primary"):
            with st.spinner("Choose → run in the lab → learn"):
                svc.run_round(cid)
            st.rerun()
        if b2.button("Run to end of budget", disabled=not active, use_container_width=True):
            with st.spinner("Running the campaign…"):
                svc.run(cid)
            st.rerun()

        left, right = st.columns([1.5, 1])
        with left:
            st.markdown("#### Best result against experiments used")
            show_base = st.toggle("Compare with random and grid search on the same data", value=True)
            curve = svc.curve(cid)
            fig = go.Figure()

            def to_nat(scores):
                arr = np.array([np.nan if s is None else s for s in scores], float)
                if spec.window:
                    return -arr, f"Distance outside {spec.window.low:g}–{spec.window.high:g} eV (eV, lower is better)", False
                o = spec.objectives[0]
                return untransform(o.property, arr * (1 if o.direction == "max" else -1)), f"Best {PROPERTIES[o.property]['label'].lower()} ({PROPERTIES[o.property]['unit']})", PROPERTIES[o.property]["log"]

            comp = None
            ylabel, logy = "", False
            if show_base and spec.kind != "pareto":
                comp = comparison(pid, cid, 5)
                for m, vals in comp["curves"].items():
                    y, ylabel, logy = to_nat(vals)
                    fig.add_trace(go.Scatter(x=list(range(1, len(y) + 1)), y=y, name=f"{LABELS.get(m, m)} (mean of 5 replays)", line=dict(color=COLORS.get(m, "#999"), width=1.5, dash="dot")))
            if curve and spec.kind != "pareto":
                y, ylabel, logy = to_nat([c["best_score"] for c in curve])
                fig.add_trace(go.Scatter(x=[c["experiments"] for c in curve], y=y, name="This campaign", mode="lines+markers", line=dict(color="#111827", width=3)))
                prob = svc.problem(cid)
                sc = prob.true_scores()
                if np.isfinite(sc).any():
                    yb, _, _ = to_nat([float(np.max(sc[np.isfinite(sc)]))])
                    fig.add_hline(y=yb[0], line=dict(color="#18a07a", dash="dash", width=1), annotation_text="best sample in the data", annotation_font_color="#18a07a")
            fig.update_yaxes(type="log" if logy else "linear", title=ylabel)
            fig.update_layout(height=420, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Experiments used", legend=dict(orientation="h", y=1.1, x=0))
            st.plotly_chart(fig, use_container_width=True)
            if comp:
                meth = comp["methods"]
                k = comp["headline_k"]
                saved = comp.get("experiments_saved_vs") or {}
                if "bayes-gp" in meth:
                    lines = [f"To find **{k} of the top-5%** samples ({comp['top_size']} of {comp['pool']}), the median replay needed "
                             + ", ".join(f"**{v['experiments_to_top_median']:.0f}** experiments with {LABELS.get(m, m).lower()}" for m, v in meth.items()) + "."]
                    best_save = max(saved.values()) if saved else 0
                    if best_save > 0:
                        lines.append(f"That is up to **{best_save:.0f} fewer experiments** than the baselines, about **\\${best_save * spec.cost_per_experiment:,.0f}** at \\${spec.cost_per_experiment:,.0f} per experiment.")
                    else:
                        lines.append("On this goal the optimizer did **not** beat the baselines; that happens when the property barely varies or a sweep already covers it.")
                    (st.success if best_save > 0 else st.info)(" ".join(lines))
                st.caption(f"Medians over {comp['seeds']} replays of each strategy, same budget ({comp['budget']}) and seed experiments; 'not found' counts as budget + 1.")

        with right:
            st.markdown("#### This round's picks: why each sample")
            cands = scope.list_candidates(cid)
            if cands:
                latest = max(c["round"] for c in cands)
                for c in [c for c in cands if c["round"] == latest]:
                    with st.container(border=True):
                        st.markdown(f"**{c['label']}** <span class='mf-note'>· {c['record_id']} · {c['status']}</span>", unsafe_allow_html=True)
                        st.markdown(f"<div class='mf-why'>{c['why']}</div>", unsafe_allow_html=True)
                if any(any(isinstance(v, dict) and "std_inflation" in v for v in (c.get("prediction") or {}).values()) for c in cands if c["round"] == latest):
                    st.caption("Ranges are 68% intervals calibrated on held-out data: the model's ±1σ, widened by a factor learned on held-out samples of other material systems "
                               "(reports/calibration). The optimizer itself ranks with the unwidened model.")
            else:
                st.caption("Run a round to see the optimizer's picks.")

        st.markdown("#### Every experiment, with provenance")
        res = svc.results_table(cid)
        if res:
            df = pd.DataFrame(res)
            cols = ["n", "round", "label", "status", "conductivity", "band_gap", "visible_transmittance", "reason", "htem_sample_id", "library", "url"]
            df = df[[c for c in cols if c in df.columns]]
            st.dataframe(df, hide_index=True, use_container_width=True, height=260,
                         column_config={"url": st.column_config.LinkColumn("source", display_text="HTEM API"), "conductivity": st.column_config.NumberColumn("σ (S/cm)", format="%.4g"),
                                        "band_gap": st.column_config.NumberColumn("gap (eV)", format="%.2f"), "visible_transmittance": st.column_config.NumberColumn("T_vis", format="%.2f")})
        with st.expander("What would the model try next? (new recipes, not in the data: unverified)"):
            if st.button("Suggest recipes"):
                try:
                    for r in svc.suggest_recipes(cid, 5):
                        pr = "; ".join(f"{PROPERTIES[p]['label'].lower()} ≈ {fmtv(p, v['mean'])} ({fmtv(p, v['low'])}–{fmtv(p, v['high'])})" for p, v in r["predicted"].items())
                        st.markdown(f"- **{r['text']}**: {pr}")
                    st.caption("Predictions from the campaign's model; ranges are 68% intervals calibrated on held-out data. These films were never made; they need a real experiment.")
                except CampaignError as e:
                    st.warning(str(e))
        with st.expander("Export (FAIR-style JSON)"):
            from lodestone.export import export_campaign

            st.download_button("Download campaign JSON", json.dumps(export_campaign(scope, cid), indent=2, default=str), file_name=f"{cid}.json", mime="application/json")


# -------------------------------------------------------------------------------------- ④ evidence
with tabs[3]:
    st.markdown("#### Does the optimizer find the best samples with fewer experiments? (benchmark on real HTEM data)")
    path = config.REPORTS_DIR / "benchmark"
    if (path / "summary.json").exists():
        summ = json.loads((path / "summary.json").read_text())
        curves = json.loads((path / "curves.json").read_text())
        key = st.selectbox("Campaign", list(summ["campaigns"]), format_func=lambda k: summ["campaigns"][k]["spec"]["name"])
        camp = summ["campaigns"][key]
        sp = CampaignSpec(**camp["spec"])
        f = go.Figure()
        ylabel, logy = "", False
        for m, cs in curves[key].items():
            arr = np.array(cs, float)
            arr[~np.isfinite(arr)] = np.nan
            if sp.window:
                nat, ylabel, logy = -arr, "Distance outside the window (eV)", False
            else:
                o = sp.objectives[0]
                nat, ylabel, logy = untransform(o.property, arr), f"Best {PROPERTIES[o.property]['label'].lower()} ({PROPERTIES[o.property]['unit']})", PROPERTIES[o.property]["log"]
            med = np.nanmedian(nat, axis=0)
            f.add_trace(go.Scatter(x=list(range(1, len(med) + 1)), y=med, name=LABELS.get(m, m), line=dict(color=COLORS.get(m), width=3)))
        f.update_yaxes(type="log" if logy else "linear", title=ylabel)
        f.update_layout(height=380, template="plotly_white", xaxis_title="Experiments", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(f, use_container_width=True)
        k = camp["headline_k"]
        st.dataframe(pd.DataFrame([{"method": LABELS.get(m, m), f"experiments to find {k} of top 5%": v["experiments_to_top_median"], "runs that got there": f"{v['found_top_in_budget']}/{v['runs']}",
                                    "first top-1% sample": v["experiments_to_first_top1_median"], "top-5% samples found": round(v["top_found_mean"], 1),
                                    "failed experiments": round(v["failures_mean"], 1)} for m, v in camp["methods"].items()]), hide_index=True, use_container_width=True)
        st.caption(f"{summ['config']['seeds']} seeds · pool {camp['pool']} real samples · top 5% = {camp['top_size']} samples · medians; 'not found' = budget + 1.")
        st.image(str(path / "benchmark.png"), use_container_width=True)
    else:
        st.caption("Run `uv run lodestone bench` to produce the benchmark.")
    a, b = st.columns(2)
    with a:
        p = config.REPORTS_DIR / "onboarding" / "onboarding.json"
        if p.exists():
            o = json.loads(p.read_text())
            st.markdown("#### Data onboarding")
            st.dataframe(pd.DataFrame([{"system": s, **{k: v for k, v in t.items() if k != "reasons"}} for s, t in o["systems"].items()]), hide_index=True)
            br = o["broken_records"]
            st.markdown(f"Deliberately broken records handled correctly: **{br['correct']}/{br['cases']}**")
    with b:
        for pol in ("claude", "scripted"):
            p = config.REPORTS_DIR / "agent_eval" / f"{pol}.json"
            if p.exists():
                s = json.loads(p.read_text())["summary"]
                st.markdown(f"#### Agent ({pol})")
                st.markdown(f"Success {s['success_rate']:.0%} over {s['tasks']} tasks · tasks with invented numbers: {s['tasks_with_invented_numbers']}")
                st.dataframe(pd.DataFrame([{"kind": k, **v} for k, v in s["by_kind"].items()]), hide_index=True)


# ----------------------------------------------------------------------------------------- ⑤ agent
with tabs[4]:
    st.markdown("#### Claude runs a campaign end to end through the MCP tools")
    has_key = has_anthropic_credentials()
    if not has_key:
        st.info("No Anthropic credentials found (set ANTHROPIC_API_KEY and restart). The scripted reference policy drives the same MCP tools without an LLM.")
    task = st.text_area("Task", "Run a campaign to find the most conductive zinc tin nitride with 30 experiments, then tell me the best sample and whether Bayesian optimization beat random search.", height=80)
    c1, c2 = st.columns([1, 3])
    policy = c1.radio("Policy", ["claude", "scripted reference"], index=0 if has_key else 1, disabled=not has_key)
    if c1.button("Run agent", type="primary"):
        from lodestone.agent import run_claude, run_scripted, run_sync

        log = c2.empty()
        lines: list[str] = []

        def on_event(ev):
            lines.append(f"{'✓' if ev.ok else '✗'} {ev.name}({json.dumps(ev.input)[:100]})")
            log.code("\n".join(lines[-25:]))

        with st.spinner("Agent working…"):
            if policy == "claude":
                run = run_sync(run_claude(task, partner_id=pid, on_event=on_event))
            else:
                run = run_sync(run_scripted({"kind": "campaign", "create": {"goal": task}, "compare": True}, task, partner_id=pid, on_event=on_event))
        scope.save_agent_run((run.final_json or {}).get("campaign_id"), run.transcript(), run.metrics())
        st.rerun()
    for r in scope.list_agent_runs()[:5]:
        m = r["metrics"]
        with st.expander(f"{r['created_at']} · {m['policy']} · {m['tool_calls']} tool calls · {m['valid_tool_calls']} valid · invented numbers: {len(m.get('invented_numbers', []))}"):
            st.markdown(r["transcript"][-1]["final_text"].split("```json")[0])
            st.dataframe(pd.DataFrame(r["transcript"][:-1])[["name", "ok", "seconds"]], hide_index=True)


# ------------------------------------------------------------------------------ ⑥ audit & isolation
with tabs[5]:
    a, b = st.columns([1.2, 1])
    with a:
        st.markdown("#### Audit log (this workspace)")
        log = scope.list_audit(300)
        if log:
            st.dataframe(pd.DataFrame(log)[["ts", "actor", "action", "object_type", "object_id", "outcome", "detail"]], hide_index=True, use_container_width=True, height=440)
    with b:
        st.markdown("#### Live isolation probe")
        st.caption("Platform view: take one real id from every *other* lab's tables and try to read it through this workspace.")
        if st.button("Run probe"):
            readers = {"libraries": scope.get_library, "records": scope.get_record, "campaigns": scope.get_campaign, "candidates": scope.get_candidate,
                       "experiments": scope.get_experiment, "results": scope.get_result, "agent_runs": scope.get_agent_run}
            rows = []
            for table, oid in platform.probe_ids(pid).items():
                if table in readers:
                    try:
                        readers[table](oid)
                        rows.append({"object": table, "id": oid, "result": "LEAKED"})
                    except NotFound:
                        rows.append({"object": table, "id": oid, "result": "blocked (not found)"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.success(f"{sum(r['result'].startswith('blocked') for r in rows)}/{len(rows)} cross-lab reads blocked; each attempt is in the audit log as 'denied'.")
        st.markdown("#### Rows per lab (counts only)")
        rep = pd.DataFrame(platform.isolation_report())
        if len(rep):
            st.dataframe(rep.pivot(index="table", columns="partner_id", values="rows").fillna(0).astype(int), use_container_width=True)
