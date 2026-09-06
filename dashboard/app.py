"""
Adversarial AI Safety Lab — Scientific Dashboard

Reads all result files from results/ and displays:
- Cross-model leaderboard
- Failure taxonomy breakdown
- Boundary distance maps
- Multi-turn stress depth
- Decision boundary search results
- Trace explorer
"""

import json
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


# ── Data Loading ──────────────────────────────────────────────

def clean_obj(obj):
    """Recursively strip whitespace from keys and string values."""
    if isinstance(obj, dict):
        return {str(k).strip(): clean_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_obj(item) for item in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return default


def load_all_reports():
    """Load all JSON reports from the results directory."""
    reports = {
        "reverse_arena": [],
        "jailbreak": [],
        "taxonomy": [],
        "boundary_distance": [],
        "multiturn": [],
        "boundary_search": [],
    }

    if not RESULTS_DIR.exists():
        return reports

    for path in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = clean_obj(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue

        step = data.get("step", "")
        filename = path.name

        if "step10" in step or "reverse_arena" in filename:
            reports["reverse_arena"].append({"file": filename, "data": data})
        elif "step12" in step or "jailbreak" in filename:
            reports["jailbreak"].append({"file": filename, "data": data})
        elif "step13" in step or "taxonomy" in filename:
            reports["taxonomy"].append({"file": filename, "data": data})
        elif "step14" in step or "boundary_distance" in filename:
            reports["boundary_distance"].append({"file": filename, "data": data})
        elif "step15" in step or "multiturn" in filename:
            reports["multiturn"].append({"file": filename, "data": data})
        elif "step16" in step or "boundary_search" in filename:
            reports["boundary_search"].append({"file": filename, "data": data})

    return reports


# ── Leaderboard Builder ──────────────────────────────────────

def build_leaderboard(reports):
    """Build a cross-model leaderboard from all available reports."""
    rows = []

    # From Step 10 (Reverse Arena)
    for item in reports["reverse_arena"]:
        data = item["data"]
        metrics = data.get("metrics", {})
        stats = data.get("stats", {})
        target = data.get("target_model", data.get("target", "unknown"))
        attacker = data.get("attacker_model", "unknown")

        rows.append({
            "file": item["file"],
            "source": "Reverse Arena (Step 10)",
            "target_model": target,
            "attacker_model": attacker,
            "total_attacks": safe_int(stats.get("total_attack_attempts", 0)),
            "full_evasion": safe_int(stats.get("full_evasion", 0)),
            "partial_evasion": safe_int(stats.get("partial_evasion", 0)),
            "contained": safe_int(stats.get("contained", 0)),
            "over_contained": safe_int(stats.get("over_contained", 0)),
            "attack_success_rate": safe_float(metrics.get("attack_success_rate", 0)),
            "evasion_score": safe_float(metrics.get("evasion_score", 0)),
            "safety_under_attack": safe_float(metrics.get("safety_under_attack", 0)),
            "risk_percent": safe_float(metrics.get("risk_percent", 0)),
        })

    # From Step 16 (Boundary Search)
    for item in reports["boundary_search"]:
        data = item["data"]
        report = data.get("report", {})
        target = data.get("target_model", "unknown")
        attacker = data.get("attacker_model", "unknown")

        rows.append({
            "file": item["file"],
            "source": "Boundary Search (Step 16)",
            "target_model": target,
            "attacker_model": attacker,
            "total_attacks": safe_int(report.get("total_seeds", 0)),
            "full_evasion": safe_int(report.get("fully_collapsed", 0)),
            "partial_evasion": safe_int(report.get("degraded", 0)) - safe_int(report.get("fully_collapsed", 0)),
            "contained": safe_int(report.get("total_seeds", 0)) - safe_int(report.get("degraded", 0)),
            "over_contained": 0,
            "attack_success_rate": safe_float(report.get("collapse_rate", 0)),
            "evasion_score": safe_float(report.get("degradation_rate", 0)),
            "safety_under_attack": 100 - safe_float(report.get("collapse_rate", 0)),
            "risk_percent": safe_float(report.get("collapse_rate", 0)),
        })

    # From Step 12 (Jailbreak)
    for item in reports["jailbreak"]:
        data = item["data"]
        target = data.get("target_model", "unknown")
        attacker = data.get("attacker_model", "unknown")
        total = safe_int(data.get("total_seeds", 0))
        successes = safe_int(data.get("successful_jailbreaks", 0))
        asr = safe_float(data.get("jailbreak_success_rate", 0))

        rows.append({
            "file": item["file"],
            "source": "Jailbreak (Step 12)",
            "target_model": target,
            "attacker_model": attacker,
            "total_attacks": total,
            "full_evasion": successes,
            "partial_evasion": 0,
            "contained": total - successes,
            "over_contained": 0,
            "attack_success_rate": asr,
            "evasion_score": asr,
            "safety_under_attack": 100 - asr,
            "risk_percent": asr,
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ── Taxonomy Builder ─────────────────────────────────────────

def build_taxonomy_view(reports):
    """Build taxonomy breakdown from Step 13 results."""
    rows = []

    for item in reports["taxonomy"]:
        data = item["data"]
        report = data.get("report", {})
        target = data.get("target_model", "unknown")
        cat_counts = report.get("category_counts", {})

        for category, count in cat_counts.items():
            if count > 0:
                rows.append({
                    "file": item["file"],
                    "target_model": target,
                    "category": category,
                    "count": count,
                })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ── Boundary Search Detail ───────────────────────────────────

def build_boundary_chains(reports):
    """Build transformation chain details from Step 16."""
    rows = []

    for item in reports["boundary_search"]:
        data = item["data"]
        target = data.get("target_model", "unknown")
        results = data.get("results", [])

        for r in results:
            chain = r.get("transformation_chain", [])
            chain_str = " → ".join(
                f"{step.get('strategy', '?')}: {step.get('prev_action', '?')}→{step.get('curr_action', '?')}"
                for step in chain
            )

            rows.append({
                "file": item["file"],
                "target_model": target,
                "seed_id": r.get("seed_id", ""),
                "seed_category": r.get("seed_category", ""),
                "seed_severity": r.get("seed_severity", 0),
                "initial_action": r.get("initial_action", ""),
                "final_action": r.get("final_action", ""),
                "full_collapse": r.get("full_collapse", False),
                "chain_length": r.get("chain_length", 0),
                "total_reward": safe_float(r.get("total_reward", 0)),
                "chain_summary": chain_str,
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ── Multi-Turn Detail ────────────────────────────────────────

def build_multiturn_view(reports):
    """Build multi-turn pressure details from Step 15."""
    rows = []

    for item in reports["multiturn"]:
        data = item["data"]
        target = data.get("target_model", "unknown")
        results = data.get("results", [])
        report = data.get("report", {})

        for r in results:
            rows.append({
                "file": item["file"],
                "target_model": target,
                "seed_id": r.get("seed_id", ""),
                "seed_category": r.get("seed_category", ""),
                "initial_action": r.get("initial_action", ""),
                "final_action": r.get("final_action", ""),
                "safety_stress_depth": r.get("safety_stress_depth"),
                "total_turns": r.get("total_turns", 0),
                "failed": r.get("failed", False),
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ── UI ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Adversarial AI Safety Lab",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Adversarial AI Safety Lab")
st.caption(
    "Scientific adversarial robustness benchmarking for LLM moderation systems. "
    "Cross-model evaluation with failure taxonomy, boundary distance mapping, and decision boundary search."
)

# Load data
with st.spinner("Loading reports..."):
    reports = load_all_reports()

total_reports = sum(len(v) for v in reports.values())

if total_reports == 0:
    st.warning(
        "No result files found in `results/`. "
        "Run the evaluation pipeline first:\n\n"
        "```bash\npython steps/step17_batch_runner.py granite4.2:3b\n```"
    )
    st.stop()

st.sidebar.header("Controls")
st.sidebar.info(f"📁 {total_reports} reports loaded")

if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

# Build views
leaderboard = build_leaderboard(reports)
taxonomy_df = build_taxonomy_view(reports)
boundary_chains = build_boundary_chains(reports)
multiturn_df = build_multiturn_view(reports)

# ── Tabs ──────────────────────────────────────────────────────

tab_leaderboard, tab_taxonomy, tab_boundary, tab_multiturn, tab_chains, tab_explorer = st.tabs([
    "🏆 Leaderboard",
    "📊 Failure Taxonomy",
    "📉 Boundary Distance",
    "🔄 Multi-Turn Pressure",
    "🔗 Boundary Search Chains",
    "🔍 Trace Explorer",
])

# ── Tab 1: Leaderboard ───────────────────────────────────────

with tab_leaderboard:
    st.subheader("Cross-Model Leaderboard")

    if leaderboard.empty:
        st.info("No leaderboard data available yet.")
    else:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Models Tested", leaderboard["target_model"].nunique())
        col2.metric("Total Reports", len(leaderboard))
        col3.metric("Total Attacks", int(leaderboard["total_attacks"].sum()))
        col4.metric("Avg Safety Score", f"{leaderboard['safety_under_attack'].mean():.1f}%")

        # Sort controls
        sort_col = st.selectbox(
            "Sort by",
            options=["safety_under_attack", "attack_success_rate", "evasion_score", "total_attacks"],
            index=0,
        )
        ascending = sort_col in ["attack_success_rate", "evasion_score"]

        sorted_lb = leaderboard.sort_values(sort_col, ascending=ascending)

        # Display table
        display_cols = [
            "target_model", "source", "total_attacks",
            "full_evasion", "partial_evasion", "contained",
            "attack_success_rate", "safety_under_attack",
        ]
        available_cols = [c for c in display_cols if c in sorted_lb.columns]
        st.dataframe(sorted_lb[available_cols], use_container_width=True, hide_index=True)

        # Chart
        st.subheader("Safety Score Comparison")
        chart_data = sorted_lb.set_index("target_model")["safety_under_attack"]
        st.bar_chart(chart_data)

# ── Tab 2: Failure Taxonomy ──────────────────────────────────

with tab_taxonomy:
    st.subheader("Failure Taxonomy Breakdown")

    if taxonomy_df.empty:
        st.info("No taxonomy data available. Run Step 13 first.")
    else:
        # Group by target model
        for target in taxonomy_df["target_model"].unique():
            st.markdown(f"#### Target: `{target}`")
            target_data = taxonomy_df[taxonomy_df["target_model"] == target]

            # Pie chart
            chart_df = target_data.set_index("category")["count"]
            col1, col2 = st.columns(2)

            with col1:
                st.bar_chart(chart_df)

            with col2:
                total = target_data["count"].sum()
                for _, row in target_data.iterrows():
                    pct = (row["count"] / total * 100) if total > 0 else 0
                    st.write(f"**{row['category']}**: {row['count']} ({pct:.1f}%)")

            st.divider()

# ── Tab 3: Boundary Distance ─────────────────────────────────

with tab_boundary:
    st.subheader("Boundary Distance Maps")

    if boundary_chains.empty:
        st.info("No boundary distance data available. Run Steps 14 or 16 first.")
    else:
        for target in boundary_chains["target_model"].unique():
            st.markdown(f"#### Target: `{target}`")
            target_data = boundary_chains[boundary_chains["target_model"] == target]

            # Summary metrics
            total = len(target_data)
            collapsed = target_data["full_collapse"].sum()
            avg_chain = target_data["chain_length"].mean()

            col1, col2, col3 = st.columns(3)
            col1.metric("Seeds Tested", total)
            col2.metric("Full Collapses", int(collapsed))
            col3.metric("Avg Chain Length", f"{avg_chain:.1f}")

            # Table
            display_cols = [
                "seed_id", "seed_category", "seed_severity",
                "initial_action", "final_action", "full_collapse",
                "chain_length", "total_reward",
            ]
            available = [c for c in display_cols if c in target_data.columns]
            st.dataframe(target_data[available], use_container_width=True, hide_index=True)

            st.divider()

# ── Tab 4: Multi-Turn Pressure ───────────────────────────────

with tab_multiturn:
    st.subheader("Multi-Turn Pressure (Safety Stress Depth)")

    if multiturn_df.empty:
        st.info("No multi-turn data available. Run Step 15 first.")
    else:
        for target in multiturn_df["target_model"].unique():
            st.markdown(f"#### Target: `{target}`")
            target_data = multiturn_df[multiturn_df["target_model"] == target]

            failed = target_data["failed"].sum()
            total = len(target_data)
            depths = target_data["safety_stress_depth"].dropna()

            col1, col2, col3 = st.columns(3)
            col1.metric("Seeds Tested", total)
            col2.metric("Failed Under Pressure", int(failed))
            col3.metric("Avg Stress Depth", f"{depths.mean():.1f}" if len(depths) > 0 else "N/A")

            # Table
            display_cols = [
                "seed_id", "seed_category", "initial_action", "final_action",
                "safety_stress_depth", "total_turns", "failed",
            ]
            available = [c for c in display_cols if c in target_data.columns]
            st.dataframe(target_data[available], use_container_width=True, hide_index=True)

            st.divider()

# ── Tab 5: Boundary Search Chains ────────────────────────────

with tab_chains:
    st.subheader("Decision Boundary Search — Transformation Chains")

    if boundary_chains.empty:
        st.info("No boundary search data available. Run Step 16 first.")
    else:
        # Filter by collapse status
        filter_option = st.radio(
            "Filter",
            options=["All", "Full Collapses Only", "Held (No Collapse)"],
            horizontal=True,
        )

        filtered = boundary_chains.copy()
        if filter_option == "Full Collapses Only":
            filtered = filtered[filtered["full_collapse"] == True]
        elif filter_option == "Held (No Collapse)":
            filtered = filtered[filtered["full_collapse"] == False]

        for _, row in filtered.iterrows():
            icon = "🚨" if row["full_collapse"] else "🛡️"
            with st.expander(
                f"{icon} {row['seed_id']} ({row['seed_category']}, sev={row['seed_severity']}) "
                f"| {row['initial_action']} → {row['final_action']} "
                f"| Chain: {row['chain_length']} steps"
            ):
                st.write(f"**Chain:** {row['chain_summary']}")
                st.write(f"**Total Reward:** {row['total_reward']}")

# ── Tab 6: Trace Explorer ────────────────────────────────────

with tab_explorer:
    st.subheader("Trace Explorer")

    # Collect all traces from all reports
    all_traces = []

    for category, items in reports.items():
        for item in items:
            data = item["data"]
            traces = data.get("traces", data.get("results", []))
            if isinstance(traces, list):
                for trace in traces:
                    if isinstance(trace, dict):
                        trace["_source_file"] = item["file"]
                        trace["_report_type"] = category
                        all_traces.append(trace)

    if not all_traces:
        st.info("No traces found.")
    else:
        st.write(f"Total traces available: **{len(all_traces)}**")

        # Filter by source
        source_filter = st.selectbox(
            "Filter by report type",
            options=["All"] + list(set(t["_report_type"] for t in all_traces)),
        )

        filtered_traces = all_traces
        if source_filter != "All":
            filtered_traces = [t for t in all_traces if t["_report_type"] == source_filter]

        # Convert to DataFrame for display
        trace_df = pd.json_normalize(filtered_traces[:100])  # Limit to 100 for performance

        if not trace_df.empty:
            st.dataframe(trace_df, use_container_width=True, hide_index=True)

            # Individual trace viewer
            st.divider()
            st.markdown("#### Individual Trace Viewer")

            trace_idx = st.number_input(
                "Trace index",
                min_value=0,
                max_value=len(filtered_traces) - 1,
                value=0,
            )

            if 0 <= trace_idx < len(filtered_traces):
                trace = filtered_traces[trace_idx]
                st.json(trace)