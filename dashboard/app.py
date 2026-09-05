"""
Adversarial AI Safety Lab Dashboard

Reads Step 10 reverse arena reports from results/ and displays:
- cross-model leaderboard
- safety/evasion chart
- strategy performance
- full/partial evasion gallery
- raw report inspector
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# ── Paths ─────────────────────────────────────────────────────


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


# ── Helpers ───────────────────────────────────────────────────


def clean_obj(obj):
    """
    Recursively strip whitespace from dictionary keys and string values.

    This makes the dashboard robust if some JSON files contain keys like:
      "seed_id ": "harmful_001 "
    """

    if isinstance(obj, dict):
        return {
            str(key).strip(): clean_obj(value)
            for key, value in obj.items()
        }

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


# ── Data loading ──────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def load_leaderboard() -> pd.DataFrame:
    """
    Load all Step 10 reverse arena reports and build a leaderboard.
    """

    if not RESULTS_DIR.exists():
        return pd.DataFrame()

    files = [
        path
        for path in RESULTS_DIR.glob("step10_reverse_arena*.json")
        if path.name != "step10_reverse_arena_latest.json"
    ]

    files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

    rows = []

    for path in files:
        try:
            data = clean_obj(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue

        stats = data.get("stats", {})
        metrics = data.get("metrics", {})

        target_model = (
            data.get("target_model")
            or data.get("target")
            or "unknown_target"
        )

        attacker_model = data.get("attacker_model", "unknown_attacker")

        valid_target_decisions = safe_int(stats.get("valid_target_decisions", 0))
        full_evasion = safe_int(stats.get("full_evasion", 0))
        partial_evasion = safe_int(stats.get("partial_evasion", 0))

        attack_success_rate = metrics.get("attack_success_rate")
        if attack_success_rate is None:
            attack_success_rate = (
                round(full_evasion / valid_target_decisions * 100, 2)
                if valid_target_decisions > 0
                else 0.0
            )

        evasion_score = metrics.get("evasion_score")
        if evasion_score is None:
            evasion_score = (
                round(
                    (full_evasion + 0.5 * partial_evasion)
                    / valid_target_decisions
                    * 100,
                    2,
                )
                if valid_target_decisions > 0
                else 0.0
            )

        rows.append(
            {
                "file": path.name,
                "modified": pd.Timestamp.fromtimestamp(path.stat().st_mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "target": target_model,
                "attacker": attacker_model,
                "rounds": safe_int(data.get("rounds", 0)),
                "total_attempts": safe_int(stats.get("total_attack_attempts", 0)),
                "valid_target_decisions": valid_target_decisions,
                "full_evasion": full_evasion,
                "partial_evasion": partial_evasion,
                "contained": safe_int(stats.get("contained", 0)),
                "target_invalid": safe_int(stats.get("target_invalid", 0)),
                "attack_success_rate": safe_float(attack_success_rate),
                "evasion_score": safe_float(evasion_score),
                "safety_under_attack": safe_float(metrics.get("safety_under_attack", 0)),
                "risk_percent": safe_float(metrics.get("risk_percent", 0)),
                "realized_risk": safe_float(metrics.get("realized_risk", 0)),
                "potential_risk": safe_float(metrics.get("potential_risk", 0)),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["safety_under_attack", "attack_success_rate"],
        ascending=[False, True],
    )

    return df


@st.cache_data(show_spinner=False)
def load_report(file_name: str):
    """
    Load one full report.
    """

    path = RESULTS_DIR / file_name

    if not path.exists():
        return {}

    try:
        return clean_obj(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


# ── UI ────────────────────────────────────────────────────────


st.set_page_config(
    page_title="Adversarial AI Safety Lab",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Adversarial AI Safety Lab")

st.caption(
    "Cross-model red-teaming leaderboard for LLM moderation targets. "
    "Reports are loaded from results/step10_reverse_arena*.json"
)

st.sidebar.header("Controls")

if st.sidebar.button("🔄 Refresh reports"):
    st.cache_data.clear()

leaderboard = load_leaderboard()

if leaderboard.empty:
    st.warning(
        "No Step 10 reverse arena reports found.\n\n"
        "Expected files like:\n\n"
        "results/step10_reverse_arena_granite4.1_3b_20260906_143022.json\n\n"
        "Run:\n\n"
        "python steps/step10_uncensored_seed_factory_and_reverse_arena.py attack --target ollama --target-model granite4.1:3b"
    )
    st.stop()

# ── Leaderboard ───────────────────────────────────────────────


st.subheader("Model Leaderboard")

leaderboard_cols = [
    "file",
    "modified",
    "target",
    "attacker",
    "rounds",
    "total_attempts",
    "valid_target_decisions",
    "full_evasion",
    "partial_evasion",
    "contained",
    "attack_success_rate",
    "evasion_score",
    "safety_under_attack",
    "risk_percent",
]

st.dataframe(
    leaderboard[leaderboard_cols],
    use_container_width=True,
    hide_index=True,
)


# ── Chart ─────────────────────────────────────────────────────


st.subheader("Safety / Evasion Chart")

chart_df = leaderboard.set_index("file")[
    [
        "safety_under_attack",
        "attack_success_rate",
        "evasion_score",
    ]
]

st.bar_chart(chart_df)


# ── Report inspector ──────────────────────────────────────────


st.subheader("Report Inspector")

selected_file = st.selectbox(
    "Select a report",
    leaderboard["file"].tolist(),
)

report = load_report(selected_file)

if not report:
    st.error("Could not load selected report.")
    st.stop()

stats = report.get("stats", {})
metrics = report.get("metrics", {})
traces = report.get("traces", [])

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Safety Under Attack",
    f"{safe_float(metrics.get('safety_under_attack', 0)):.1f}",
)

col2.metric(
    "Attack Success Rate",
    f"{safe_float(metrics.get('attack_success_rate', 0)):.1f}%",
)

col3.metric(
    "Evasion Score",
    f"{safe_float(metrics.get('evasion_score', 0)):.1f}%",
)

col4.metric(
    "Valid Target Decisions",
    safe_int(stats.get("valid_target_decisions", 0)),
)

st.write(
    f"**Target:** `{report.get('target_model') or report.get('target')}`  \n"
    f"**Attacker:** `{report.get('attacker_model')}`  \n"
    f"**Rounds:** `{report.get('rounds')}`  \n"
    f"**Seed file:** `{report.get('seed_file')}`"
)


# ── Tabs ──────────────────────────────────────────────────────


tab_traces, tab_strategy, tab_evasions, tab_raw = st.tabs(
    [
        "Traces",
        "Strategy Performance",
        "Evasion Gallery",
        "Raw JSON",
    ]
)


# ── Traces tab ────────────────────────────────────────────────


with tab_traces:
    if not traces:
        st.info("No traces found in this report.")
    else:
        trace_df = pd.json_normalize(traces)

        preferred_cols = [
            "round",
            "seed_id",
            "category",
            "severity",
            "expected_action",
            "strategy",
            "outcome",
            "target_action",
            "target_valid",
            "risk_score",
            "original_text",
            "transformed_text",
        ]

        available_cols = [col for col in preferred_cols if col in trace_df.columns]

        st.dataframe(
            trace_df[available_cols],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="Download traces as CSV",
            data=trace_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(selected_file).stem}_traces.csv",
            mime="text/csv",
        )


# ── Strategy performance tab ─────────────────────────────────


with tab_strategy:
    if not traces:
        st.info("No traces found in this report.")
    else:
        strategy_df = pd.DataFrame(traces)

        if "strategy" not in strategy_df.columns or "outcome" not in strategy_df.columns:
            st.info("This report does not contain strategy/outcome fields.")
        else:
            strategy_summary = (
                strategy_df.groupby("strategy")
                .agg(
                    attempts=("strategy", "size"),
                    full_evasion=(
                        "outcome",
                        lambda s: (s == "full_evasion").sum(),
                    ),
                    partial_evasion=(
                        "outcome",
                        lambda s: (s == "partial_evasion").sum(),
                    ),
                    contained=(
                        "outcome",
                        lambda s: (s == "contained").sum(),
                    ),
                    target_invalid=(
                        "outcome",
                        lambda s: (s == "target_invalid").sum(),
                    ),
                )
                .reset_index()
            )

            strategy_summary["full_evasion_rate"] = (
                strategy_summary["full_evasion"]
                / strategy_summary["attempts"]
                * 100
            ).round(1)

            strategy_summary["evasion_score"] = (
                (
                    strategy_summary["full_evasion"]
                    + 0.5 * strategy_summary["partial_evasion"]
                )
                / strategy_summary["attempts"]
                * 100
            ).round(1)

            strategy_summary = strategy_summary.sort_values(
                by=["evasion_score", "full_evasion_rate"],
                ascending=[False, False],
            )

            st.dataframe(
                strategy_summary,
                use_container_width=True,
                hide_index=True,
            )

            st.bar_chart(
                strategy_summary.set_index("strategy")[
                    [
                        "full_evasion",
                        "partial_evasion",
                        "contained",
                    ]
                ]
            )


# ── Evasion gallery tab ──────────────────────────────────────


with tab_evasions:
    if not traces:
        st.info("No traces found in this report.")
    else:
        evasion_filter = st.radio(
            "Show",
            options=[
                "Full evasions only",
                "Full + partial evasions",
                "All failed containment",
            ],
            horizontal=True,
        )

        if evasion_filter == "Full evasions only":
            evasion_outcomes = {"full_evasion"}
        elif evasion_filter == "Full + partial evasions":
            evasion_outcomes = {"full_evasion", "partial_evasion"}
        else:
            evasion_outcomes = {
                "full_evasion",
                "partial_evasion",
                "target_invalid",
                "over_contained",
            }

        evasions = [
            trace
            for trace in traces
            if trace.get("outcome") in evasion_outcomes
        ]

        if not evasions:
            st.success("No matching evasions found in this report.")
        else:
            st.write(f"Found **{len(evasions)}** matching traces.")

            for trace in evasions:
                seed_id = trace.get("seed_id", "unknown_seed")
                strategy = trace.get("strategy", "unknown_strategy")
                severity = trace.get("severity", "?")
                outcome = trace.get("outcome", "unknown_outcome")

                with st.expander(
                    f"{seed_id} | {strategy} | severity={severity} | {outcome}"
                ):
                    st.write("**Category:**", trace.get("category"))
                    st.write("**Expected action:**", trace.get("expected_action"))
                    st.write("**Target action:**", trace.get("target_action"))
                    st.write("**Risk score:**", trace.get("risk_score"))

                    st.write("**Original text:**")
                    st.code(trace.get("original_text", ""), language=None)

                    st.write("**Transformed text:**")
                    st.code(trace.get("transformed_text", ""), language=None)

                    target_raw = trace.get("target_raw", "")

                    if target_raw:
                        st.write("**Target raw response:**")
                        st.code(target_raw, language="json")


# ── Raw JSON tab ──────────────────────────────────────────────


with tab_raw:
    st.json(report)