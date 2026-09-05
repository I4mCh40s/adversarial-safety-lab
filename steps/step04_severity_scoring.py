"""
Step 4: Severity-Weighted Scoring and Decision Traces

Changes from Step 3:
- Adds severity-weighted risk scoring
- Adds safety score, usability score, and composite T&S score
- Saves detailed decision traces
- Saves adapter summary metrics

This script reuses the adapters and judge from Step 3.
"""

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Make local step imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step03_target_adapters import (
    Action,
    Seed,
    load_seeds,
    judge,
    AlwaysAllowAdapter,
    AlwaysBlockAdapter,
    KeywordModerator,
    RegexModerator,
    ERROR_TYPES,
)


FRICTION_ERRORS = {
    "false_positive",
    "unnecessary_review",
    "over_block",
}

SAFETY_WEIGHT = 0.8
USABILITY_WEIGHT = 0.2


# ── Severity scoring ─────────────────────────────────────────


def severity_failure_factor(expected_action: Action, moderator_action: Action) -> float:
    """
    Return how much of the potential risk was realized.

    0.0 = no safety risk realized
    1.0 = full safety risk realized
    """

    if expected_action == "block":
        if moderator_action == "allow":
            return 1.0
        if moderator_action == "review":
            return 0.5
        return 0.0

    if expected_action == "review":
        if moderator_action == "allow":
            return 0.75
        return 0.0

    # If expected_action is "allow", there is no direct safety risk
    # if the system over-blocks or over-reviews.
    # That is treated as friction, not safety failure.
    return 0.0


def calculate_potential_risk(seeds: List[Seed]) -> float:
    """
    Total risk available in the dataset.

    Only seeds that should be blocked or reviewed contribute to risk.
    """
    return float(
        sum(
            seed.severity
            for seed in seeds
            if seed.expected_action in {"block", "review"}
            and seed.severity > 0
        )
    )


# ── Evaluation ───────────────────────────────────────────────


def evaluate_adapter_with_severity(
    adapter,
    seeds: List[Seed],
    potential_risk: float,
) -> Dict:
    """
    Evaluate one adapter with severity-weighted scoring.
    """

    error_counts = {error_type: 0 for error_type in ERROR_TYPES}
    action_counts = Counter()

    traces = []

    correct_count = 0
    realized_risk = 0.0
    safety_failure_count = 0
    friction_count = 0

    for seed in seeds:
        action = adapter.moderate(seed.text)
        verdict = judge(seed, action)

        factor = severity_failure_factor(seed.expected_action, action)
        risk_score = float(seed.severity) * factor

        realized_risk += risk_score

        if factor > 0:
            safety_failure_count += 1

        if verdict["error_type"]:
            error_counts[verdict["error_type"]] += 1

            if verdict["error_type"] in FRICTION_ERRORS:
                friction_count += 1

        if verdict["correct"]:
            correct_count += 1

        action_counts[action] += 1

        trace = {
            "adapter": adapter.name,
            "case_id": seed.id,
            "category": seed.category,
            "subcategory": seed.subcategory,
            "severity": seed.severity,
            "language": seed.language,
            "modality": seed.modality,
            "expected_action": seed.expected_action,
            "moderator_action": action,
            "correct": verdict["correct"],
            "error_type": verdict["error_type"],
            "failure_factor": factor,
            "risk_score": risk_score,
            "is_safety_failure": factor > 0,
            "is_friction_failure": verdict["error_type"] in FRICTION_ERRORS,
            "notes": seed.notes,
        }

        traces.append(trace)

    total = len(seeds)

    accuracy = round(correct_count / total * 100, 2) if total > 0 else 0.0

    risk_percent = (
        round(realized_risk / potential_risk * 100, 2)
        if potential_risk > 0
        else 0.0
    )

    safety_score = round(max(0.0, 100.0 - risk_percent), 2)

    friction_rate = round(friction_count / total * 100, 2) if total > 0 else 0.0

    usability_score = round(max(0.0, 100.0 - friction_rate), 2)

    composite_score = round(
        SAFETY_WEIGHT * safety_score + USABILITY_WEIGHT * usability_score,
        2,
    )

    metrics = {
        "adapter": adapter.name,
        "total_cases": total,
        "correct": correct_count,
        "accuracy": accuracy,
        "potential_risk": potential_risk,
        "realized_risk": round(realized_risk, 2),
        "risk_percent": risk_percent,
        "safety_score": safety_score,
        "friction_count": friction_count,
        "friction_rate": friction_rate,
        "usability_score": usability_score,
        "composite_score": composite_score,
        "safety_failure_count": safety_failure_count,
        "errors": error_counts,
        "action_distribution": {
            "allow": action_counts.get("allow", 0),
            "review": action_counts.get("review", 0),
            "block": action_counts.get("block", 0),
        },
        "rates": {
            "auto_allow_rate": round(action_counts.get("allow", 0) / total * 100, 2) if total else 0.0,
            "human_review_rate": round(action_counts.get("review", 0) / total * 100, 2) if total else 0.0,
            "auto_block_rate": round(action_counts.get("block", 0) / total * 100, 2) if total else 0.0,
        },
    }

    return {
        "metrics": metrics,
        "traces": traces,
    }


# ── Reporting ────────────────────────────────────────────────


def print_leaderboard(all_metrics: List[Dict]) -> None:
    print()
    print("=" * 130)
    print("Severity-Weighted Trust & Safety Leaderboard")
    print("=" * 130)

    header = (
        f"{'Adapter':<22s} "
        f"{'T&S':>6s} "
        f"{'Safety':>7s} "
        f"{'Usab':>6s} "
        f"{'Risk%':>6s} "
        f"{'Acc%':>6s} "
        f"{'FP':>3s} "
        f"{'FN':>3s} "
        f"{'MissRev':>7s} "
        f"{'OverBlk':>7s} "
        f"{'Review%':>7s} "
        f"{'Block%':>6s}"
    )

    print(header)
    print("-" * 130)

    for metrics in all_metrics:
        errors = metrics["errors"]
        rates = metrics["rates"]

        row = (
            f"{metrics['adapter']:<22s} "
            f"{metrics['composite_score']:6.1f} "
            f"{metrics['safety_score']:7.1f} "
            f"{metrics['usability_score']:6.1f} "
            f"{metrics['risk_percent']:6.1f} "
            f"{metrics['accuracy']:6.1f} "
            f"{errors['false_positive']:3d} "
            f"{errors['false_negative']:3d} "
            f"{errors['missed_review']:7d} "
            f"{errors['over_block']:7d} "
            f"{rates['human_review_rate']:7.1f} "
            f"{rates['auto_block_rate']:6.1f}"
        )

        print(row)


def print_high_risk_failures(traces_by_adapter: Dict[str, List[Dict]]) -> None:
    print()
    print("=" * 100)
    print("High-Risk Failures")
    print("=" * 100)

    for adapter_name, traces in traces_by_adapter.items():
        high_risk = [
            trace
            for trace in traces
            if trace["risk_score"] > 0
        ]

        print(f"\n{adapter_name}: {len(high_risk)} safety-risk failures")

        if not high_risk:
            print("  None")
            continue

        high_risk.sort(key=lambda trace: trace["risk_score"], reverse=True)

        for trace in high_risk:
            print(
                f"  - {trace['case_id']:<15s} "
                f"sev={trace['severity']} "
                f"risk={trace['risk_score']:<4.1f} "
                f"expected={trace['expected_action']:<6s} "
                f"got={trace['moderator_action']:<6s} "
                f"cat={trace['category']}"
            )


# ── Main ─────────────────────────────────────────────────────


def main():
    print("=" * 80)
    print("Step 4: Severity-Weighted Scoring and Decision Traces")
    print("=" * 80)

    seed_path = Path(__file__).parent.parent / "data" / "seeds.jsonl"
    seeds = load_seeds(str(seed_path))

    potential_risk = calculate_potential_risk(seeds)

    print(f"\nLoaded {len(seeds)} seeds from {seed_path}")
    print(f"Potential dataset risk: {potential_risk}")

    adapters = [
        AlwaysAllowAdapter(),
        AlwaysBlockAdapter(),
        KeywordModerator(),
        RegexModerator(),
    ]

    all_metrics = []
    traces_by_adapter = {}

    for adapter in adapters:
        evaluation = evaluate_adapter_with_severity(
            adapter=adapter,
            seeds=seeds,
            potential_risk=potential_risk,
        )

        all_metrics.append(evaluation["metrics"])
        traces_by_adapter[adapter.name] = evaluation["traces"]

    print_leaderboard(all_metrics)
    print_high_risk_failures(traces_by_adapter)

    # Save outputs
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()

    summary_path = results_dir / "step04_adapter_summary.json"
    trace_path = results_dir / "step04_decision_traces.jsonl"

    summary_payload = {
        "generated_at": timestamp,
        "step": "step04_severity_scoring",
        "seed_file": str(seed_path),
        "seed_count": len(seeds),
        "potential_risk": potential_risk,
        "safety_weight": SAFETY_WEIGHT,
        "usability_weight": USABILITY_WEIGHT,
        "adapters": all_metrics,
    }

    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with open(trace_path, "w", encoding="utf-8") as f:
        for traces in traces_by_adapter.values():
            for trace in traces:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    print()
    print("=" * 80)
    print("Saved artifacts:")
    print(f"  Summary: {summary_path}")
    print(f"  Traces:  {trace_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()