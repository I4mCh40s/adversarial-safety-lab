"""
Step 3: Target System Adapters

Changes from Step 2:
- Moderation systems are now pluggable adapters
- We evaluate multiple adapters on the same dataset
- We add baseline adapters: always_allow and always_block
- We add a regex-based moderator
- We save adapter metrics to results/
"""

import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, List, Dict


Action = Literal["allow", "block", "review"]


ERROR_TYPES = [
    "false_positive",
    "false_negative",
    "missed_review",
    "over_block",
    "unnecessary_review",
    "under_block",
]


@dataclass
class Seed:
    id: str
    text: str
    category: str
    subcategory: Optional[str]
    severity: int
    language: str
    modality: str
    expected_action: Action
    notes: Optional[str] = None


# ── Data loading ──────────────────────────────────────────────


def load_seeds(path: str) -> List[Seed]:
    """Load seeds from a JSONL file."""
    seeds = []
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                seeds.append(
                    Seed(
                        id=data["id"],
                        text=data["text"],
                        category=data["category"],
                        subcategory=data.get("subcategory"),
                        severity=data.get("severity", 0),
                        language=data.get("language", "en"),
                        modality=data.get("modality", "text"),
                        expected_action=data["expected_action"],
                        notes=data.get("notes"),
                    )
                )
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping line {line_number}: {e}")

    return seeds


# ── Target adapter interface ─────────────────────────────────


class ModerationAdapter(ABC):
    """
    Base class for all moderation targets.

    Later this can wrap:
    - rule engines
    - LLMs
    - VLMs
    - safety classifiers
    - CLIP-Guard
    - commercial moderation APIs
    """

    name: str = "base_adapter"

    @abstractmethod
    def moderate(self, text: str) -> Action:
        raise NotImplementedError


# ── Baseline adapters ────────────────────────────────────────


class AlwaysAllowAdapter(ModerationAdapter):
    """
    Baseline that allows everything.

    This is useful to measure false negatives and missed reviews.
    """

    name = "always_allow"

    def moderate(self, text: str) -> Action:
        return "allow"


class AlwaysBlockAdapter(ModerationAdapter):
    """
    Baseline that blocks everything.

    This is useful to measure false positives and over-blocking.
    """

    name = "always_block"

    def moderate(self, text: str) -> Action:
        return "block"


# ── Step 2 keyword moderator ─────────────────────────────────


class KeywordModerator(ModerationAdapter):
    """
    The simple keyword moderator from Step 2.
    """

    name = "keyword_moderator"

    HARM_KEYWORDS = [
        "harass",
        "scam",
        "threaten",
        "phishing",
        "fake news",
        "hurt",
        "attack",
    ]

    REVIEW_KEYWORDS = [
        "insult",
        "extremist",
        "recruit",
        "social engineering",
        "threatening",
        "manipulat",
    ]

    def moderate(self, text: str) -> Action:
        text_lower = text.lower()

        for keyword in self.HARM_KEYWORDS:
            if keyword in text_lower:
                return "block"

        for keyword in self.REVIEW_KEYWORDS:
            if keyword in text_lower:
                return "review"

        return "allow"


# ── Regex moderator ──────────────────────────────────────────


class RegexModerator(ModerationAdapter):
    """
    A slightly more precise rule-based moderator.

    It tries to reduce false positives by looking for harmful intent patterns,
    not just isolated keywords.

    It also routes ambiguous educational/fictional/security-research contexts
    to review.
    """

    name = "regex_moderator"

    REVIEW_PATTERNS = [
        re.compile(r"\b(novel|fiction|screenplay|story)\b", re.IGNORECASE),
        re.compile(r"\b(research|study|paper|linguistics)\b", re.IGNORECASE),
        re.compile(r"\b(train|training|employees|recognize)\b", re.IGNORECASE),
        re.compile(r"\b(counter[- ]terrorism|ngo)\b", re.IGNORECASE),
        re.compile(r"\b(insult|social engineering|extremis\w*|recruit\w*)\b", re.IGNORECASE),
    ]

    HARM_PATTERNS = [
        re.compile(
            r"\b(write|generate|create|help me create|draft)\b.*"
            r"\b(harass\w*|scam\w*|phishing|fake news|threaten\w*|threatening)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(scam|phishing)\b.*\b(trick|money|bank|credentials)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(fake news)\b.*\b(commit\w*|crime|accus\w*)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(threaten\w*|threatening)\b.*\b(violence|kill|harm|hurt)\b",
            re.IGNORECASE,
        ),
    ]

    def moderate(self, text: str) -> Action:
        # Review patterns are checked first because ambiguous contexts
        # should often go to human review rather than auto-block.
        for pattern in self.REVIEW_PATTERNS:
            if pattern.search(text):
                return "review"

        for pattern in self.HARM_PATTERNS:
            if pattern.search(text):
                return "block"

        return "allow"


# ── Judge ─────────────────────────────────────────────────────


def judge(seed: Seed, action: Action) -> Dict:
    """
    Compare the adapter action against the expected action.
    """
    correct = seed.expected_action == action

    error_type: Optional[str] = None

    if not correct:
        if seed.expected_action == "allow":
            if action == "block":
                error_type = "false_positive"
            elif action == "review":
                error_type = "unnecessary_review"

        elif seed.expected_action == "block":
            if action == "allow":
                error_type = "false_negative"
            elif action == "review":
                error_type = "under_block"

        elif seed.expected_action == "review":
            if action == "allow":
                error_type = "missed_review"
            elif action == "block":
                error_type = "over_block"

    return {
        "case_id": seed.id,
        "category": seed.category,
        "severity": seed.severity,
        "expected_action": seed.expected_action,
        "moderator_action": action,
        "correct": correct,
        "error_type": error_type,
    }


# ── Metrics ───────────────────────────────────────────────────


def percentage(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total * 100, 1)


def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculate evaluation metrics for one adapter.
    """
    total = len(results)
    correct = sum(1 for r in results if r["correct"])

    error_counts = {error_type: 0 for error_type in ERROR_TYPES}

    for result in results:
        if result["error_type"]:
            error_counts[result["error_type"]] += 1

    action_counts = Counter(result["moderator_action"] for result in results)

    # Per-category accuracy
    categories: Dict[str, Dict] = {}

    for result in results:
        category = result["category"]

        if category not in categories:
            categories[category] = {
                "total": 0,
                "correct": 0,
            }

        categories[category]["total"] += 1

        if result["correct"]:
            categories[category]["correct"] += 1

    for category, data in categories.items():
        total_category = data["total"]
        correct_category = data["correct"]
        data["accuracy"] = percentage(correct_category, total_category)

    return {
        "total_cases": total,
        "correct": correct,
        "accuracy": percentage(correct, total),
        "errors": error_counts,
        "action_distribution": {
            "allow": action_counts.get("allow", 0),
            "review": action_counts.get("review", 0),
            "block": action_counts.get("block", 0),
        },
        "rates": {
            "auto_allow_rate": percentage(action_counts.get("allow", 0), total),
            "human_review_rate": percentage(action_counts.get("review", 0), total),
            "auto_block_rate": percentage(action_counts.get("block", 0), total),
        },
        "by_category": categories,
    }


# ── Evaluation runner ────────────────────────────────────────


def evaluate_adapter(adapter: ModerationAdapter, seeds: List[Seed]) -> Dict:
    """
    Evaluate one adapter on the full seed set.
    """
    results = []

    for seed in seeds:
        action = adapter.moderate(seed.text)
        verdict = judge(seed, action)
        results.append(verdict)

    metrics = calculate_metrics(results)
    metrics["adapter"] = adapter.name

    return {
        "metrics": metrics,
        "results": results,
    }


def print_failures(adapter_name: str, results: List[Dict]) -> None:
    """
    Print only the cases where the adapter failed.
    """
    failures = [result for result in results if not result["correct"]]

    print(f"\nFailures for {adapter_name}: {len(failures)}")

    if not failures:
        print("  None")
        return

    for result in failures:
        print(
            f"  - {result['case_id']:<15s} "
            f"expected={result['expected_action']:<6s} "
            f"got={result['moderator_action']:<6s} "
            f"error={result['error_type']}"
        )


def print_leaderboard(all_metrics: List[Dict]) -> None:
    """
    Print a compact leaderboard comparing adapters.
    """
    print()
    print("=" * 110)
    print("Adapter Leaderboard")
    print("=" * 110)

    header = (
        f"{'Adapter':<22s} "
        f"{'Acc%':>6s} "
        f"{'FP':>3s} "
        f"{'FN':>3s} "
        f"{'MissRev':>7s} "
        f"{'OverBlk':>7s} "
        f"{'UnnRev':>6s} "
        f"{'UnderBlk':>8s} "
        f"{'Allow%':>6s} "
        f"{'Review%':>7s} "
        f"{'Block%':>6s}"
    )

    print(header)
    print("-" * 110)

    for metrics in all_metrics:
        errors = metrics["errors"]
        rates = metrics["rates"]

        row = (
            f"{metrics['adapter']:<22s} "
            f"{metrics['accuracy']:6.1f} "
            f"{errors['false_positive']:3d} "
            f"{errors['false_negative']:3d} "
            f"{errors['missed_review']:7d} "
            f"{errors['over_block']:7d} "
            f"{errors['unnecessary_review']:6d} "
            f"{errors['under_block']:8d} "
            f"{rates['auto_allow_rate']:6.1f} "
            f"{rates['human_review_rate']:7.1f} "
            f"{rates['auto_block_rate']:6.1f}"
        )

        print(row)


# ── Main ──────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("Step 3: Target System Adapters")
    print("=" * 70)

    seed_path = Path(__file__).parent.parent / "data" / "seeds.jsonl"
    seeds = load_seeds(str(seed_path))

    print(f"\nLoaded {len(seeds)} seeds from {seed_path}")

    adapters: List[ModerationAdapter] = [
        AlwaysAllowAdapter(),
        AlwaysBlockAdapter(),
        KeywordModerator(),
        RegexModerator(),
    ]

    all_metrics = []
    results_by_adapter = {}

    for adapter in adapters:
        evaluation = evaluate_adapter(adapter, seeds)

        all_metrics.append(evaluation["metrics"])
        results_by_adapter[adapter.name] = evaluation["results"]

    # Show failures for each adapter
    for adapter in adapters:
        print_failures(adapter.name, results_by_adapter[adapter.name])

    # Show leaderboard
    print_leaderboard(all_metrics)

    # Save results
    output_path = Path(__file__).parent.parent / "results" / "step03_adapter_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "step": "step03_target_adapters",
        "seed_file": str(seed_path),
        "seed_count": len(seeds),
        "adapters": all_metrics,
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(f"Saved metrics to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()