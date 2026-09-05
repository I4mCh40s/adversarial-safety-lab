"""
Step 2: Structured Dataset

Changes from Step 1:
- Seeds are loaded from data/seeds.jsonl instead of being hardcoded
- Each seed has severity, language, modality, and notes
- The moderator now returns three actions: allow, block, review
- Metrics include breakdowns by category
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, List, Dict


Action = Literal["allow", "block", "review"]


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
                seeds.append(Seed(
                    id=data["id"],
                    text=data["text"],
                    category=data["category"],
                    subcategory=data.get("subcategory"),
                    severity=data.get("severity", 0),
                    language=data.get("language", "en"),
                    modality=data.get("modality", "text"),
                    expected_action=data["expected_action"],
                    notes=data.get("notes"),
                ))
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Skipping line {line_number}: {e}")

    return seeds


# ── Target moderator ──────────────────────────────────────────


HARM_KEYWORDS = [
    "harass", "scam", "threaten", "phishing",
    "fake news", "hurt", "attack",
]

REVIEW_KEYWORDS = [
    "insult", "extremist", "recruit", "social engineering",
    "threatening", "manipulat",
]


def target_moderator(text: str) -> Action:
    """
    Improved toy moderator with three-tier decisions.

    Block  → high-confidence harmful
    Review → ambiguous or sensitive
    Allow  → appears safe
    """
    text_lower = text.lower()

    for keyword in HARM_KEYWORDS:
        if keyword in text_lower:
            return "block"

    for keyword in REVIEW_KEYWORDS:
        if keyword in text_lower:
            return "review"

    return "allow"


# ── Judge ─────────────────────────────────────────────────────


def judge(seed: Seed, action: Action) -> Dict:
    """
    Judge the moderator's action against the expected action.
    """
    correct = seed.expected_action == action

    error_type: Optional[str] = None

    if not correct:
        if action == "block" and seed.expected_action == "allow":
            error_type = "false_positive"
        elif action == "allow" and seed.expected_action == "block":
            error_type = "false_negative"
        elif action == "allow" and seed.expected_action == "review":
            error_type = "missed_review"
        elif action == "block" and seed.expected_action == "review":
            error_type = "over_block"
        elif action == "review" and seed.expected_action == "allow":
            error_type = "unnecessary_review"
        elif action == "review" and seed.expected_action == "block":
            error_type = "under_block"

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


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate overall and per-category metrics."""

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0.0

    # Count error types
    error_counts: Dict[str, int] = {}
    for r in results:
        if r["error_type"]:
            error_counts[r["error_type"]] = (
                error_counts.get(r["error_type"], 0) + 1
            )

    # Per-category breakdown
    categories: Dict[str, Dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r["correct"]:
            categories[cat]["correct"] += 1

    for cat in categories:
        t = categories[cat]["total"]
        c = categories[cat]["correct"]
        categories[cat]["accuracy"] = round(c / t * 100, 2) if t > 0 else 0.0

    return {
        "total_cases": total,
        "correct": correct,
        "accuracy": round(accuracy * 100, 2),
        "errors": error_counts,
        "by_category": categories,
    }


# ── Main ──────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Step 2: Structured Dataset Evaluation")
    print("=" * 60)

    # Load seeds from file
    seed_path = Path(__file__).parent.parent / "data" / "seeds.jsonl"
    seeds = load_seeds(str(seed_path))

    print(f"\nLoaded {len(seeds)} seeds from {seed_path}\n")

    results = []

    for seed in seeds:
        action = target_moderator(seed.text)
        verdict = judge(seed, action)
        results.append(verdict)

        status = "PASS" if verdict["correct"] else "FAIL"

        print(f"[{status:4s}]  {seed.id:<15s}  "
              f"expected={seed.expected_action:<6s}  "
              f"got={action:<6s}  "
              f"cat={seed.category}")

        if verdict["error_type"]:
            print(f"        └── error: {verdict['error_type']}")

    # Calculate and display metrics
    metrics = calculate_metrics(results)

    print()
    print("=" * 60)
    print("Overall Metrics")
    print("=" * 60)
    print(f"Total cases:  {metrics['total_cases']}")
    print(f"Correct:      {metrics['correct']}")
    print(f"Accuracy:     {metrics['accuracy']}%")

    if metrics["errors"]:
        print()
        print("Error breakdown:")
        for error_type, count in metrics["errors"].items():
            print(f"  {error_type}: {count}")

    print()
    print("=" * 60)
    print("Per-Category Breakdown")
    print("=" * 60)

    for cat, data in metrics["by_category"].items():
        print(f"  {cat:<15s}  "
              f"{data['correct']}/{data['total']}  "
              f"({data['accuracy']}%)")


if __name__ == "__main__":
    main()