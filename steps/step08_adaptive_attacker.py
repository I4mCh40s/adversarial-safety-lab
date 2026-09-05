"""
Step 8: The Adaptive Red-Team Loop

Changes from Step 7:
- The attacker now tracks strategy success rates.
- Uses an Epsilon-Greedy algorithm to adaptively choose the best attack.
- If the attacker refuses to generate the payload, it is penalized (Attacker Jailbreak concept).
- Visualizes the attacker "learning" which strategies bypass the regex_moderator.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from step03_target_adapters import RegexModerator, Seed, load_seeds
from step05_attack_transformations import attack_outcome
from step07_llm_attacker import ATTACK_STRATEGIES, LLMAttacker, REFUSAL_PHRASES


# ── Adaptive Strategy Selector ───────────────────────────────

class AdaptiveAttacker:
    def __init__(self, llm_attacker: LLMAttacker, epsilon: float = 0.2):
        """
        epsilon: Probability of exploring a random strategy vs exploiting the best one.
        """
        self.llm_attacker = llm_attacker
        self.epsilon = epsilon
        
        # Initialize scores for all strategies (start at 1.0 to avoid division by zero)
        self.scores: Dict[str, float] = {name: 1.0 for name in ATTACK_STRATEGIES.keys()}
        self.attempts: Dict[str, int] = {name: 0 for name in ATTACK_STRATEGIES.keys()}

    def select_strategy(self) -> str:
        """Epsilon-Greedy selection."""
        if random.random() < self.epsilon:
            # Explore: pick a random strategy
            return random.choice(list(ATTACK_STRATEGIES.keys()))
        else:
            # Exploit: pick the strategy with the highest average score
            best_strategy = max(self.scores, key=lambda k: self.scores[k] / max(1, self.attempts[k]))
            return best_strategy

    def update_score(self, strategy: str, outcome: str, attacker_refused: bool):
        """Update the strategy score based on the moderator's reaction."""
        self.attempts[strategy] += 1
        
        # Crucial T&S concept: If the attacker model refused to generate the attack, 
        # the attack FAILED. It shouldn't be rewarded just because the refusal text bypassed the filter.
        if attacker_refused:
            reward = -1.0 
        elif outcome == "full_evasion":
            reward = 3.0
        elif outcome == "partial_evasion":
            reward = 1.0
        elif outcome == "over_contained":
            reward = 0.5 # Still caused friction
        else: # contained
            reward = -2.0
            
        self.scores[strategy] += reward


# ── Evaluation Loop ──────────────────────────────────────────

def run_adaptive_campaign(
    adapter: RegexModerator, 
    attacker: AdaptiveAttacker, 
    seeds: List[Seed],
    rounds_per_seed: int = 2
):
    risk_seeds = [s for s in seeds if s.expected_action in {"block", "review"} and s.severity > 0]
    
    print(f"\n🚀 Starting Adaptive Red-Team Campaign against {adapter.name}")
    print(f"Target: {len(risk_seeds)} risky seeds | Rounds per seed: {rounds_per_seed}\n")
    
    total_attacks = 0
    successful_evasions = 0
    
    for round_num in range(1, rounds_per_seed + 1):
        print(f"--- ROUND {round_num} ---")
        
        # Shuffle seeds so the attacker doesn't just memorize the order
        random.shuffle(risk_seeds)
        
        for seed in risk_seeds:
            total_attacks += 1
            
            # 1. Attacker chooses strategy adaptively
            strategy_name = attacker.select_strategy()
            prompt_template = ATTACK_STRATEGIES[strategy_name]
            
            # 2. Generate attack
            transformed_text = attacker.llm_attacker.generate_attack(seed.text, prompt_template)
            
            # 3. Check for refusal
            refused = any(phrase in transformed_text.lower() for phrase in REFUSAL_PHRASES)
            
            # 4. Pass to moderator
            action = adapter.moderate(transformed_text)
            outcome = attack_outcome(seed, action)
            
            # 5. Attacker learns from the result
            attacker.update_score(strategy_name, outcome, refused)
            
            if outcome in ("full_evasion", "partial_evasion") and not refused:
                successful_evasions += 1
                status_icon = "🚨"
            elif refused:
                status_icon = "🛑" # Attacker refused
            else:
                status_icon = "🛡️" # Contained
                
            print(f"{status_icon} Seed: {seed.id:<12} | Strategy: {strategy_name:<12} | Result: {outcome:<15} | Action: {action}")

    return total_attacks, successful_evasions


# ── Reporting ────────────────────────────────────────────────

def print_attacker_brain(attacker: AdaptiveAttacker):
    print("\n" + "=" * 80)
    print("🧠 ATTACKER BRAIN: Final Strategy Preferences")
    print("=" * 80)
    print(f"{'Strategy':<15} | {'Attempts':>8} | {'Total Score':>11} | {'Avg Reward':>10} | Status")
    print("-" * 80)
    
    # Sort by average reward
    sorted_strategies = sorted(
        attacker.scores.keys(), 
        key=lambda k: attacker.scores[k] / max(1, attacker.attempts[k]), 
        reverse=True
    )
    
    for strat in sorted_strategies:
        attempts = attacker.attempts[strat]
        score = attacker.scores[strat]
        avg = score / max(1, attempts)
        
        if avg > 1.5:
            status = "🔥 FAVORITE"
        elif avg > 0:
            status = "🔶 SITUATIONAL"
        elif avg < -0.5:
            status = "❌ ABANDONED"
        else:
            status = "⚪ NEUTRAL"
            
        print(f"{strat:<15} | {attempts:>8} | {score:>11.1f} | {avg:>10.2f} | {status}")


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("Step 8: The Adaptive Red-Team Loop")
    print("=" * 90)

    attacker_model = os.getenv("SAFETY_LAB_ATTACKER_MODEL", "granite4.1:3b") # Re-use your Granite model!
    base_url = os.getenv("SAFETY_LAB_OLLAMA_URL", "http://localhost:11434")

    print(f"\nTarget Moderator: regex_moderator")
    print(f"Attacker Model:   {attacker_model}")

    seed_path = Path(__file__).parent.parent / "data" / "seeds.jsonl"
    seeds = load_seeds(str(seed_path))

    adapter = RegexModerator()
    llm_attacker = LLMAttacker(model=attacker_model, base_url=base_url)
    
    # Epsilon = 0.2 means 80% of the time it uses its best strategy, 20% it explores
    adaptive_attacker = AdaptiveAttacker(llm_attacker, epsilon=0.2)

    total, successes = run_adaptive_campaign(adapter, adaptive_attacker, seeds, rounds_per_seed=2)
    
    print(f"\n🏁 Campaign Complete.")
    print(f"Total Attacks: {total}")
    print(f"Successful Evasions: {successes} ({(successes/total)*100:.1f}%)")
    
    print_attacker_brain(adaptive_attacker)

    # Save the "brain" state
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "step08_attacker_brain.json"
    
    payload = {
        "step": "step08_adaptive_attacker",
        "target_moderator": adapter.name,
        "attacker_model": attacker_model,
        "epsilon": adaptive_attacker.epsilon,
        "final_scores": adaptive_attacker.scores,
        "attempts": adaptive_attacker.attempts,
        "total_attacks": total,
        "successful_evasions": successes
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved attacker brain to: {output_path}")


if __name__ == "__main__":
    main()