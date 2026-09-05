"""
Step 9: LangGraph Multi-Agent Orchestration

Changes from Step 8:
- We transition from a standard Python loop to a LangGraph StateGraph.
- We define distinct nodes: get_next_seed, attacker_choose_strategy, 
  attacker_generate_payload, moderator_evaluate, judge_and_learn.
- This mirrors enterprise agentic pipelines where components operate 
  as independent nodes in a directed graph.
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import TypedDict, List, Dict
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    raise SystemExit("Missing dependency: langgraph\nPlease run: pip install langgraph")

from step03_target_adapters import RegexModerator, load_seeds
from step05_attack_transformations import attack_outcome
from step07_llm_attacker import ATTACK_STRATEGIES, LLMAttacker, REFUSAL_PHRASES


# ── LangGraph State Definition ───────────────────────────────

class CampaignState(TypedDict):
    seed_queue: List[Dict]
    current_seed: Dict
    current_round: int
    max_rounds: int
    
    strategy: str
    payload: str
    attacker_refused: bool
    moderator_action: str
    outcome: str
    
    attacker_scores: Dict[str, float]
    attacker_attempts: Dict[str, int]
    epsilon: float
    
    successful_evasions: int
    total_attacks: int


# ── Node Functions ───────────────────────────────────────────

def get_next_seed(state: CampaignState) -> Dict:
    """Router node: Pops the next seed or resets for the next round."""
    queue = list(state["seed_queue"])
    current_round = state["current_round"]
    
    if not queue:
        if current_round < state["max_rounds"]:
            print(f"\n--- ROUND {current_round + 1} ---")
            # Reshuffle and start next round
            seeds = load_seeds(str(Path(__file__).parent.parent / "data" / "seeds.jsonl"))
            risk_seeds = [s.__dict__ for s in seeds if s.expected_action in {"block", "review"} and s.severity > 0]
            random.shuffle(risk_seeds)
            return {"seed_queue": risk_seeds, "current_round": current_round + 1, "current_seed": {}}
        else:
            # Campaign finished
            return {"seed_queue": [], "current_seed": {}}
            
    next_seed = queue.pop(0)
    return {"seed_queue": queue, "current_seed": next_seed}


def should_continue(state: CampaignState) -> str:
    """Conditional edge: Decides whether to keep attacking or end the graph."""
    if not state["seed_queue"] and not state.get("current_seed"):
        return "end"
    if not state.get("current_seed"):
        return "end"
    return "continue"


def attacker_choose_strategy(state: CampaignState) -> Dict:
    """Attacker Agent Node: Uses Epsilon-Greedy to pick a tactic."""
    scores = state["attacker_scores"]
    attempts = state["attacker_attempts"]
    epsilon = state["epsilon"]
    
    if random.random() < epsilon:
        strategy = random.choice(list(ATTACK_STRATEGIES.keys()))
    else:
        strategy = max(scores, key=lambda k: scores[k] / max(1, attempts[k]))
        
    return {"strategy": strategy}


def attacker_generate_payload(state: CampaignState) -> Dict:
    """LLM Node: Generates the adversarial prompt."""
    model = os.getenv("SAFETY_LAB_ATTACKER_MODEL", "granite4.1:3b")
    base_url = os.getenv("SAFETY_LAB_OLLAMA_URL", "http://localhost:11434")
    attacker = LLMAttacker(model=model, base_url=base_url)
    
    text = state["current_seed"]["text"]
    strategy = state["strategy"]
    prompt_template = ATTACK_STRATEGIES[strategy]
    
    payload = attacker.generate_attack(text, prompt_template)
    refused = any(phrase in payload.lower() for phrase in REFUSAL_PHRASES)
    
    return {"payload": payload, "attacker_refused": refused}


def moderator_evaluate(state: CampaignState) -> Dict:
    """Defender Node: Passes the payload to the target moderator."""
    moderator = RegexModerator()
    action = moderator.moderate(state["payload"])
    return {"moderator_action": action}


def judge_and_learn(state: CampaignState) -> Dict:
    """Evaluator Node: Scores the outcome and updates the Attacker's brain."""
    seed_dict = state["current_seed"]
    
    # Reconstruct a minimal Seed-like object for attack_outcome
    class MockSeed:
        def __init__(self, d):
            self.expected_action = d["expected_action"]
            self.severity = d["severity"]
            
    mock_seed = MockSeed(seed_dict)
    
    outcome = attack_outcome(mock_seed, state["moderator_action"])
    refused = state["attacker_refused"]
    strategy = state["strategy"]
    
    scores = dict(state["attacker_scores"])
    attempts = dict(state["attacker_attempts"])
    
    attempts[strategy] = attempts.get(strategy, 0) + 1
    
    if refused:
        reward = -1.0
    elif outcome == "full_evasion":
        reward = 3.0
    elif outcome == "partial_evasion":
        reward = 1.0
    elif outcome == "over_contained":
        reward = 0.5
    else:
        reward = -2.0
        
    scores[strategy] = scores.get(strategy, 0.0) + reward
    
    success = 1 if (outcome in ("full_evasion", "partial_evasion") and not refused) else 0
    
    # Print step log
    icon = "🚨" if success else ("🛑" if refused else "🛡️")
    print(f"{icon} Seed: {seed_dict['id']:<12} | Strat: {strategy:<12} | Result: {outcome:<15} | Action: {state['moderator_action']}")
    
    return {
        "outcome": outcome,
        "attacker_scores": scores,
        "attacker_attempts": attempts,
        "successful_evasions": state["successful_evasions"] + success,
        "total_attacks": state["total_attacks"] + 1
    }


# ── Graph Construction ───────────────────────────────────────

def build_graph():
    workflow = StateGraph(CampaignState)
    
    # Add nodes
    workflow.add_node("get_next_seed", get_next_seed)
    workflow.add_node("attacker_choose_strategy", attacker_choose_strategy)
    workflow.add_node("attacker_generate_payload", attacker_generate_payload)
    workflow.add_node("moderator_evaluate", moderator_evaluate)
    workflow.add_node("judge_and_learn", judge_and_learn)
    
    # Set entry point
    workflow.set_entry_point("get_next_seed")
    
    # Add conditional edges (Routing)
    workflow.add_conditional_edges(
        "get_next_seed",
        should_continue,
        {
            "continue": "attacker_choose_strategy",
            "end": END
        }
    )
    
    # Add standard edges
    workflow.add_edge("attacker_choose_strategy", "attacker_generate_payload")
    workflow.add_edge("attacker_generate_payload", "moderator_evaluate")
    workflow.add_edge("moderator_evaluate", "judge_and_learn")
    workflow.add_edge("judge_and_learn", "get_next_seed")
    
    return workflow.compile()


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("Step 9: LangGraph Multi-Agent Orchestration")
    print("=" * 90)
    
    seeds = load_seeds(str(Path(__file__).parent.parent / "data" / "seeds.jsonl"))
    risk_seeds = [s.__dict__ for s in seeds if s.expected_action in {"block", "review"} and s.severity > 0]
    random.shuffle(risk_seeds)
    
    initial_state = {
        "seed_queue": risk_seeds,
        "current_seed": {},
        "current_round": 1,
        "max_rounds": 2,
        "strategy": "",
        "payload": "",
        "attacker_refused": False,
        "moderator_action": "",
        "outcome": "",
        "attacker_scores": {name: 1.0 for name in ATTACK_STRATEGIES.keys()},
        "attacker_attempts": {name: 0 for name in ATTACK_STRATEGIES.keys()},
        "epsilon": 0.2,
        "successful_evasions": 0,
        "total_attacks": 0
    }
    
    print(f"\n🚀 Starting LangGraph Agentic Campaign...")
    print(f"Target: {len(risk_seeds)} risky seeds | Rounds: {initial_state['max_rounds']}\n--- ROUND 1 ---")
    
    app = build_graph()
    final_state = app.invoke(initial_state)
    
    print(f"\n🏁 Campaign Complete.")
    print(f"Total Attacks: {final_state['total_attacks']}")
    print(f"Successful Evasions: {final_state['successful_evasions']}")
    
    print("\n🧠 ATTACKER BRAIN (LangGraph State):")
    for strat in ATTACK_STRATEGIES.keys():
        score = final_state["attacker_scores"][strat]
        att = final_state["attacker_attempts"][strat]
        print(f"  {strat:<12} | Attempts: {att:<3} | Score: {score:>5.1f}")

    # Save state
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "step09_langgraph_state.json"
    
    # Convert state to serializable dict
    save_state = dict(final_state)
    save_state["seed_queue"] = [] # Don't save the empty queue
    save_state["current_seed"] = {}
    
    output_path.write_text(json.dumps(save_state, indent=2), encoding="utf-8")
    print(f"\nSaved LangGraph final state to: {output_path}")


if __name__ == "__main__":
    main()