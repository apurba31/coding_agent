import json
import os
from typing import Dict, Literal, TypedDict

# 1. DEFINE THE GRAPH STATE SCHEMA
class GraphState(TypedDict):
    goal: str
    code: str
    test_passed: bool
    step_count: int
    max_steps: int
    current_node: Literal["planner", "coder", "tester", "debugger", "end"]

class AdvancedNanoHarness:
    def __init__(self, goal: str, max_steps: int = 6):
        self.state: GraphState = {
            "goal": goal,
            "code": "",
            "test_passed": False,
            "step_count": 0,
            "max_steps": max_steps,
            "current_node": "planner"
        }

    def checkpoint(self):
        """Saves session state to disk for step-by-step auditing."""
        os.makedirs(".harness_logs", exist_ok=True)
        filename = f".harness_logs/step_{self.state['step_count']}_{self.state['current_node']}.json"
        with open(filename, "w") as f:
            json.dump(self.state, f, indent=2)

    def run(self):
        print(f"[*] Initializing Graph Harness for Goal: '{self.state['goal']}'\n")
        
        while self.state["current_node"] != "end" and self.state["step_count"] < self.state["max_steps"]:
            self.state["step_count"] += 1
            node = self.state["current_node"]
            print(f"========================================")
            print(f"Step {self.state['step_count']} | Active Node: [{node.upper()}]")
            print(f"========================================")

            # --- NODE 1: PLANNER ---
            if node == "planner":
                print("[Planner Node] Analyzing goal and breaking down requirements...")
                # Transition to coder node
                self.state["current_node"] = "coder"

            # --- NODE 2: CODER ---
            elif node == "coder":
                print("[Coder Node] Writing initial implementation...")
                # Simulating an LLM making a deliberate mistake on the first attempt
                self.state["code"] = "def compute_sum(a, b):\n    return a - b  # Bug introduced here!"
                print(f"Generated Code:\n{self.state['code']}\n")
                # Transition to tester node
                self.state["current_node"] = "tester"

            # --- NODE 3: TESTER ---
            elif node == "tester":
                print("[Tester Node] Executing deterministic unit tests...")
                try:
                    # Execute code in a safe local namespace to verify
                    local_namespace = {}
                    exec(self.state["code"], {}, local_namespace)
                    result = local_namespace["compute_sum"](10, 5)
                    
                    if result == 15:
                        self.state["test_passed"] = True
                        print(f"[Tester] Test passed successfully! Output: {result}")
                    else:
                        self.state["test_passed"] = False
                        print(f"[Tester] Test FAILED. Expected 15, got {result}")
                except Exception as e:
                    self.state["test_passed"] = False
                    print(f"[Tester] Exception raised during test: {e}")

                # CONDITIONAL EDGE ROUTING (Cyclic Graph Logic)
                if self.state["test_passed"]:
                    self.state["current_node"] = "end"
                else:
                    self.state["current_node"] = "debugger"

            # --- NODE 4: DEBUGGER (Self-Correction Loop) ---
            elif node == "debugger":
                print("[Debugger Node] Analyzing test failure and rewriting code...")
                # Fixing the bug based on test feedback
                self.state["code"] = "def compute_sum(a, b):\n    return a + b  # Bug fixed!"
                print(f"Corrected Code:\n{self.state['code']}\n")
                # Route back to tester to re-verify
                self.state["current_node"] = "tester"

            self.checkpoint()

        print(f"\n[Harness Finished] Final Status -> Goal Achieved: {self.state['test_passed']}")

if __name__ == "__main__":
    harness = AdvancedNanoHarness(goal="Create a working compute_sum function")
    harness.run()