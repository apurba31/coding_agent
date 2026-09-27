import json
import os
from collections.abc import Callable
from typing import TypedDict


# 1. DEFINE THE STATE SCHEMA
class AgentState(TypedDict):
    goal: str
    messages: list[dict[str, str]]
    step_count: int
    max_steps: int
    status: str  # "RUNNING", "SUCCESS", "FAILED"


# 2. DEFINE A SIMPLE TOOL REGISTRY
def tool_read_file(filepath: str) -> str:
    """Reads a local file."""
    try:
        with open(filepath) as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


TOOL_REGISTRY: dict[str, Callable] = {"read_file": tool_read_file}


# 3. THE HARNESS LOOP & STATE MACHINE
class NanoHarness:
    def __init__(self, goal: str, max_steps: int = 5):
        self.state: AgentState = {
            "goal": goal,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a focused coding agent. Accomplish the goal using available tools."
                    ),
                }
            ],
            "step_count": 0,
            "max_steps": max_steps,
            "status": "RUNNING",
        }

    def checkpoint(self):
        """Saves state to disk for inspection (Crucial for debugging loops)."""
        os.makedirs(".harness_logs", exist_ok=True)
        filename = f".harness_logs/step_{self.state['step_count']}.json"
        with open(filename, "w") as f:
            json.dump(self.state, f, indent=2)
        print(f"[*] Checkpointed state to {filename}")

    def step(self):
        """A single execution tick of the loop graph."""
        self.state["step_count"] += 1
        print(f"\n--- STEP {self.state['step_count']} / {self.state['max_steps']} ---")

        # Termination Guard: Max Steps Check
        if self.state["step_count"] > self.state["max_steps"]:
            self.state["status"] = "FAILED"
            print("[!] Max steps reached. Aborting loop to prevent infinite cost/thrashing.")
            return

        # Simulate LLM Decision (In a real app, call OpenAI/Anthropic API here with tool schemas)
        # For this toy, we simulate the model deciding to read a file on step 1, then finishing.
        if self.state["step_count"] == 1:
            action = {"tool": "read_file", "args": {"filepath": "sample.txt"}}
            print(f"[Model Decision] Calling tool: {action['tool']} with {action['args']}")

            # Execute Tool
            tool_output = TOOL_REGISTRY[action["tool"]](**action["args"])
            print(f"[Tool Output] {tool_output}")

            # Append to state history
            self.state["messages"].append({"role": "assistant", "content": "Used tool read_file"})
            self.state["messages"].append({"role": "tool", "content": tool_output})
        else:
            # Model finishes task
            self.state["status"] = "SUCCESS"
            print("[Model Decision] Task completed successfully.")

        self.checkpoint()

    def run(self):
        """The main orchestration loop."""
        while self.state["status"] == "RUNNING":
            self.step()
        print(f"\n[Harness Finished] Final Status: {self.state['status']}")


# --- EXECUTION ---
if __name__ == "__main__":
    # Create a dummy file to test our harness
    with open("sample.txt", "w") as f:
        f.write("Hello from the local workspace file!")

    harness = NanoHarness(goal="Read sample.txt and summarize it", max_steps=3)
    harness.run()
