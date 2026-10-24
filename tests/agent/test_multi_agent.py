"""Tests for bounded researcher delegation and synthesis."""

import threading

from coding_agent.agent.team import MultiAgentCoordinator


def _state(status: str, content: str = "") -> dict:
    return {
        "goal": "question",
        "messages": [{"role": "assistant", "content": content}],
        "step_count": 1,
        "max_steps": 5,
        "status": status,
    }


class StubAgent:
    def __init__(self, state):
        self.state = state
        self.calls = []

    def execute(self, goal, conversation_id=None, additional_context=None):
        self.calls.append(
            {
                "goal": goal,
                "conversation_id": conversation_id,
                "additional_context": additional_context,
            }
        )
        return self.state

    def close(self):
        pass


def test_coordinator_hands_research_to_synthesizer_and_preserves_user_goal():
    researcher = StubAgent(_state("SUCCESS", "The function is in src/math.py and adds 42."))
    synthesizer = StubAgent(_state("SUCCESS", "It adds 42."))
    coordinator = MultiAgentCoordinator(researcher, synthesizer)

    result = coordinator.execute("Explain the calculation", conversation_id="thread-1")

    assert result.status == "SUCCESS"
    assert result.answer == "It adds 42."
    assert researcher.calls == [
        {
            "goal": "Explain the calculation",
            "conversation_id": None,
            "additional_context": None,
        }
    ]
    assert synthesizer.calls[0]["goal"] == "Explain the calculation"
    assert synthesizer.calls[0]["conversation_id"] == "thread-1"
    assert "src/math.py" in synthesizer.calls[0]["additional_context"]


def test_coordinator_stops_if_research_agent_fails():
    researcher = StubAgent(_state("FAILED", "research error"))
    synthesizer = StubAgent(_state("SUCCESS", "should not run"))

    result = MultiAgentCoordinator(researcher, synthesizer).execute("Explain this")

    assert result.status == "FAILED"
    assert result.synthesis is None
    assert result.answer == ""
    assert synthesizer.calls == []


def test_coordinator_decomposes_and_runs_research_tasks_in_parallel():
    lock = threading.Lock()
    barrier = threading.Barrier(5)
    state = {"active": 0, "max_active": 0}
    started = {"count": 0}
    contexts = []

    class ParallelStubAgent:
        def execute(self, goal, conversation_id=None, additional_context=None):
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                started["count"] += 1
                should_wait = started["count"] <= 5
                contexts.append(additional_context)
            try:
                if should_wait:
                    barrier.wait(timeout=2)
                return _state("SUCCESS", additional_context)
            finally:
                with lock:
                    state["active"] -= 1

        def close(self):
            pass

    synthesizer = StubAgent(_state("SUCCESS", "Combined findings."))
    coordinator = MultiAgentCoordinator(
        researcher=StubAgent(_state("SUCCESS", "unused")),
        synthesizer=synthesizer,
        researcher_factory=ParallelStubAgent,
        max_parallel_researchers=5,
    )

    result = coordinator.execute(
        "Find implementation; Find tests; Find docs; Find callers; Find risks; Find config"
    )

    assert result.status == "SUCCESS"
    assert len(result.researches) == 6
    assert state["max_active"] == 5
    assert len(contexts) == 6
    assert "Find implementation" in synthesizer.calls[0]["additional_context"]
    assert "Find config" in synthesizer.calls[0]["additional_context"]


def test_coordinator_keeps_simple_goal_as_one_research_task():
    researcher = StubAgent(_state("SUCCESS", "One focused investigation."))
    synthesizer = StubAgent(_state("SUCCESS", "Answer."))

    result = MultiAgentCoordinator(researcher, synthesizer).execute(
        "Explain the calculation"
    )

    assert result.status == "SUCCESS"
    assert len(result.researches) == 1
    assert researcher.calls[0]["additional_context"] is None
