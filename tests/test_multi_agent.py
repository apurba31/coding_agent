"""Tests for sequential researcher-to-synthesizer orchestration."""

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
