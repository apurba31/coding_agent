import importlib


def test_coding_agent_app_imports():
    module = importlib.import_module("src.coding_agent.app")
    assert hasattr(module, "main")
