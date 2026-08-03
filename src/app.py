try:
    from .coding_agent.app import main
except ImportError:  # pragma: no cover - fallback for direct script execution
    from src.coding_agent.app import main


if __name__ == "__main__":
    main()
