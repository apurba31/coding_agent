def main() -> None:
    """Run the package CLI when invoked as `python -m src.app`."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from coding_agent.app import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
