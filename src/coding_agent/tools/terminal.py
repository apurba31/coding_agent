"""Safe terminal execution wrapper for repository-scoped tasks."""

from __future__ import annotations

import subprocess
from pathlib import Path


class TerminalTool:
    """Run shell commands while blocking dangerous shell metacharacters."""

    UNSAFE_TOKENS = {";", "&&", "||", "|", "*", "?", "<", ">", "`", "$"}

    def __init__(self, cwd: str | Path | None = None):
        self.cwd = str(Path(cwd).resolve()) if cwd is not None else None

    def _is_safe(self, command: str) -> bool:
        return not any(token in command for token in self.UNSAFE_TOKENS)

    def run(self, command: str) -> dict[str, str | bool | int]:
        if not self._is_safe(command):
            return {
                "success": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "",
                "error": "Command rejected: unsafe shell metacharacters are not allowed.",
            }
        completed = subprocess.run(
            command,
            shell=True,
            cwd=self.cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "success": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "error": "" if completed.returncode == 0 else completed.stderr or completed.stdout,
        }
