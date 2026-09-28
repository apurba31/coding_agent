"""Safe terminal execution wrapper for repository-scoped tasks."""

from __future__ import annotations

import subprocess
from pathlib import Path

from coding_agent.security import SecurityPolicy


class TerminalTool:
    """Run shell commands with timeout, output caps, and a destructive-command denylist."""

    def __init__(
        self,
        cwd: str | Path | None = None,
        policy: SecurityPolicy | None = None,
        timeout_seconds: float = 15.0,
        max_output_bytes: int = 32_000,
    ):
        self.cwd = str(Path(cwd).resolve()) if cwd is not None else None
        self.policy = policy or SecurityPolicy(cwd or Path("."), max_output_bytes=max_output_bytes)
        self.timeout_seconds = timeout_seconds

    def run(self, command: str) -> dict[str, str | bool | int]:
        allowed, reason = self.policy.allow_command(command)
        if not allowed:
            return {
                "success": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "",
                "error": reason,
            }
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "exit_code": 124,
                "stdout": "",
                "stderr": "",
                "error": f"Command timed out after {self.timeout_seconds}s",
            }
        stdout = self.policy.clip_output(completed.stdout)
        stderr = self.policy.clip_output(completed.stderr)
        return {
            "success": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "error": "" if completed.returncode == 0 else stderr or stdout,
        }
