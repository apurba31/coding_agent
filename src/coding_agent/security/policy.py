"""Shared safety checks for filesystem, secrets, and terminal tools.

Keeping these rules in one policy avoids scattering path-traversal and
destructive-command checks across CLI, tools, and the scanner.
"""

from __future__ import annotations

from pathlib import Path

SECRET_FILENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
}

SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx", ".keystore"}

# Substrings that indicate a command the agent must never run automatically.
UNSAFE_COMMAND_MARKERS = (
    ";",
    "&&",
    "||",
    "|",
    "*",
    "?",
    "<",
    ">",
    "`",
    "$",
    "$(",
    "${",
    "rm -rf",
    "rm -r",
    "del /s",
    "format ",
    "mkfs",
    "shutdown",
    "reboot",
    ":(){",
)


class SecurityPolicy:
    """Repository-root confinement, secret exclusion, and command gating."""

    def __init__(
        self,
        repo_root: str | Path,
        max_output_bytes: int = 32_000,
    ) -> None:
        self.root = Path(repo_root).resolve()
        self.max_output_bytes = max_output_bytes

    def resolve_inside_repo(self, relative_or_absolute: str | Path) -> Path:
        """Resolve a user-supplied path and reject anything outside the root."""
        candidate = Path(relative_or_absolute)
        target = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.root / candidate).resolve()
        )
        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                f"Requested path is outside the repository: {relative_or_absolute}"
            ) from exc
        return target

    def is_secret_path(self, path: str | Path) -> bool:
        """True when a file looks like credentials that must not be indexed or read."""
        name = Path(path).name.lower()
        suffix = Path(path).suffix.lower()
        if name in {item.lower() for item in SECRET_FILENAMES}:
            return True
        if name.startswith(".env.") and name != ".env.example":
            return True
        return suffix in SECRET_SUFFIXES

    def allow_read(self, path: str | Path) -> Path:
        """Resolve a readable path, blocking secrets and traversal."""
        target = self.resolve_inside_repo(path)
        if self.is_secret_path(target):
            raise PermissionError(f"Refusing to read secret file: {path}")
        return target

    def allow_write(self, path: str | Path) -> Path:
        """Resolve a writable path, blocking secrets and traversal."""
        target = self.resolve_inside_repo(path)
        if self.is_secret_path(target):
            raise PermissionError(f"Refusing to write secret file: {path}")
        return target

    def allow_command(self, command: str) -> tuple[bool, str]:
        """Reject shell metacharacters and well-known destructive commands."""
        lowered = command.lower()
        for marker in UNSAFE_COMMAND_MARKERS:
            if marker in lowered or marker in command:
                return False, "Command rejected: unsafe tokens or destructive operations."
        return True, ""

    def clip_output(self, text: str) -> str:
        """Cap captured stdout/stderr so a noisy command cannot flood the LLM."""
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= self.max_output_bytes:
            return text
        clipped = encoded[: self.max_output_bytes].decode("utf-8", errors="replace")
        return clipped + "\n...[output truncated]"
