from pathlib import Path

IGNORE_DIRECTORIES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "build",
    "dist",
    "target",
    ".cache",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".mini-agent",
}

IGNORE_EXTENSIONS = {
    ".class",
    ".jar",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
}

# Filenames we never index even if they sit inside the repository.
IGNORE_FILENAMES = {
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


def load_gitignore_patterns(root: Path) -> list[str]:
    """Load basic .gitignore patterns. Negation (`!`) is ignored on purpose.

    Ordinary scanning must work without Git installed, so we parse the file
    ourselves instead of calling `git check-ignore`.
    """
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return []
    patterns: list[str] = []
    for raw_line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        patterns.append(line.rstrip("/"))
    return patterns


def matches_gitignore(relative_path: Path, patterns: list[str]) -> bool:
    """Return True when a repository-relative path matches a simple gitignore glob."""
    posix = relative_path.as_posix()
    parts = relative_path.parts
    for pattern in patterns:
        if pattern.startswith("*."):
            if relative_path.suffix.lower() == pattern[1:].lower():
                return True
            continue
        if pattern.startswith("*"):
            if posix.endswith(pattern[1:]) or relative_path.name.endswith(pattern[1:]):
                return True
            continue
        if pattern in parts or posix == pattern or posix.startswith(pattern + "/"):
            return True
    return False


def looks_binary(path: Path, sample_size: int = 8192) -> bool:
    """Treat a NUL byte in the first chunk as binary so we skip blobs and secrets dumps."""
    try:
        with path.open("rb") as handle:
            sample = handle.read(sample_size)
    except OSError:
        return True
    return b"\0" in sample
