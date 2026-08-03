import hashlib
from pathlib import Path


def calculate_sha256(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()