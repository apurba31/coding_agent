from ..models.language import Language

_EXTENSION_MAP = {
    ".java": Language.JAVA,
    ".py": Language.PYTHON,
    ".js": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".kt": Language.KOTLIN,
    ".yaml": Language.YAML,
    ".yml": Language.YAML,
    ".json": Language.JSON,
    ".md": Language.MARKDOWN,
    ".xml": Language.XML,
}


def detect_language(extension: str) -> Language:
    return _EXTENSION_MAP.get(extension.lower(), Language.UNKNOWN)