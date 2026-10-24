# Parsing

Tree-sitter runs behind `SourceParser` / `ParserRegistry`. Supported languages: Python, Java, JavaScript, TypeScript, Go, Rust.

Malformed files should not crash indexing; parsers return partial symbol trees where possible.

Symbols normalize to `name`, `kind`, line range, and `parent` in `coding_agent/models/ast.py`.
