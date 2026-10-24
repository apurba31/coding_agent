# Security

- Repository-root confinement for all file tools
- Secret filename/suffix denylist for read/write/index
- Terminal command denylist (metacharacters, destructive patterns)
- Output clipping for terminal and test runs
- `.gitignore` respected by the scanner

Write and terminal capabilities require explicit CLI flags in chat mode.
