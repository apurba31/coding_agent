## Error
  |

UP006 [*] Use `dict` instead of `Dict` for type annotation
  --> src/coding_agent/parser/registry.py:8:23
   |
 6 |     def __init__(self):
 7 |
 8 |         self.parsers: Dict[
   |                       ^^^^
 9 |             language,
10 |             SourceParser,
   |
help: Replace with `dict`
  |
7 |
  -         self.parsers: Dict[
8 +         self.parsers: dict[
9 |             language,
  |

F821 Undefined name `language`
  --> src/coding_agent/parser/registry.py:9:13
   |
 8 |         self.parsers: Dict[
 9 |             language,
   |             ^^^^^^^^
10 |             SourceParser,
11 |         ] = ()
   |

I001 [*] Import block is un-sorted or un-formatted
 --> src/coding_agent/scanner/language.py:1:1
  |
1 | from ..models.language import Language
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
help: Organize imports
  |
2 |
  -
3 | _EXTENSION_MAP = {
  |

I001 [*] Import block is un-sorted or un-formatted
  --> src/coding_agent/scanner/scanner.py:1:1
   |
 1 | / from datetime import datetime
 2 | | from pathlib import Path
 3 | |
 4 | | from .ignore import IGNORE_DIRECTORIES, IGNORE_EXTENSIONS
 5 | | from .language import detect_language
 6 | | from .walker import DirectoryWalker
 7 | | from ..models.file import FileMetadata
 8 | | from ..models.repository import Repository
 9 | | from ..utils.hashing import calculate_sha256
   | |____________________________________________^
10 |
11 |   class RepositoryScanner:
   |
help: Organize imports
   |
3  |
   - from .ignore import IGNORE_DIRECTORIES, IGNORE_EXTENSIONS
   - from .language import detect_language
   - from .walker import DirectoryWalker
4  | from ..models.file import FileMetadata
5  | from ..models.repository import Repository
6  | from ..utils.hashing import calculate_sha256
7  + from .ignore import IGNORE_DIRECTORIES, IGNORE_EXTENSIONS
8  + from .language import detect_language
9  + from .walker import DirectoryWalker
10 +
11 |
   |

I001 [*] Import block is un-sorted or un-formatted
 --> src/coding_agent/scanner/walker.py:1:1
  |
1 | from pathlib import Path
  | ^^^^^^^^^^^^^^^^^^^^^^^^
2 |
3 | class DirectoryWalker:
  |
help: Organize imports
  |
2 |
3 +
4 | class DirectoryWalker:
  |

Found 14 errors.
[*] 12 fixable with the `--fix` option.
Error: Process completed with exit code 1.

## Definition of done
Fix the issue