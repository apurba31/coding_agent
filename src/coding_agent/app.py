from pathlib import Path

from .scanner.scanner import RepositoryScanner


def main():
    scanner = RepositoryScanner(Path("."))
    repo = scanner.scan(Path("."))
    print()
    print(f"Indexed : {repo.indexed_files}")
    print(f"Ignored : {repo.ignored_files}")
    print(f"Folders : {repo.directories}")
    print()

    for file in repo.files[:10]:
        print(file.path, file.language)


if __name__ == "__main__":
    main()
