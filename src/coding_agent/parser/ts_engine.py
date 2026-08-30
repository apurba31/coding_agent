from tree_sitter import Language, Parser

from .loader import GrammarLoader


def create_parser(language: str) -> Parser:
    ts_language: Language = GrammarLoader().load(language)
    return Parser(ts_language)
