from tree_sitter_language_pack import get_language


class GrammarLoader:
    def load(self, language: str):
        return get_language(language)