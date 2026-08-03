from typing import Dict
from coding_agent.models.language import Language
from .parser import SourceParser

class ParserRegistry:
    def __init__(self):

        self.parsers: Dict[
            language,
            SourceParser,
        ] = ()

    def register(
            self,
            language: Language,
            parser: SourceParser,
    ): 
        self.parsers[language] = parser

    def get(
            self,
            language: Language,
    ) -> SourceParser:
        return self.parsers[language]
    
