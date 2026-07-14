import re
from enum import Enum
from dataclasses import dataclass
from typing import List


class TokenType(Enum):
    IDENTIFIER = 1
    REGISTER = 2
    IMMEDIATE = 3
    LABEL = 4
    DOT_DIRECTIVE = 5
    COMMA = 6
    COLON = 7
    LBRACE = 8
    RBRACE = 9
    LBRACKET = 10
    RBRACKET = 11
    AT = 12
    NOT = 13
    SEMICOLON = 14
    NEWLINE = 15
    WHITESPACE = 16
    LPAREN = 17
    RPAREN = 18
    PERIOD = 19
    LT = 20
    GT = 21


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int


class PTXLexer:
    def __init__(self):
        self.token_spec = [
            ('LABEL', r'^[A-Za-z_][A-Za-z0-9_]*:'),
            ('REGISTER', r'^%[a-zA-Z][a-zA-Z0-9]*(?:\.[xyz])?'),
            ('IMMEDIATE', r'^0[fFxX][0-9a-fA-F]+|^[0-9]+\.[0-9]+|^[0-9]+'),
            ('DOT_DIRECTIVE', r'^\.[a-zA-Z][a-zA-Z0-9_]*'),
            ('AT', r'^@'),
            ('NOT', r'^!'),
            ('COMMA', r'^,'),
            ('COLON', r'^:'),
            ('LBRACE', r'^\{'),
            ('RBRACE', r'^\}'),
            ('LBRACKET', r'^\['),
            ('RBRACKET', r'^\]'),
            ('SEMICOLON', r'^;'),
            ('LPAREN', r'^\('),
            ('RPAREN', r'^\)'),
            ('PERIOD', r'^\.'),
            ('LT', r'^<'),
            ('GT', r'^>'),
            ('NEWLINE', r'^\n'),
            ('WHITESPACE', r'^[ \t]+'),
            ('IDENTIFIER', r'^[a-zA-Z_][a-zA-Z0-9_]*'),
        ]

    def tokenize(self, source: str) -> List[Token]:
        tokens = []
        line = 1
        column = 1
        source = source + "\n"

        while source:
            matched = False
            for token_name, pattern in self.token_spec:
                match = re.match(pattern, source)
                if match:
                    value = match.group(0)
                    token_type = TokenType[token_name]
                    if token_type != TokenType.WHITESPACE:
                        tokens.append(Token(token_type, value, line, column))
                    source = source[len(value):]
                    column += len(value)
                    if token_type == TokenType.NEWLINE:
                        line += 1
                        column = 1
                    matched = True
                    break
            if not matched:
                tokens.append(Token(TokenType.IDENTIFIER, source[0], line, column))
                source = source[1:]
                column += 1

        return tokens
