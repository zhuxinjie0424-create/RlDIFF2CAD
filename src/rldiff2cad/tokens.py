"""Token vocabulary and sketch sequence utilities.

The module models CAD sketches as discrete token sequences. A token can
represent a primitive (e.g., LINE, ARC), a constraint (e.g., PARALLEL), or
references to previous primitives. Minimal range checks and helpers are
provided to keep the rest of the codebase uncluttered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Sequence


class TokenKind(str, Enum):
    """Kinds of tokens found in a sketch sequence."""

    PRIMITIVE = "primitive"
    CONSTRAINT = "constraint"
    REFERENCE = "reference"
    PARAMETER = "parameter"


@dataclass(frozen=True)
class Token:
    """A single token with kind and textual value."""

    kind: TokenKind
    value: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.kind}:{self.value}"


@dataclass
class Vocabulary:
    """Vocabulary of allowable tokens.

    The vocabulary is kept intentionally small but easily extensible. Add new
    primitives, constraints, or reference markers here to teach the system
    about additional CAD concepts.
    """

    primitives: Sequence[str] = (
        "POINT",
        "LINE",
        "ARC",
        "CIRCLE",
    )
    constraints: Sequence[str] = (
        "HORIZONTAL",
        "VERTICAL",
        "PARALLEL",
        "PERPENDICULAR",
        "EQUAL",
        "RADIUS",
        "DIMENSION",
    )
    parameters: Sequence[str] = (
        "X",
        "Y",
        "ANGLE",
        "R",
    )
    max_references: int = 8

    def all_tokens(self) -> List[Token]:
        refs = [f"REF_{i}" for i in range(self.max_references)]
        return [
            *[Token(TokenKind.PRIMITIVE, t) for t in self.primitives],
            *[Token(TokenKind.CONSTRAINT, t) for t in self.constraints],
            *[Token(TokenKind.PARAMETER, t) for t in self.parameters],
            *[Token(TokenKind.REFERENCE, r) for r in refs],
        ]

    def validate(self, token: Token) -> None:
        """Validate that a token exists in the vocabulary."""

        if token.kind == TokenKind.PRIMITIVE and token.value not in self.primitives:
            raise ValueError(f"Unknown primitive token: {token.value}")
        if token.kind == TokenKind.CONSTRAINT and token.value not in self.constraints:
            raise ValueError(f"Unknown constraint token: {token.value}")
        if token.kind == TokenKind.PARAMETER and token.value not in self.parameters:
            raise ValueError(f"Unknown parameter token: {token.value}")
        if token.kind == TokenKind.REFERENCE:
            prefix, _, idx = token.value.partition("_")
            if prefix != "REF" or not idx.isdigit() or int(idx) >= self.max_references:
                raise ValueError(f"Reference token out of bounds: {token.value}")


@dataclass
class SketchSequence:
    """Sequence container for CAD sketch tokens."""

    tokens: List[Token] = field(default_factory=list)
    vocabulary: Vocabulary = field(default_factory=Vocabulary)

    def append(self, token: Token) -> None:
        self.vocabulary.validate(token)
        self.tokens.append(token)

    def extend(self, tokens: Iterable[Token]) -> None:
        for token in tokens:
            self.append(token)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.tokens)

    def __iter__(self):  # pragma: no cover - trivial
        return iter(self.tokens)

    def to_strings(self) -> List[str]:
        """Return a string representation of the sequence for logging/serialization."""

        return [str(t) for t in self.tokens]

    def summary(self) -> str:
        """Readable summary used in logs."""

        primitive_count = sum(1 for t in self.tokens if t.kind == TokenKind.PRIMITIVE)
        constraint_count = sum(1 for t in self.tokens if t.kind == TokenKind.CONSTRAINT)
        return (
            f"SketchSequence(len={len(self.tokens)}, "
            f"primitives={primitive_count}, constraints={constraint_count})"
        )
