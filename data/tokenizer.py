import json
from typing import Dict, List


class Tokenizer:
    """Simple tokenizer mapping CAD tokens to ids and back."""

    def __init__(self) -> None:
        primitives = ["LINE", "ARC", "CIRCLE", "POINT"]
        constraints = [
            "HORIZONTAL",
            "VERTICAL",
            "PARALLEL",
            "PERPENDICULAR",
            "TANGENT",
            "EQUAL",
            "RADIUS",
            "DIMENSION",
        ]
        references = [f"REF_{i}" for i in range(16)]
        parameters = [
            *(f"X_{i}" for i in range(16)),
            *(f"Y_{i}" for i in range(16)),
            *(f"LEN_{i}" for i in range(16)),
            *(f"ANG_{i}" for i in range(16)),
        ]
        self.tokens: List[str] = primitives + constraints + references + parameters
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.id_to_token: Dict[int, str] = {idx: tok for tok, idx in self.token_to_id.items()}

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id[tok] for tok in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_token[idx] for idx in ids]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.tokens, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, "r", encoding="utf-8") as f:
            tokens = json.load(f)
        tokenizer = cls()
        tokenizer.tokens = tokens
        tokenizer.token_to_id = {tok: idx for idx, tok in enumerate(tokens)}
        tokenizer.id_to_token = {idx: tok for tok, idx in tokenizer.token_to_id.items()}
        return tokenizer
