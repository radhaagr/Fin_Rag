"""Splitter strategies + factory. Add a new splitter by adding one class and
one registry entry — nothing else in the codebase needs to change."""

from dataclasses import dataclass


class CharSplitter:
    def __init__(self, chunk_size: int, overlap: int):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        if not text:
            return []
        step = self.chunk_size - self.overlap
        out = []
        for start in range(0, len(text), step):
            piece = text[start:start + self.chunk_size]
            if piece.strip():
                out.append(piece)
            if start + self.chunk_size >= len(text):
                break
        return out


class RecursiveCharSplitter:
    def __init__(self, chunk_size: int, overlap: int):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    def split(self, text: str) -> list[str]:
        return self._splitter.split_text(text) if text else []


class TokenSplitter:
    def __init__(self, chunk_size: int, overlap: int):
        from langchain_text_splitters import TokenTextSplitter
        self._splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    def split(self, text: str) -> list[str]:
        return self._splitter.split_text(text) if text else []


SPLITTER_REGISTRY = {
    "char": CharSplitter,
    "recursive_char": RecursiveCharSplitter,
    "token": TokenSplitter,
}


def get_splitter(name: str, chunk_size: int, overlap: int):
    """Factory: look up a splitter class by name and construct it.
    Raises ImportError naturally if an optional dependency is missing —
    callers should catch that and skip the config, not crash the sweep."""
    if name not in SPLITTER_REGISTRY:
        raise KeyError(f"Unknown splitter '{name}'. Registered: {list(SPLITTER_REGISTRY)}")
    return SPLITTER_REGISTRY[name](chunk_size, overlap)
    