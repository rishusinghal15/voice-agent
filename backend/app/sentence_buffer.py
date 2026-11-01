"""Sentence buffer for chunking streamed tokens into completed sentences for TTS.

Follows design doc §3 & §5:
- feed(delta) yields completed sentences at [.!?] followed by whitespace.
- First-chunk rule: emits the first chunk of a reply at a comma or clause
  boundary ([,;:] or dash) once it has >= FIRST_CHUNK_MIN_WORDS (default 6) words.
- Does not split on abbreviations (Dr., Mr., Mrs., e.g., i.e., vs.) or decimals (3.5).
- flush() returns the remaining buffered text.
"""

from __future__ import annotations

import re
from typing import Iterator, List, Optional, Set

FIRST_CHUNK_MIN_WORDS: int = 6

# Known abbreviations that end in a period and do not constitute sentence boundaries
ABBREVIATIONS: Set[str] = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "sr.",
    "jr.",
    "vs.",
    "e.g.",
    "i.e.",
    "etc.",
}

# Regex to find word immediately preceding a period
PREV_WORD_REGEX = re.compile(r"([a-zA-Z.]+)\.$")

# Regex to find [.!?] followed by one or more whitespace characters
PUNCT_SPLIT_REGEX = re.compile(r"([.!?]+)(\s+)")

# Regex to find clause boundaries [,;:] or dashes followed by whitespace
CLAUSE_SPLIT_REGEX = re.compile(r"([,;:]|--|—)(\s+)")


class SentenceBuffer:
    """Buffers streamed token deltas and yields completed sentences for TTS processing."""

    def __init__(self, first_chunk_min_words: int = FIRST_CHUNK_MIN_WORDS) -> None:
        self.buffer: str = ""
        self.first_chunk_min_words: int = first_chunk_min_words
        self.first_chunk_emitted: bool = False

    def reset(self) -> None:
        """Reset buffer and first chunk state."""
        self.buffer = ""
        self.first_chunk_emitted = False

    def _is_abbreviation(self, text_up_to_punct: str) -> bool:
        """Check if the word ending with the matched period is a known abbreviation."""
        words = text_up_to_punct.split()
        if not words:
            return False
        last_word = words[-1].lower()
        if last_word in ABBREVIATIONS:
            return True
        match = PREV_WORD_REGEX.search(text_up_to_punct)
        if match:
            candidate = match.group(0).lower()
            if candidate in ABBREVIATIONS:
                return True
        return False

    def feed(self, delta: str) -> Iterator[str]:
        """Append token delta and yield any newly completed sentences or first clause chunk."""
        self.buffer += delta

        def _generator() -> Iterator[str]:
            while True:
                # Find earliest valid sentence boundary
                valid_punct_match = None
                for match in PUNCT_SPLIT_REGEX.finditer(self.buffer):
                    punct = match.group(1)
                    punct_end = match.end(1)
                    text_before = self.buffer[:punct_end].strip()

                    # Check abbreviation if punctuation is or ends with a dot
                    if punct.endswith("."):
                        if self._is_abbreviation(text_before):
                            continue
                    valid_punct_match = (match, text_before)
                    break

                # If first chunk has not been emitted, check clause boundary with min words
                valid_clause_match = None
                if not self.first_chunk_emitted:
                    for match in CLAUSE_SPLIT_REGEX.finditer(self.buffer):
                        clause_end = match.end(1)
                        clause_text = self.buffer[:clause_end].strip()
                        words = clause_text.split()
                        if len(words) >= self.first_chunk_min_words:
                            valid_clause_match = (match, clause_text)
                            break

                # Decide which boundary to take
                if valid_clause_match and valid_punct_match:
                    clause_m, clause_text = valid_clause_match
                    punct_m, punct_text = valid_punct_match
                    if clause_m.start() < punct_m.start():
                        yield clause_text
                        self.first_chunk_emitted = True
                        self.buffer = self.buffer[clause_m.end():]
                        continue
                    else:
                        yield punct_text
                        self.first_chunk_emitted = True
                        self.buffer = self.buffer[punct_m.end():]
                        continue
                elif valid_clause_match:
                    clause_m, clause_text = valid_clause_match
                    yield clause_text
                    self.first_chunk_emitted = True
                    self.buffer = self.buffer[clause_m.end():]
                    continue
                elif valid_punct_match:
                    punct_m, punct_text = valid_punct_match
                    yield punct_text
                    self.first_chunk_emitted = True
                    self.buffer = self.buffer[punct_m.end():]
                    continue
                else:
                    break

        return _generator()

    def flush(self) -> str:
        """Return and clear any remaining text in the buffer."""
        remaining = self.buffer.strip()
        self.buffer = ""
        self.first_chunk_emitted = False
        return remaining

    def is_empty(self) -> bool:
        """Return True if the internal buffer is empty or contains only whitespace."""
        return not bool(self.buffer.strip())
