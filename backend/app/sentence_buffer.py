"""Sentence buffer for chunking streamed tokens into completed sentences for TTS.

Follows design doc §3 & §5:
- feed(delta) yields completed sentences at [.!?] followed by whitespace.
- Does not split on abbreviations (Dr., Mr., Mrs., e.g., i.e., vs.) or decimals (3.5).
- flush() returns the remaining buffered text.
"""

from __future__ import annotations

import re
from typing import Iterator, List, Set

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


class SentenceBuffer:
    """Buffers streamed token deltas and yields completed sentences for TTS processing."""

    def __init__(self) -> None:
        self.buffer: str = ""

    def _is_abbreviation(self, text_up_to_punct: str) -> bool:
        """Check if the word ending with the matched period is a known abbreviation."""
        # Split into words or tokens
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
        """Append token delta and yield any newly completed sentences."""
        self.buffer += delta

        def _generator() -> Iterator[str]:
            while True:
                # Search for candidate sentence boundary
                found_boundary = False
                for match in PUNCT_SPLIT_REGEX.finditer(self.buffer):
                    punct = match.group(1)
                    punct_end = match.end(1)
                    text_before = self.buffer[:punct_end].strip()

                    # Check abbreviation if punctuation is or ends with a dot
                    if punct.endswith("."):
                        if self._is_abbreviation(text_before):
                            # Part of abbreviation; skip this match and continue search
                            continue

                    # Valid sentence boundary found
                    candidate_sentence = text_before
                    if candidate_sentence:
                        yield candidate_sentence

                    # Slice buffer past this sentence and its trailing whitespace
                    self.buffer = self.buffer[match.end():]
                    found_boundary = True
                    break

                if not found_boundary:
                    break

        return _generator()

    def flush(self) -> str:
        """Return and clear any remaining text in the buffer."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining

    def is_empty(self) -> bool:
        """Return True if the internal buffer is empty or contains only whitespace."""
        return not bool(self.buffer.strip())
