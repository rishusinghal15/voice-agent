"""Unit tests for SentenceBuffer edge cases, abbreviations, decimals, and streaming chunks."""

from backend.app.sentence_buffer import SentenceBuffer


def test_basic_sentence_splitting():
    """Verify standard sentences ending with period, exclamation, and question mark."""
    sb = SentenceBuffer()

    # Feed with trailing space
    sentences = list(sb.feed("Hello world. How are you today? "))
    assert sentences == ["Hello world.", "How are you today?"]

    # Trailing without space is kept in buffer
    sentences2 = list(sb.feed("I am fine! And you"))
    assert sentences2 == ["I am fine!"]

    # Flush remaining text
    remaining = sb.flush()
    assert remaining == "And you"


def test_token_by_token_streaming():
    """Verify sentence buffer works when text arrives token by token."""
    sb = SentenceBuffer()
    tokens = ["Good ", "morning", ", ", "everyone", ". ", "Welcome ", "to ", "the ", "demo", "!"]
    yielded = []

    for t in tokens:
        for s in sb.feed(t):
            yielded.append(s)

    assert yielded == ["Good morning, everyone."]
    # The second sentence is flushed at the end
    assert sb.flush() == "Welcome to the demo!"


def test_abbreviation_dr_mr_mrs():
    """Verify abbreviations Dr., Mr., Mrs. do not trigger sentence boundaries."""
    sb = SentenceBuffer()
    text = "Dr. Smith met with Mr. Johnson and Mrs. Davis today. They had a great talk. "
    sentences = list(sb.feed(text))

    assert len(sentences) == 2
    assert sentences[0] == "Dr. Smith met with Mr. Johnson and Mrs. Davis today."
    assert sentences[1] == "They had a great talk."


def test_abbreviations_eg_ie_vs():
    """Verify e.g., i.e., and vs. do not trigger sentence boundaries."""
    sb = SentenceBuffer()
    text = "You can choose e.g. red or blue. In other words, i.e. your favorite color. It is red vs. blue. "
    sentences = list(sb.feed(text))

    assert len(sentences) == 3
    assert sentences[0] == "You can choose e.g. red or blue."
    assert sentences[1] == "In other words, i.e. your favorite color."
    assert sentences[2] == "It is red vs. blue."


def test_decimals_not_split():
    """Verify decimal numbers (e.g. 3.5, 99.99) are never split."""
    sb = SentenceBuffer()
    text = "The product rating is 4.8 stars out of 5.0. It costs 3.5 dollars. "
    sentences = list(sb.feed(text))

    assert len(sentences) == 2
    assert sentences[0] == "The product rating is 4.8 stars out of 5.0."
    assert sentences[1] == "It costs 3.5 dollars."


def test_multiple_punctuation_and_ellipsis():
    """Verify multiple punctuation marks (e.g. '?!' or '...') are preserved properly."""
    sb = SentenceBuffer()
    text = "Wait... what happened?! Really? Yes! "
    sentences = list(sb.feed(text))

    assert sentences == ["Wait...", "what happened?!", "Really?", "Yes!"]


def test_flush_behavior():
    """Verify flush clears buffer and subsequent flush returns empty string."""
    sb = SentenceBuffer()
    assert sb.is_empty()

    sb.feed("Incomplete sentence without punctuation")
    assert not sb.is_empty()

    remaining = sb.flush()
    assert remaining == "Incomplete sentence without punctuation"
    assert sb.is_empty()
    assert sb.flush() == ""

