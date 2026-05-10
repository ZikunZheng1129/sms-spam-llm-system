"""Risk-feature extraction for SMS messages.

These features are explainability-only signals. They are intended to be logged
alongside model predictions so notebooks can answer questions like
"on which kinds of messages does each mode fail?". They are NOT used as hard
classification rules and the benchmark pipeline does not feed them into any
classifier.
"""

from __future__ import annotations

import re
from typing import Any


_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\b\S+\.(?:com|net|org|io|co\.uk|info|biz)\b",
    re.IGNORECASE,
)

_PHONE_RE = re.compile(
    r"(?:\+?\d[\s\-]?){7,}|\b\d{4,6}\b"
)

_CURRENCY_CHARS = frozenset("£$€¥₹")
_CURRENCY_WORDS = ("pound", "pounds", "usd", "eur", "gbp", "dollar", "dollars")

_URGENT_WORDS = (
    "urgent",
    "immediately",
    "now",
    "asap",
    "today",
    "expire",
    "expires",
    "expiring",
)

_PRIZE_WORDS = (
    "prize",
    "won",
    "winner",
    "free",
    "cash",
    "reward",
    "claim",
    "voucher",
    "selected",
    "congratulations",
    "congrats",
)

_CALL_REPLY_WORDS = ("call", "reply", "text", "txt", "send", "ring")


def _contains_any_word(text: str, words: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(re.search(rf"\b{re.escape(w)}\b", lower) for w in words)


def contains_url(text: str) -> bool:
    return bool(_URL_RE.search(text))


def contains_phone_number(text: str) -> bool:
    return bool(_PHONE_RE.search(text))


def contains_currency(text: str) -> bool:
    if any(c in text for c in _CURRENCY_CHARS):
        return True
    return any(word in text.lower() for word in _CURRENCY_WORDS)


def contains_urgent_word(text: str) -> bool:
    return _contains_any_word(text, _URGENT_WORDS)


def contains_prize_word(text: str) -> bool:
    return _contains_any_word(text, _PRIZE_WORDS)


def contains_call_or_reply_instruction(text: str) -> bool:
    return _contains_any_word(text, _CALL_REPLY_WORDS)


def uppercase_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    upper = sum(1 for c in alpha if c.isupper())
    return upper / len(alpha)


def exclamation_count(text: str) -> int:
    return text.count("!")


def message_length(text: str) -> int:
    return len(text)


def extract_risk_features(text: str) -> dict[str, Any]:
    """Return a flat dict of risk features suitable for CSV columns."""
    return {
        "has_url": contains_url(text),
        "has_phone": contains_phone_number(text),
        "has_currency": contains_currency(text),
        "has_urgent_word": contains_urgent_word(text),
        "has_prize_word": contains_prize_word(text),
        "has_call_instr": contains_call_or_reply_instruction(text),
        "uppercase_ratio": round(uppercase_ratio(text), 4),
        "exclamation_count": exclamation_count(text),
        "message_length": message_length(text),
    }
