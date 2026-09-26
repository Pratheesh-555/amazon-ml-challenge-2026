"""Locale-safe normalization used only to compare the supplied records."""
from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")
_LEGAL = re.compile(r"\b(incorporated|inc|corporation|corp|limited|ltd|llc|llp|private|pvt|company|co)\b", re.I)
_ADDRESS = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "no": "number", "hwy": "highway", "nagar": "nagar",
}
_COUNTRY = {"us": "us", "usa": "us", "united states": "us", "india": "india", "in": "india"}


def text(value: str) -> str:
    """NFKC + casefold without ASCII transliteration (important for Indian text)."""
    value = unicodedata.normalize("NFKC", value or "").casefold().replace("&", " and ")
    return _SPACE.sub(" ", _PUNCT.sub(" ", value)).strip()


def name(value: str) -> str:
    return _SPACE.sub(" ", _LEGAL.sub(" ", text(value))).strip()


def address(value: str) -> str:
    tokens = [_ADDRESS.get(t, t) for t in text(value).split()]
    return " ".join(tokens)


def country(value: str) -> str:
    cleaned = text(value)
    return _COUNTRY.get(cleaned, cleaned)  # unknown countries, including France, pass through


def numbers(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+[\w-]*", value))


def informative_tokens(value: str) -> tuple[str, ...]:
    return tuple(t for t in value.split() if len(t) >= 4 and not t.isdigit())
