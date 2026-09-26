"""
Text normalization for business entity resolution.

Design principles:
- Preserve raw fields; return NEW derived strings.
- Numbers are always preserved — they are discriminative.
- Normalization is idempotent and deterministic.
- None / NaN / empty strings all return "".
- No country-specific hard-coding for names (country is passed to callers).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

# ---------------------------------------------------------------------------
# Legal suffix removal — order matters (longest first)
# ---------------------------------------------------------------------------
_LEGAL_SUFFIXES: tuple[str, ...] = (
    # English (US / India / France)
    "private limited",
    "private ltd",
    "public limited",
    "pvt limited",
    "pvt ltd",
    "llp",
    "llc",
    "incorporated",
    "corporation",
    "limited",
    "company",
    "corp",
    "ltd",
    "inc",
    "lp",
    "plc",
    "co",
    # French
    "societe anonyme",
    "societe en commandite",
    "societe par actions simplifiee",
    "sas",
    "sarl",
    "sca",
    "snc",
    "sci",
    "eurl",
    "eirl",
    # Indian
    "pvt",
    "opc",
    "ngo",
    "trust",
    "society",
    "foundation",
    "associates",
    "enterprises",
    "solutions",
    "services",
    "group",
    "trading",
    "international",
    "global",
    "india",
    "centre",
    "center",
    "partners",
)

# Pre-compile suffix pattern for stripping at end of string.
# We only strip the suffix if it is a whole "word" at the end.
_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _LEGAL_SUFFIXES) + r")\s*$",
    re.IGNORECASE,
)

# Address abbreviation expansions (applied after lower-casing)
_ADDR_ABBREVS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bst\b"), "street"),
    (re.compile(r"\bave?\b"), "avenue"),
    (re.compile(r"\bblvd\b"), "boulevard"),
    (re.compile(r"\brd\b"), "road"),
    (re.compile(r"\bdr\b"), "drive"),
    (re.compile(r"\bln\b"), "lane"),
    (re.compile(r"\bct\b"), "court"),
    (re.compile(r"\bpl\b"), "place"),
    (re.compile(r"\bpkwy\b"), "parkway"),
    (re.compile(r"\bhwy\b"), "highway"),
    (re.compile(r"\bfwy\b"), "freeway"),
    (re.compile(r"\bsq\b"), "square"),
    (re.compile(r"\bft\b"), "fort"),
    (re.compile(r"\bmt\b"), "mount"),
    (re.compile(r"\bpts?\b"), "point"),
    (re.compile(r"\bjct\b"), "junction"),
    # US state abbreviations → full name
    (re.compile(r"\bnc\b"), "north carolina"),
    (re.compile(r"\boh\b"), "ohio"),
    (re.compile(r"\bin\b"), "indiana"),
    (re.compile(r"\btx\b"), "texas"),
    (re.compile(r"\bca\b"), "california"),
    (re.compile(r"\bfl\b"), "florida"),
    (re.compile(r"\bny\b"), "new york"),
    (re.compile(r"\bpa\b"), "pennsylvania"),
    (re.compile(r"\bga\b"), "georgia"),
    (re.compile(r"\bva\b"), "virginia"),
    (re.compile(r"\bma\b"), "massachusetts"),
    (re.compile(r"\bmd\b"), "maryland"),
    (re.compile(r"\baz\b"), "arizona"),
    (re.compile(r"\bco\b"), "colorado"),
    (re.compile(r"\bwi\b"), "wisconsin"),
    (re.compile(r"\bmn\b"), "minnesota"),
    (re.compile(r"\bmo\b"), "missouri"),
    (re.compile(r"\bwa\b"), "washington"),
    (re.compile(r"\bor\b"), "oregon"),
    (re.compile(r"\bsc\b"), "south carolina"),
    (re.compile(r"\bmi\b"), "michigan"),
    (re.compile(r"\bal\b"), "alabama"),
    (re.compile(r"\bla\b"), "louisiana"),
    (re.compile(r"\bnj\b"), "new jersey"),
    (re.compile(r"\bct\b"), "connecticut"),
    (re.compile(r"\bks\b"), "kansas"),
    (re.compile(r"\bky\b"), "kentucky"),
    (re.compile(r"\bne\b"), "nebraska"),
    (re.compile(r"\bnv\b"), "nevada"),
    (re.compile(r"\bnm\b"), "new mexico"),
    (re.compile(r"\bwv\b"), "west virginia"),
    (re.compile(r"\bms\b"), "mississippi"),
    (re.compile(r"\bar\b"), "arkansas"),
    (re.compile(r"\biowa\b"), "iowa"),
    (re.compile(r"\bok\b"), "oklahoma"),
    (re.compile(r"\but\b"), "utah"),
    (re.compile(r"\bme\b"), "maine"),
    (re.compile(r"\bnh\b"), "new hampshire"),
    (re.compile(r"\bvt\b"), "vermont"),
    (re.compile(r"\bri\b"), "rhode island"),
    (re.compile(r"\bde\b"), "delaware"),
    (re.compile(r"\bnd\b"), "north dakota"),
    (re.compile(r"\bsd\b"), "south dakota"),
    (re.compile(r"\bwy\b"), "wyoming"),
    (re.compile(r"\bmt\b"), "montana"),
    (re.compile(r"\bid\b"), "idaho"),
    (re.compile(r"\bhi\b"), "hawaii"),
    (re.compile(r"\bak\b"), "alaska"),
    # Indian state abbreviations
    (re.compile(r"\bmh\b"), "maharashtra"),
    (re.compile(r"\bwb\b"), "west bengal"),
    (re.compile(r"\bup\b"), "uttar pradesh"),
    (re.compile(r"\bmp\b"), "madhya pradesh"),
    (re.compile(r"\bka\b"), "karnataka"),
    (re.compile(r"\bkl\b"), "kerala"),
    (re.compile(r"\bap\b"), "andhra pradesh"),
    (re.compile(r"\bts\b"), "telangana"),
    (re.compile(r"\btg\b"), "telangana"),
    (re.compile(r"\bhr\b"), "haryana"),
    (re.compile(r"\bpb\b"), "punjab"),
    (re.compile(r"\brj\b"), "rajasthan"),
    (re.compile(r"\bgu\b"), "gujarat"),
    (re.compile(r"\bbr\b"), "bihar"),
    (re.compile(r"\bjh\b"), "jharkhand"),
    (re.compile(r"\bor\b"), "odisha"),
    (re.compile(r"\bcg\b"), "chhattisgarh"),
    (re.compile(r"\bua\b"), "uttarakhand"),
    (re.compile(r"\bhp\b"), "himachal pradesh"),
    (re.compile(r"\bsk\b"), "sikkim"),
    (re.compile(r"\bmz\b"), "mizoram"),
    (re.compile(r"\bmn\b"), "manipur"),
    (re.compile(r"\bnl\b"), "nagaland"),
    (re.compile(r"\bar\b"), "arunachal pradesh"),
    (re.compile(r"\bml\b"), "meghalaya"),
    (re.compile(r"\btr\b"), "tripura"),
    (re.compile(r"\bas\b"), "assam"),
    (re.compile(r"\bgoa\b"), "goa"),
    (re.compile(r"\bdl\b"), "delhi"),
    (re.compile(r"\bpb\b"), "puducherry"),
    (re.compile(r"\bchandigarh\b"), "chandigarh"),
    (re.compile(r"\bjk\b"), "jammu kashmir"),
    (re.compile(r"\bla\b"), "ladakh"),
    # Unit/Suite abbreviations
    (re.compile(r"\bapt\b"), "apartment"),
    (re.compile(r"\bste\b"), "suite"),
    (re.compile(r"\bfl\b"), "floor"),
    (re.compile(r"\bunit\b"), "unit"),
]

_NUMBER_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
# Strips leading zeros from numeric tokens like "0337" → "337"
_LEADING_ZERO_RE = re.compile(r"\b0+(\d)")


def _to_str(x) -> str:
    """Safely coerce value to string, returning '' for None/NaN."""
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("nan", "none", "null", ""):
        return ""
    return s


def _unicode_normalize(text: str) -> str:
    """NFKC normalize + strip combining diacritics from latin characters.

    We use NFKD to decompose, then re-encode only ASCII-compatible characters,
    but keep non-latin Unicode intact (Hindi, Bengali, French accents handled
    separately).  For this task we keep the full NFKC form so Hindi/Bengali
    characters survive — we only want to collapse ligatures and half-width forms.
    """
    return unicodedata.normalize("NFKC", text)


def normalize_name(raw) -> str:
    """Normalize a business name.

    Steps:
    1. Coerce to string
    2. NFKC unicode normalize
    3. Lowercase
    4. Replace & with ' and '
    5. Collapse punctuation/special chars to space (preserve numbers)
    6. Collapse whitespace

    Non-ASCII kept so that Devanagari / Bengali names are not destroyed.
    """
    s = _to_str(raw)
    if not s:
        return ""
    s = _unicode_normalize(s)
    s = s.lower()
    s = s.replace("&", " and ")
    # Replace anything that is not a letter, digit, or whitespace with space.
    # \w in Python includes Unicode letters, which preserves Hindi/Bengali.
    s = re.sub(r"[^\w\s]", " ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def normalize_name_stripped(raw) -> str:
    """Normalize name AND strip trailing legal suffixes (iteratively).

    Useful as a blocking key — two entities may differ only in suffix.
    """
    s = normalize_name(raw)
    if not s:
        return ""
    # Iteratively strip suffixes until stable
    prev = None
    while prev != s:
        prev = s
        s = _SUFFIX_PATTERN.sub("", s).strip()
    return s


# Pre-compile a single regex for all address abbreviations
_ABBREV_MAP = {k: v for pattern, v in _ADDR_ABBREVS for k in pattern.pattern.replace(r"\b", "").split("|")}
# Because the patterns might contain regex components, let's extract the actual keys safely.
# Wait, _ADDR_ABBREVS has keys like r"\bst\b". Let's build the map directly.
_ABBREV_DICT = {
    "st": "street", "ave": "avenue", "blvd": "boulevard", "rd": "road", 
    "dr": "drive", "ln": "lane", "ct": "court", "pl": "place", "pkwy": "parkway", 
    "hwy": "highway", "fwy": "freeway", "sq": "square", "ft": "fort", "mt": "mount", 
    "pt": "point", "pts": "point", "jct": "junction",
    "nc": "north carolina", "oh": "ohio", "in": "indiana", "tx": "texas", 
    "ca": "california", "fl": "florida", "ny": "new york", "pa": "pennsylvania", 
    "ga": "georgia", "va": "virginia", "ma": "massachusetts", "md": "maryland", 
    "az": "arizona", "co": "colorado", "wi": "wisconsin", "mn": "minnesota", 
    "mo": "missouri", "wa": "washington", "or": "oregon", "sc": "south carolina", 
    "mi": "michigan", "al": "alabama", "la": "louisiana", "nj": "new jersey", 
    "ks": "kansas", "ky": "kentucky", "ne": "nebraska", "nv": "nevada", 
    "nm": "new mexico", "wv": "west virginia", "ms": "mississippi", "ar": "arkansas", 
    "iowa": "iowa", "ok": "oklahoma", "ut": "utah", "me": "maine", "nh": "new hampshire", 
    "vt": "vermont", "ri": "rhode island", "de": "delaware", "nd": "north dakota", 
    "sd": "south dakota", "wy": "wyoming", "id": "idaho", "hi": "hawaii", "ak": "alaska",
    "mh": "maharashtra", "wb": "west bengal", "up": "uttar pradesh", "mp": "madhya pradesh", 
    "ka": "karnataka", "kl": "kerala", "ap": "andhra pradesh", "ts": "telangana", 
    "tg": "telangana", "hr": "haryana", "pb": "punjab", "rj": "rajasthan", 
    "gu": "gujarat", "br": "bihar", "jh": "jharkhand", "cg": "chhattisgarh", 
    "ua": "uttarakhand", "hp": "himachal pradesh", "sk": "sikkim", "mz": "mizoram", 
    "mn": "manipur", "nl": "nagaland", "ml": "meghalaya", "tr": "tripura", "as": "assam", 
    "goa": "goa", "dl": "delhi", "chandigarh": "chandigarh", "jk": "jammu kashmir", "la": "ladakh",
    "apt": "apartment", "ste": "suite", "fl": "floor", "unit": "unit"
}
_ADDR_ABBREV_PATTERN = re.compile(r"\b(" + "|".join(_ABBREV_DICT.keys()) + r")\b")

def _replace_abbrev(match):
    return _ABBREV_DICT.get(match.group(1), match.group(1))

def normalize_address(raw) -> str:
    """Normalize a business address."""
    s = _to_str(raw)
    if not s:
        return ""
    s = _unicode_normalize(s)
    s = s.lower()
    s = s.replace("&", " and ")
    s = _LEADING_ZERO_RE.sub(r"\1", s)
    # Expand abbreviations using single regex
    s = _ADDR_ABBREV_PATTERN.sub(_replace_abbrev, s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def extract_numbers(text) -> List[str]:
    """Return sorted list of numeric tokens from a raw text field.

    Used as a hard feature — entities sharing street numbers / PIN codes are
    much more likely to match.
    """
    s = _to_str(text)
    if not s:
        return []
    # Strip leading zeros before extraction
    s = _LEADING_ZERO_RE.sub(r"\1", s)
    nums = _NUMBER_RE.findall(s)
    return sorted(set(nums))


def tokenize(text: str) -> List[str]:
    """Split normalized text into sorted, deduplicated tokens (≥2 chars)."""
    if not text:
        return []
    return sorted(set(t for t in text.split() if len(t) >= 2))


def normalize_country(raw) -> str:
    """Normalize country to a canonical lowercase string."""
    s = _to_str(raw).lower().strip()
    # Canonicalize common variants
    _aliases = {
        "united states": "us",
        "usa": "us",
        "u.s.a.": "us",
        "u.s.": "us",
        "america": "us",
        "india": "india",
        "bharat": "india",
        "france": "france",
        "fr": "france",
        "french republic": "france",
    }
    return _aliases.get(s, s)


def build_name_key(raw_name: str) -> str:
    """Composite key for exact-name blocking: normalized name."""
    return normalize_name(raw_name)


def build_name_stripped_key(raw_name: str) -> str:
    """Composite key for suffix-stripped name blocking."""
    return normalize_name_stripped(raw_name)


def build_address_key(raw_address: str) -> str:
    """Composite key for exact-address blocking: normalized address."""
    return normalize_address(raw_address)


def build_name_address_key(raw_name: str, raw_address: str) -> str:
    """Composite key: normalized name + '||' + normalized address."""
    n = normalize_name(raw_name)
    a = normalize_address(raw_address)
    if not n or not a:
        return ""
    return f"{n}||{a}"
