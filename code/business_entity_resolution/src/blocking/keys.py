"""Candidate keys.  Their union is the exact input to the ML scorer."""
from __future__ import annotations

from preprocessing.normalization import address, country, informative_tokens, name, numbers


def normalized(row: dict[str, str]) -> dict[str, str]:
    return {"name": name(row.get("business_name", "")), "address": address(row.get("business_address", "")), "country": country(row.get("country", ""))}


def keys(row: dict[str, str]) -> list[tuple[str, str]]:
    """High-precision, language-agnostic keys; no country whitelist is used."""
    n = normalized(row); c, nm, ad = n["country"], n["name"], n["address"]
    result: list[tuple[str, str]] = []
    if nm: result.append(("exact_name", f"{c}|{nm}"))
    if ad: result.append(("exact_address", f"{c}|{ad}"))
    if nm and ad: result.append(("exact_pair", f"{c}|{nm}|{ad}"))
    compact = nm.replace(" ", "")
    if len(compact) >= 5: result.append(("name_prefix", f"{c}|{compact[:7]}"))
    for token in informative_tokens(nm)[:4]: result.append(("name_token", f"{c}|{token}"))
    for num in numbers(ad)[:2]: result.append(("address_number", f"{c}|{num}"))
    return result
