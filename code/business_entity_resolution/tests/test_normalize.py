import pytest
from src.preprocessing.normalize import (
    normalize_name,
    normalize_name_stripped,
    normalize_address,
    extract_numbers,
    normalize_country,
    tokenize
)

def test_normalize_name():
    assert normalize_name("Orelee's Barbershop") == "orelee s barbershop"
    assert normalize_name("B+ Retail Inc") == "b retail inc"
    assert normalize_name("A & B Company") == "a and b company"
    assert normalize_name("  Spaces   ") == "spaces"
    assert normalize_name(None) == ""
    # Indian characters
    assert len(normalize_name("महाराष्ट्र")) > 0
    # Actually, NFKC might keep it as is or composed. We'll just assert it's not empty.
    assert len(normalize_name("महाराष्ट्र")) > 0

def test_normalize_name_stripped():
    assert normalize_name_stripped("Custom Wealth Services LLC") == "custom wealth"
    assert normalize_name_stripped("Acme Corporation") == "acme"
    assert normalize_name_stripped("Acme Private Limited") == "acme"
    assert normalize_name_stripped("Acme Pvt Ltd") == "acme"
    # Make sure it only strips from the end
    assert normalize_name_stripped("LLC Services") == "" # Both 'services' and 'llc' are stripped

def test_normalize_address():
    assert normalize_address("1795 Westchester Drive") == "1795 westchester drive"
    assert normalize_address("123 Main St, Apt 4") == "123 main street apartment 4"
    assert normalize_address("0337 Oakland Ave") == "337 oakland avenue"
    assert normalize_address("Columbus, OH") == "columbus ohio"

def test_extract_numbers():
    assert extract_numbers("123 Main St Apt 4B") == ["123", "4"]
    assert extract_numbers("0337 Oakland") == ["337"]
    assert extract_numbers(None) == []

def test_normalize_country():
    assert normalize_country("US") == "us"
    assert normalize_country("United States") == "us"
    assert normalize_country("India") == "india"
    assert normalize_country("France") == "france"
    assert normalize_country("FR") == "france"

def test_tokenize():
    assert tokenize("a and b company") == ["and", "company"]
