import pandas as pd
import pytest
from src.blocking.exact_name import ExactNameBlocker
from src.blocking.hybrid import HybridBlocker

@pytest.fixture
def sample_data():
    s2 = pd.DataFrame([
        {"entity_id": "S2-1", "business_name": "Acme Inc", "business_address": "123 Main", "country": "US"},
        {"entity_id": "S2-2", "business_name": "Acme Inc", "business_address": "456 Oak", "country": "India"},
    ])
    s3 = pd.DataFrame([
        {"entity_id": "S3-1", "business_name": "Acme Inc", "business_address": "123 Main", "country": "US"},
    ])
    return s2, s3

def test_exact_name_blocker(sample_data):
    s2, s3 = sample_data
    blocker = ExactNameBlocker(use_country=True)
    blocker.build_index(s2, s3)
    
    # Matching name and country
    s1_us = {"entity_id": "S1-1", "business_name": "Acme Inc", "country": "US"}
    candidates = blocker.generate_candidates(s1_us)
    assert candidates == {"S2-1", "S3-1"}
    
    # Matching name but different country
    s1_in = {"entity_id": "S1-2", "business_name": "Acme Inc", "country": "India"}
    candidates = blocker.generate_candidates(s1_in)
    assert candidates == {"S2-2"}
    
    # No match
    s1_none = {"entity_id": "S1-3", "business_name": "Globex", "country": "US"}
    assert blocker.generate_candidates(s1_none) == set()
