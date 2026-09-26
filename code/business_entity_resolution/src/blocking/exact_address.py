"""
Exact address blocker.
Uses the normalized address as a hard blocking key.
Country is used as a hard filter.
"""
import pandas as pd
from typing import Dict, Set, List
from collections import defaultdict

from .base import Blocker
from ..preprocessing.normalize import normalize_address, normalize_country

class ExactAddressBlocker(Blocker):
    def __init__(self, use_country: bool = True):
        self.use_country = use_country
        self.index: Dict[str, List[str]] = defaultdict(list)
        
    def _make_key(self, raw_addr: str, raw_country: str) -> str:
        addr_key = normalize_address(raw_addr)
        if not addr_key:
            return ""
        if self.use_country:
            c = normalize_country(raw_country)
            return f"{c}||{addr_key}"
        return addr_key

    def build_index(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> None:
        self.index.clear()
        
        for df in (s2_df, s3_df):
            for row in df.itertuples(index=False):
                key = self._make_key(row.business_address, row.country)
                if key:
                    self.index[key].append(row.entity_id)
                    
    def generate_candidates(self, s1_record: dict) -> Set[str]:
        key = self._make_key(s1_record.get('business_address'), s1_record.get('country'))
        if not key:
            return set()
        return set(self.index.get(key, []))
