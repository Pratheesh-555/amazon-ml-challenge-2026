"""
Exact name blocker.
Uses the normalized name as a hard blocking key.
Since cross-country matches do not exist, country is also used as a hard filter.
"""
import pandas as pd
from typing import Dict, Set, List
from collections import defaultdict

from .base import Blocker
from ..preprocessing.normalize import normalize_name, normalize_country

class ExactNameBlocker(Blocker):
    def __init__(self, use_country: bool = True):
        self.use_country = use_country
        # index: key -> list of entity_ids
        self.index: Dict[str, List[str]] = defaultdict(list)
        
    def _make_key(self, raw_name: str, raw_country: str) -> str:
        name_key = normalize_name(raw_name)
        if not name_key:
            return ""
        if self.use_country:
            c = normalize_country(raw_country)
            return f"{c}||{name_key}"
        return name_key

    def build_index(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> None:
        self.index.clear()
        
        for df in (s2_df, s3_df):
            for row in df.itertuples(index=False):
                key = self._make_key(row.business_name, row.country)
                if key:
                    self.index[key].append(row.entity_id)
                    
    def generate_candidates(self, s1_record: dict) -> Set[str]:
        key = self._make_key(s1_record.get('business_name'), s1_record.get('country'))
        if not key:
            return set()
        return set(self.index.get(key, []))
