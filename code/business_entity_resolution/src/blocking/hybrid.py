"""
Hybrid Blocker.
Combines candidates from multiple underlying blockers (Union).
"""
import pandas as pd
from typing import Dict, Set, List

from .base import Blocker

class HybridBlocker(Blocker):
    def __init__(self, blockers: List[Blocker]):
        self.blockers = blockers
        
    def build_index(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> None:
        for b in self.blockers:
            b.build_index(s2_df, s3_df)
            
    def generate_candidates(self, s1_record: dict) -> Set[str]:
        candidates = set()
        for b in self.blockers:
            candidates.update(b.generate_candidates(s1_record))
        return candidates
