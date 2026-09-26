"""
Base blocker interface.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Set

class Blocker(ABC):
    """Abstract base class for candidate generation strategies."""
    
    @abstractmethod
    def build_index(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> None:
        """Build internal indexes from Source 2 and Source 3 DataFrames."""
        pass
        
    @abstractmethod
    def generate_candidates(self, s1_record: dict) -> Set[str]:
        """Generate a set of candidate entity_ids from Source 2 and Source 3."""
        pass
