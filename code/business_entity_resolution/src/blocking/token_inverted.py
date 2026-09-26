"""
Token inverted index blocker.
Builds an inverted index of name (or address) tokens, partitioned by country.
Queries candidates that share at least N tokens.
"""
import pandas as pd
from typing import Dict, Set, List
from collections import defaultdict, Counter

from .base import Blocker
from ..preprocessing.normalize import tokenize, normalize_name, normalize_country

class TokenInvertedBlocker(Blocker):
    def __init__(self, min_overlap: int = 2, field: str = "business_name", use_country: bool = True):
        """
        min_overlap: Minimum number of shared tokens to be considered a candidate.
        field: "business_name" or "business_address"
        """
        self.min_overlap = min_overlap
        self.field = field
        self.use_country = use_country
        
        # Structure: country -> token -> list of entity_ids
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        
        # Store token counts per entity for fast Jaccard/overlap checking if needed
        # (For this simple version we'll just count matching tokens during retrieval)
        
    def build_index(self, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> None:
        self.index.clear()
        
        for df in (s2_df, s3_df):
            for row in df.itertuples(index=False):
                country_key = normalize_country(row.country) if self.use_country else "GLOBAL"
                val = getattr(row, self.field, "")
                # use normalize_name's string processing before tokenizing (handles punctuation)
                tokens = tokenize(normalize_name(val)) 
                
                for t in tokens:
                    self.index[country_key][t].append(row.entity_id)
                    
    def generate_candidates(self, s1_record: dict) -> Set[str]:
        country_key = normalize_country(s1_record.get('country')) if self.use_country else "GLOBAL"
        val = s1_record.get(self.field, "")
        tokens = tokenize(normalize_name(val))
        
        if not tokens:
            return set()
            
        candidate_counts = Counter()
        idx_for_country = self.index.get(country_key, {})
        
        # Filter tokens by frequency to avoid stop-word explosion
        # If a token has > 100,000 matches, skip it (unless it's the only token)
        MAX_DF = 100000
        valid_tokens = [t for t in tokens if t in idx_for_country and len(idx_for_country[t]) <= MAX_DF]
        
        # Fallback: if all tokens were too common, just use the least common one
        if not valid_tokens and tokens:
            # Sort by frequency
            sorted_tokens = sorted([t for t in tokens if t in idx_for_country], key=lambda x: len(idx_for_country[x]))
            if sorted_tokens:
                valid_tokens = [sorted_tokens[0]]
                
        for t in valid_tokens:
            for eid in idx_for_country[t]:
                candidate_counts[eid] += 1
                    
        candidates = {
            eid for eid, count in candidate_counts.items()
            if count >= self.min_overlap
        }
        
        # If the query itself has fewer valid tokens than min_overlap, 
        # allow exact token match (overlap == len(valid_tokens))
        if len(valid_tokens) < self.min_overlap and valid_tokens:
            candidates.update({
                eid for eid, count in candidate_counts.items()
                if count == len(valid_tokens)
            })
            
        return candidates
