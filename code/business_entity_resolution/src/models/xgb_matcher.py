"""Training and threshold selection for the precision-weighted competition metric."""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
from xgboost import XGBClassifier
from features.pair_features import FEATURE_NAMES


def macro_f_beta(truth: dict[str, set[str]], predicted: dict[str, set[str]], beta: float = .5) -> float:
    scores = []
    b2 = beta * beta
    for entity, actual in truth.items():
        guess = predicted.get(entity, set())
        if not actual and not guess: scores.append(1.0); continue
        if not actual or not guess: scores.append(0.0); continue
        hit = len(actual & guess); p, r = hit / len(guess), hit / len(actual)
        scores.append((1 + b2) * p * r / (b2 * p + r) if p + r else 0.0)
    return float(np.mean(scores))


def fit(rows: list[dict[str, float]], labels: list[int], groups: list[str], target_ids: list[str], truth: dict[str, set[str]], model_path: Path) -> tuple[object, float, float]:
    x = np.asarray([[r[f] for f in FEATURE_NAMES] for r in rows], dtype=np.float32)
    model = XGBClassifier(n_estimators=350, max_depth=6, learning_rate=.07, subsample=.8, colsample_bytree=.85,
                          objective="binary:logistic", eval_metric="logloss", n_jobs=-1, random_state=2026)
    model.fit(x, np.asarray(labels))
    probabilities = model.predict_proba(x)[:, 1]
    best = (-1.0, .5)
    for threshold in np.linspace(.30, .995, 140):
        predicted: dict[str, set[str]] = {}
        for group, target, probability in zip(groups, target_ids, probabilities):
            if probability >= threshold: predicted.setdefault(group, set()).add(target)
        score = macro_f_beta(truth, predicted)
        if score > best[0]: best = (score, float(threshold))
    joblib.dump({"model": model, "features": FEATURE_NAMES, "threshold": best[1]}, model_path)
    return model, best[1], best[0]
