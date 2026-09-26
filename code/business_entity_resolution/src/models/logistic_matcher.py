"""A dependency-free calibrated baseline matcher.

XGBoost is an optional upgrade once it wins the same held-out evaluation; this
model keeps the submitted pipeline runnable on a plain Python installation.
"""
from __future__ import annotations
import json, math, random
from pathlib import Path
from features.pair_features import FEATURE_NAMES

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, x))))

def _macro(truth, pred):
    values=[]
    for s1, actual in truth.items():
        guess=pred.get(s1,set())
        if not actual and not guess: values.append(1.0); continue
        if not actual or not guess: values.append(0.0); continue
        hit=len(actual & guess); precision=hit/len(guess); recall=hit/len(actual)
        values.append(1.25*precision*recall/(.25*precision+recall) if precision+recall else 0.0)
    return sum(values)/len(values)

def fit(rows, labels, groups, target_ids, truth, model_path: Path):
    matrix=[[float(row[f]) for f in FEATURE_NAMES] for row in rows]
    means=[sum(col)/len(col) for col in zip(*matrix)]
    scales=[max(1e-6, (sum((x-m)**2 for x in col)/len(col))**.5) for col,m in zip(zip(*matrix),means)]
    matrix=[[(x-m)/s for x,m,s in zip(row,means,scales)] for row in matrix]
    weights=[0.0]*len(FEATURE_NAMES); bias=0.0; order=list(range(len(matrix))); rng=random.Random(2026)
    positives=sum(labels); negatives=len(labels)-positives; pos_weight=negatives/max(1,positives)
    for epoch in range(90):
        rng.shuffle(order); rate=.10/(1+epoch/25)
        for i in order:
            score=bias+sum(w*x for w,x in zip(weights,matrix[i])); error=(_sigmoid(score)-labels[i])*(pos_weight if labels[i] else 1.0)
            for j,x in enumerate(matrix[i]): weights[j]-=rate*error*x
            bias-=rate*error
    probabilities=[_sigmoid(bias+sum(w*x for w,x in zip(weights,row))) for row in matrix]
    best=(-1.0,.5)
    for step in range(300,996,5):
        threshold=step/1000; pred={}
        for group,target,p in zip(groups,target_ids,probabilities):
            if p>=threshold: pred.setdefault(group,set()).add(target)
        value=_macro(truth,pred)
        if value>best[0]: best=(value,threshold)
    artifact={"type":"logistic_sgd","features":FEATURE_NAMES,"mean":means,"scale":scales,"weights":weights,"bias":bias,"threshold":best[1]}
    model_path.write_text(json.dumps(artifact,indent=2)); return best[1],best[0]

def predict(artifact, rows):
    result=[]
    for values in rows:
        xs=[(values[f]-m)/s for f,m,s in zip(artifact["features"],artifact["mean"],artifact["scale"])]
        result.append(_sigmoid(artifact["bias"]+sum(w*x for w,x in zip(artifact["weights"],xs))))
    return result
