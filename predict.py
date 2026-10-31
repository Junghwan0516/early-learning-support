"""Rank a feature table using the frozen validation-selected method."""
from pathlib import Path
import argparse
import json
import math
import urllib.request
import numpy as np
import pandas as pd
import els

def validate_inputs(frame):
    missing=set(els.FEATURES)-set(frame.columns)
    if missing: raise ValueError(f'Missing feature columns: {sorted(missing)}')
    x=frame[els.FEATURES].apply(pd.to_numeric,errors='raise')
    if x.empty or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Inputs must be nonempty and contain finite numeric features.')
    if not x[els.FEATURES[:4]].ge(0).all().all(): raise ValueError('Weekly clicks must be nonnegative.')
    if not x.active_days.between(0,28).all() or not x.days_since_activity.between(0,28).all():
        raise ValueError('Active days and recency must lie between 0 and 28.')
    for column in els.FEATURES:
        if not np.equal(x[column],np.floor(x[column])).all():
            raise ValueError('Prepared click and day features must be integer-valued.')
    if not np.allclose(x[els.FEATURES[:4]].sum(axis=1),x.total_clicks):
        raise ValueError('Total clicks must equal the sum of weekly clicks.')
    if not np.allclose(x.clicks_w4-x.clicks_w3,x.last_week_change):
        raise ValueError('Last-week change must equal week 4 minus week 3.')
    no_activity=x.total_clicks.eq(0)
    if not (x.loc[no_activity,'active_days'].eq(0).all() and x.loc[no_activity,'days_since_activity'].eq(28).all()):
        raise ValueError('No-activity inputs require active_days=0 and days_since_activity=28.')
    if not (x.loc[~no_activity,'active_days'].gt(0).all() and x.loc[~no_activity,'days_since_activity'].lt(28).all()):
        raise ValueError('Positive activity requires active days and recency below 28.')
    if not x.active_days.le(x.total_clicks).all():
        raise ValueError('Every active day requires at least one recorded click.')
    if not x.active_days.le(28-x.days_since_activity).all():
        raise ValueError('Active-day count is incompatible with the last active day.')
    return x

def rank_learners(frame, fraction=.1):
    x=validate_inputs(frame)
    if not 0<fraction<=1: raise ValueError('Review fraction must be in (0,1].')
    path=Path('reports/frozen_selection.json')
    if not path.exists():
        with urllib.request.urlopen(els.REPO_RAW+str(path),timeout=60) as response:
            path.parent.mkdir(exist_ok=True);path.write_bytes(response.read())
    selection=json.loads(path.read_text())
    data=els.load_data(); parts,_=els.split_data(data)
    if selection['family']=='Low-activity rule':scores=-x.total_clicks.to_numpy()
    else:
        model=els.model_candidates()[(selection['family'],selection['config'])]
        model.fit(parts['train'][els.FEATURES],parts['train'].target)
        scores=els.scores_for(model,x)
    # Same seed defines an outcome-free tie priority in every incoming batch.
    priority=np.random.default_rng(2026).permutation(len(x))
    order=priority[np.argsort(-scores[priority],kind='stable')]
    ranks=np.empty(len(x),int);ranks[order]=np.arange(1,len(x)+1)
    result=pd.DataFrame({'row_number':np.arange(1,len(x)+1),'ranking_score':scores,'review_rank':ranks,
                         'selected_for_review':ranks<=math.ceil(fraction*len(x))})
    return result.sort_values('review_rank')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',default='outputs/ranked_learners.csv')
    parser.add_argument('--fraction',type=float,default=.1)
    args=parser.parse_args()
    output=rank_learners(pd.read_csv(args.input),args.fraction)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    output.to_csv(path,index=False)
    print(output.to_string(index=False))
    print('Research ranking only. Scores are not calibrated individual withdrawal probabilities.')
