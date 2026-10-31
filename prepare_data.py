"""Day-28 OULAD withdrawal research prototype. No model runs on import."""
from pathlib import Path
import hashlib
import math
import urllib.request
import zipfile
import numpy as np
import pandas as pd

URL = 'https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip'
ARCHIVE_SHA256 = 'f2ed1902616c1fe8d2824d872c0b7d2d72be435bf0124d077044fe4be2c6d3e4'
MODULE = 'BBB'
PERIODS = ('2013B', '2013J', '2014J')
CUTOFF = 28  # prediction at start of relative day 28; observe days 0..27
FEATURES = ['clicks_w1', 'clicks_w2', 'clicks_w3', 'clicks_w4',
            'active_days', 'days_since_activity', 'last_week_change', 'total_clicks']
KEYS = ['code_module', 'code_presentation', 'id_student']

def download(path='data/oulad.zip'):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix('.partial')
        with urllib.request.urlopen(URL, timeout=90) as response, temp.open('wb') as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        if not zipfile.is_zipfile(temp):
            temp.unlink()
            raise ValueError('Downloaded content is not a ZIP file. Use the official UCI download.')
        temp.rename(path)
    if not zipfile.is_zipfile(path):
        raise ValueError('Local archive is invalid; remove it and download again.')
    if hashlib.sha256(path.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError('Archive checksum differs from the documented source. Investigate before use.')
    return path

def metadata(path):
    with zipfile.ZipFile(path) as z:
        info = pd.read_csv(z.open('studentInfo.csv'), na_values=['?'])
        reg = pd.read_csv(z.open('studentRegistration.csv'), na_values=['?'])
    if info.duplicated(KEYS).any() or reg.duplicated(KEYS).any():
        raise ValueError('Duplicate student-course-presentation key')
    return info, reg

def eligible_cohort(info, reg):
    info = info[(info.code_module == MODULE) & info.code_presentation.isin(PERIODS)].copy()
    base = info.merge(reg, on=KEYS, how='left', validate='one_to_one', indicator=True)
    if (base['_merge'] != 'both').any():
        raise ValueError('Missing registration table row')
    base = base.drop(columns='_merge')
    # Withdrawn rows with unknown outcome date cannot define our at-risk cohort.
    unknown_withdrawal = base.final_result.eq('Withdrawn') & base.date_unregistration.isna()
    keep = (base.date_registration.notna() & base.date_registration.le(0)
            & (base.date_unregistration.isna() | base.date_unregistration.ge(CUTOFF))
            & ~unknown_withdrawal)
    cohort = base.loc[keep].copy()
    expected = {'Withdrawn', 'Fail', 'Pass', 'Distinction'}
    if not set(cohort.final_result).issubset(expected):
        raise ValueError('Unexpected final_result label')
    contradictory = (~cohort.final_result.eq('Withdrawn')) & cohort.date_unregistration.notna()
    # Exclude inconsistent outcome records rather than silently relabeling them.
    cohort = cohort.loc[~contradictory].copy()
    cohort['target'] = cohort.final_result.eq('Withdrawn').astype('int8')
    # id_student is linkage/grouping only. Outcome and registration dates never enter X.
    return cohort[KEYS + ['target']]

def aggregate_days(daily):
    """Daily events must have presentation, student, date and sum_click columns."""
    daily = daily[daily.date.ge(0) & daily.date.lt(CUTOFF)].copy()
    daily = daily.groupby(['code_presentation', 'id_student', 'date'], as_index=False).sum_click.sum()
    daily = daily[daily.sum_click > 0]
    if daily.empty:
        return pd.DataFrame(columns=['code_presentation', 'id_student'] + FEATURES)
    daily['week'] = daily.date // 7 + 1
    pivot = daily.pivot_table(index=['code_presentation', 'id_student'], columns='week',
                             values='sum_click', aggfunc='sum', fill_value=0)
    pivot = pivot.reindex(columns=[1, 2, 3, 4], fill_value=0)
    pivot.columns = ['clicks_w1', 'clicks_w2', 'clicks_w3', 'clicks_w4']
    summary = daily.groupby(['code_presentation', 'id_student']).agg(
        active_days=('date', 'nunique'), last_activity=('date', 'max'),
        total_clicks=('sum_click', 'sum'))
    summary['days_since_activity'] = (CUTOFF - 1) - summary.pop('last_activity')
    result = pivot.join(summary)
    result['last_week_change'] = result.clicks_w4 - result.clicks_w3
    return result.reset_index()

def make_features(path):
    info, reg = metadata(path)
    cohort = eligible_cohort(info, reg)
    parts = []
    # Chunk the 400+ MB CSV rather than loading the full activity table into RAM.
    with zipfile.ZipFile(path) as z:
        for chunk in pd.read_csv(z.open('studentVle.csv'), chunksize=250_000,
                                 usecols=KEYS + ['date', 'sum_click']):
            x = chunk[(chunk.code_module == MODULE) & chunk.code_presentation.isin(PERIODS)
                      & chunk.date.ge(0) & chunk.date.lt(CUTOFF)]
            if not x.empty:
                parts.append(x.groupby(['code_presentation', 'id_student', 'date'],
                                       as_index=False).sum_click.sum())
    daily = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=['code_presentation', 'id_student', 'date', 'sum_click'])
    observed = aggregate_days(daily)
    result = cohort.merge(observed, on=['code_presentation', 'id_student'],
                          how='left', validate='one_to_one')
    result[FEATURES] = result[FEATURES].apply(pd.to_numeric)
    result['days_since_activity'] = result.days_since_activity.fillna(CUTOFF)
    result[FEATURES] = result[FEATURES].fillna(0)
    if result.duplicated(KEYS).any():
        raise ValueError('Duplicate feature row')
    return result[KEYS + FEATURES + ['target']]

def temporal_splits(frame):
    """Never put a repeated learner in both an earlier and later split."""
    parts, seen, audit = {}, set(), []
    for label, period in zip(('train', 'validation', 'test'), PERIODS):
        raw = frame[frame.code_presentation == period].copy()
        clean = raw[~raw.id_student.isin(seen)].copy()
        seen.update(raw.id_student)
        if clean.empty or clean.target.nunique() != 2:
            raise ValueError(f'{period}: need nonempty data with both outcomes')
        parts[label] = clean
        audit.append({'split': label, 'presentation': period, 'eligible_before_dedup': len(raw),
                      'excluded_repeat_learners': len(raw)-len(clean), 'rows': len(clean),
                      'withdrawn': int(clean.target.sum()), 'withdrawal_rate': float(clean.target.mean())})
    return parts, pd.DataFrame(audit)

def rebuild(archive='data/raw/oulad.zip', output='data/features_day28.csv'):
    """Rebuild features and retain a record of the source checksum and split audit."""
    import json
    source = download(archive)
    frame = make_features(source)
    parts, audit = temporal_splits(frame)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    audit.to_csv(output.parent / 'split_audit.csv', index=False)
    return frame, audit

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='data/raw/oulad.zip')
    parser.add_argument('--output', default='data/features_day28.csv')
    args = parser.parse_args()
    frame, audit = rebuild(args.archive, args.output)
    print(audit.to_string(index=False))

