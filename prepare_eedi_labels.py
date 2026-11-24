"""Reconstruct Eedi evaluation labels locally from the provider's data.zip (no fitting)."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()

def prepare(source, output):
    output.mkdir(parents=True, exist_ok=True)
    expected = json.loads((ROOT / 'eedi_local_checksums.json').read_text())
    with zipfile.ZipFile(source) as z:
        with z.open('data/train_data/train_task_1_2.csv') as f:
            df = pd.read_csv(f, usecols=['UserId', 'QuestionId', 'IsCorrect'],
                             dtype={'UserId': 'int32', 'QuestionId': 'int32', 'IsCorrect': 'uint8'})
        with z.open('data/metadata/question_metadata_task_1_2.csv') as f:
            questions = pd.read_csv(f)
        with z.open('data/metadata/subject_metadata.csv') as f:
            subjects = pd.read_csv(f)
    if df.duplicated(['UserId', 'QuestionId']).any() or not df.IsCorrect.isin([0, 1]).all():
        raise ValueError('Unexpected source records; use the original Tasks 1 and 2 training data.')
    users = np.sort(df.UserId.unique()); items = np.sort(df.QuestionId.unique())
    df = df.sort_values(['UserId', 'QuestionId']).reset_index(drop=True)
    u = np.searchsorted(users, df.UserId); i = np.searchsorted(items, df.QuestionId)
    y = df.IsCorrect.to_numpy()
    counts = np.bincount(u, minlength=len(users)); eligible = np.flatnonzero(counts >= 80)
    order = np.argsort(u, kind='stable'); offset = np.r_[0, counts.cumsum()]
    levels = dict(zip(subjects.SubjectId, subjects.Level))
    parents = dict(zip(subjects.SubjectId, subjects.ParentId))
    def category(raw):
        found = set()
        for k in ast.literal_eval(raw):
            seen = set()
            while k in levels and levels[k] > 1 and k not in seen:
                seen.add(k); k = parents[k]
            if k in levels and levels[k] == 1:
                found.add(int(k))
        return {32: 0, 49: 1, 71: 2, 101: 3}.get(next(iter(found)), -1) if len(found) == 1 else -1
    labels = questions.set_index('QuestionId').loc[items].SubjectId.map(category).to_numpy(dtype=np.int64)
    def save(name, arrays):
        if set(arrays) != set(expected[name]):
            raise ValueError('Unexpected fields: ' + name)
        for key, value in arrays.items():
            if digest(value) != expected[name][key]:
                raise ValueError(f'Source/split mismatch: {name}/{key}; no file written')
        np.savez_compressed(output / name, **arrays)
    activity = {'counts': counts}
    for seed in range(42, 47):
        rng = np.random.default_rng(seed)
        people = np.sort(rng.choice(eligible, 500, replace=False))
        query = np.stack([rng.choice(order[offset[p]:offset[p+1]], 40, replace=False) for p in people])
        train = np.random.default_rng(seed).random(len(df), dtype=np.float32) >= .1
        train[query.ravel()] = False
        n = np.bincount(i[train], minlength=len(items))
        total = np.bincount(i[train], weights=y[train].astype(float), minlength=len(items))
        rate = np.divide(total, n, out=np.zeros(len(items)), where=n > 0)
        common = {'people': people, 'y': y[query], 'item_rate': rate[i[query]]}
        save(f'Eedi_correctness_{seed}.npz', common)
        save(f'Eedi_bilinear_sensitivity_{seed}.npz', common)
        chosen = np.flatnonzero((labels >= 0) & (rate > 0) & (rate < 1) & (n > 0))
        save(f'Eedi_item_{seed}.npz', {'items': chosen, 'y': labels[chosen], 'groups': chosen})
        activity[f'seed{seed}'] = people
        print(f'Eedi seed {seed}: reconstructed fields match archived evaluation arrays', flush=True)
    save('Eedi_response_counts.npz', activity)
    print('Local data ready:', output)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-zip', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'local_data')
    a = p.parse_args(); prepare(a.data_zip, a.output)
