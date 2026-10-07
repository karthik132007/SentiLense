"""Training leakage protections and evaluation-only polarity regressions."""

import json
from pathlib import Path

import pytest

from scripts.train_tweets import prepare
from sentiment_model import predict_sentiment
from sentiment_model.preprocessing import normalize_text

ROOT = Path(__file__).resolve().parents[1]
REGRESSIONS = json.loads((ROOT / 'artifacts/sentiment_regressions.json').read_text())['examples']


@pytest.mark.parametrize('example', REGRESSIONS, ids=[r['text'] for r in REGRESSIONS])
def test_evaluation_only_complaints_negation_and_positive_profanity(example):
    result = predict_sentiment(example['text'])
    assert result['sentiment'] == example['expected_sentiment']
    assert result['scores'] == pytest.approx(example['prediction']['scores'], abs=1e-5)
    assert result['confidence'] == pytest.approx(example['prediction']['confidence'], abs=1e-5)


def test_official_splits_drop_duplicates_conflicts_and_neutral(tmp_path):
    fixtures = {
        'train': [('Duplicate text', 2), ('Training only', 0), ('Conflict', 0), ('Neutral', 1)],
        'val': [('Validation only', 2), ('Conflict', 2)],
        'test': [('Duplicate text', 2), ('Test only', 0)],
    }
    for split, rows in fixtures.items():
        (tmp_path / f'{split}_text.txt').write_text('\n'.join(r[0] for r in rows) + '\n')
        (tmp_path / f'{split}_labels.txt').write_text('\n'.join(str(r[1]) for r in rows) + '\n')
    groups, inspection = prepare(tmp_path)
    assert [r[0] for r in groups['train']] == ['Training only']
    assert [r[0] for r in groups['val']] == ['Validation only']
    assert [r[0] for r in groups['test']] == ['Duplicate text', 'Test only']
    assert inspection['conflicting_normalized_groups'] == 1
    all_ids = [r[2] for rows in groups.values() for r in rows]
    assert len(all_ids) == len(set(all_ids))


@pytest.mark.skipif(not (ROOT / 'data/tweeteval/train_text.txt').exists(), reason='Optional training data not deployed')
def test_regression_examples_were_not_added_to_training():
    groups, _ = prepare(ROOT / 'data/tweeteval')
    training_text = {normalize_text(r[0]) for name in ('train', 'val') for r in groups[name]}
    assert all(normalize_text(r['text']) not in training_text for r in REGRESSIONS)
