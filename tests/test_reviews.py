"""Recompute IMDb metrics and verify the separate movie-review artifact."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from sentiment_model.inference import _load
from sentiment_model.reviews import predict_review

ROOT = Path(__file__).resolve().parents[1]


def test_review_test_metrics_and_splits_are_auditable():
    directory = ROOT / 'artifacts/reviews'
    evaluation = json.loads((directory / 'evaluation.json').read_text())
    metadata = json.loads((ROOT / 'model/reviews/model_metadata.json').read_text())
    with np.load(directory / 'heldout_predictions.npz') as data:
        y, p = data['test_labels'], data['test_positive_probability']
        assert accuracy_score(y, p >= 0.5) == evaluation['test']['accuracy'] == metadata['accuracy']
        assert f1_score(y, p >= 0.5) == metadata['f1']
        assert roc_auc_score(y, p) == metadata['roc_auc']
    splits = json.loads((directory / 'split_files.json').read_text())
    combined = [name for names in splits.values() for name in names]
    assert len(combined) == len(set(combined)) == metadata['records_used']
    assert len(splits['train']) == metadata['records_trained']
    assert all(name.startswith('aclImdb/test/') for name in splits['test'])


def test_review_unknown_vocabulary_abstains_and_saved_preprocessing_matches():
    assert predict_review('qzxvtrjkpnmlzqx')['scores'] == {'positive': 0.5, 'negative': 0.5}
    pipeline, metadata = _load(str(ROOT / 'model/reviews'))
    assert metadata['model_family'] == 'svm'
    assert isinstance(pipeline['classifier'], CalibratedClassifierCV)
    assert isinstance(pipeline['classifier'].estimator.estimator, LinearSVC)
    text = 'I love this brilliant film &amp; its wonderful screenplay!'
    assert predict_review(text)['scores']['positive'] == pipeline.predict_proba([text])[0, 1]
    assert 0 <= metadata['negative_threshold'] < 0.5 < metadata['positive_threshold'] <= 1
