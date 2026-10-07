"""Validate both experts, three-class aggregation, leakage and held-out metrics."""
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, log_loss

from sentiment_model import predict_sentiment
from sentiment_model.fusion import fusion_features, score_result, roberta_text, verify_manifest
from sentiment_model.inference import _load

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_preprocessing_preserves_context():
    assert roberta_text('@Alice This is NOT bad! https://example.org') == '@user This is NOT bad! http'


def test_scores_and_neutral_are_three_class_decisions():
    result = score_result({'negative': .1, 'neutral': .7, 'positive': .2})
    assert result['sentiment'] == 'neutral'
    assert result['confidence'] == .7
    with pytest.raises(ValueError):
        score_result({'negative': .5, 'positive': .5})


def test_corrupt_or_unsafe_artifact_manifest_is_rejected(tmp_path):
    (tmp_path / 'weights').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_manifest(tmp_path, {'files_sha256': {'weights': 'incorrect'}})
    with pytest.raises(ValueError, match='Invalid model artifact path'):
        verify_manifest(tmp_path, {'files_sha256': {'../weights': 'incorrect'}})


def test_saved_fusion_uses_both_experts_and_correct_class_order():
    model, metadata = _load(str(ROOT / 'model'))
    assert metadata['model_family'] == 'svm_roberta_fusion'
    assert metadata['roberta_model_id'] == 'cardiffnlp/twitter-roberta-base-sentiment'
    assert list(model.head.classes_) == [0, 1, 2]
    assert list(model.svm.classes_) == [0, 1]
    same_roberta = np.tile([.2, .5, .3], (2, 1))
    changed_svm = model.head.predict_proba(fusion_features([.1, .9], same_roberta))
    assert not np.allclose(changed_svm[0], changed_svm[1])
    same_svm = [.5, .5]
    changed_roberta = model.head.predict_proba(fusion_features(same_svm, [[.8,.1,.1],[.1,.1,.8]]))
    assert not np.allclose(changed_roberta[0], changed_roberta[1])


def test_batch_and_single_prediction_agree():
    model, _ = _load(str(ROOT / 'model'))
    texts = ['I absolutely love this!', 'This subject is trash and a waste of time.']
    batch = model.predict_many(texts)
    for text, result in zip(texts, batch, strict=True):
        single = predict_sentiment(text)
        assert result['sentiment'] == single['sentiment']
        assert result['scores'] == pytest.approx(single['scores'], abs=1e-5)
        assert sum(result['scores'].values()) == pytest.approx(1)
        assert result['confidence'] == max(result['scores'].values())


def test_fusion_partitions_are_disjoint_from_base_training():
    with np.load(ROOT / 'artifacts/fusion/split_indices.npz') as splits:
        names = ['train', 'calibration', 'validation', 'test', 'svm_source_train']
        all_ids = np.concatenate([splits[n] for n in names])
        assert len(all_ids) == len(np.unique(all_ids))
        assert len(splits['train']) == 1000
        assert len(splits['calibration']) == len(splits['validation']) == 500


def test_independent_three_class_evaluation_recomputation():
    report = json.loads((ROOT / 'artifacts/fusion/evaluation.json').read_text())['test']
    with np.load(ROOT / 'artifacts/fusion/heldout_predictions.npz') as data:
        y, p = data['test_labels'], data['test_probabilities']
        assert set(y) == {0, 1, 2}
        assert accuracy_score(y, p.argmax(axis=1)) == report['accuracy']
        assert f1_score(y, p.argmax(axis=1), average='macro') == report['macro_f1']
        assert log_loss(y, p, labels=[0, 1, 2]) == report['log_loss']
