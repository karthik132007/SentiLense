"""Contract checks, leakage audit, and independent evaluation recomputation."""

import json
from pathlib import Path
import unittest

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from sentiment_model import predict_sentiment
from sentiment_model.preprocessing import normalize_text, tokenize

ROOT = Path(__file__).resolve().parents[1]


class PreprocessingTests(unittest.TestCase):
    def test_preserves_sentiment_and_removes_identity(self):
        text = normalize_text("@Alice I do NOT like it :( Visit https://example.org/a!")
        self.assertIn("do not like", text)
        self.assertIn(":(", tokenize(text))
        self.assertIn("!", tokenize(text))
        self.assertIn("usertoken", text)
        self.assertIn("urltoken", text)
        self.assertNotIn("alice", text)
        self.assertNotIn("example.org", text)

    def test_repairs_characters_without_discarding_unicode(self):
        self.assertEqual(normalize_text("FranÃ§ais &amp; 😊"), "français & 😊")
        self.assertEqual(normalize_text("I can’t\x00wait\ufffd!"), "i can't wait !")
        self.assertIn("<3", tokenize(normalize_text("love <3")))
        self.assertIn("can't", tokenize(normalize_text("can't")))

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            normalize_text(None)
        with self.assertRaises(ValueError):
            predict_sentiment("  \n\t")


@unittest.skipUnless((ROOT / "model/model_metadata.json").exists(), "Train model first")
class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary_dir = ROOT / 'model/svm' if (ROOT / 'model/svm').exists() else ROOT / 'model'
        cls.pipeline = joblib.load(cls.binary_dir / "sentiment_pipeline.joblib")
        cls.metadata = json.loads((cls.binary_dir / "model_metadata.json").read_text())
        cls.binary_artifacts = ROOT / 'artifacts/svm' if (ROOT / 'artifacts/svm').exists() else ROOT / 'artifacts'
        cls.evaluation = json.loads((cls.binary_artifacts / "evaluation.json").read_text())

    def test_backend_contract_and_examples(self):
        examples = json.loads((ROOT / "artifacts/inference_examples.json").read_text())
        for example in examples["examples"]:
            result = predict_sentiment(example["text"])
            self.assertEqual(result, example["prediction"])
            self.assertEqual(set(result), {"sentiment", "confidence", "scores"})
            self.assertAlmostEqual(sum(result["scores"].values()), 1.0)
            self.assertEqual(result["confidence"], max(result["scores"].values()))
        self.assertEqual(predict_sentiment("I absolutely love this!")["sentiment"], "positive")
        self.assertEqual(predict_sentiment("I hate this product.")["sentiment"], "negative")
        self.assertEqual(list(self.pipeline.classes_), [0, 1])

    def test_saved_classifier_is_calibrated_svm(self):
        classifier = self.pipeline['classifier']
        self.assertIsInstance(classifier, CalibratedClassifierCV)
        self.assertIsInstance(classifier.estimator, FrozenEstimator)
        self.assertIsInstance(classifier.estimator.estimator, LinearSVC)
        self.assertEqual(self.metadata['model_family'], 'svm')
        self.assertTrue(self.metadata['calibration']['disjoint_from_train_validation_test'])
        self.assertEqual(self.metadata['calibration']['records'], self.metadata['splits']['calibration']['records'])
        self.assertEqual(classifier.method, 'sigmoid')

    def test_unknown_vocabulary_abstains(self):
        result = predict_sentiment("qzxvtrjkpnmlzqx", model_dir=self.binary_dir)
        self.assertEqual(result['sentiment'], 'neutral')
        self.assertEqual(result['scores']['positive'], 0.5)

    def test_raw_pipeline_runs_shared_preprocessing(self):
        raw = "@someone I can’t hate this &amp; love it!!! http://example.org/test"
        normalized = normalize_text(raw)
        np.testing.assert_allclose(self.pipeline.predict_proba([raw]), self.pipeline[1:].predict_proba([normalized]))

    def test_independently_recomputes_test_metrics(self):
        with np.load(self.binary_artifacts / "heldout_predictions.npz") as predictions:
            labels = predictions["test_labels"]
            probability = predictions["test_positive_probability"]
            report = self.evaluation["test"]
            self.assertEqual(accuracy_score(labels, probability >= 0.5), report["accuracy"])
            self.assertEqual(f1_score(labels, probability >= 0.5), report["f1"])
            self.assertEqual(roc_auc_score(labels, probability), report["roc_auc"])
            self.assertEqual(confusion_matrix(labels, probability >= 0.5).tolist(), report["confusion_matrix"])

    def test_split_row_indices_do_not_overlap(self):
        with np.load(self.binary_artifacts / "split_indices.npz") as splits:
            arrays = [splits[k] for k in ["train", "calibration", "validation", "test"]]
            combined = np.concatenate(arrays)
            self.assertEqual(len(combined), len(np.unique(combined)))
            self.assertEqual(len(combined), self.metadata["records_used"])
            self.assertEqual(len(arrays[0]), self.metadata["records_trained"])
            self.assertEqual(len(arrays[1]), self.metadata['calibration']['records'])


if __name__ == "__main__":
    unittest.main()
