"""Frozen SVM/RoBERTa experts with a learned three-class stacking head."""
from pathlib import Path
from threading import RLock
import hashlib
import re

import joblib
import numpy as np

LABELS = ('negative', 'neutral', 'positive')
MODEL_ID = 'cardiffnlp/twitter-roberta-base-sentiment'
REVISION = 'daefdd1f6ae931839bce4d0f3db0a1a4265cd50f'


def roberta_text(text):
    """Preserve case/context; use the checkpoint's mention and URL placeholders."""
    from .preprocessing import normalize_text
    if not normalize_text(text):
        raise ValueError('text must contain non-whitespace content')
    import ftfy
    import html
    text = html.unescape(ftfy.fix_text(text))
    text = re.sub(r'https?://\S+|www\.\S+', 'http', text)
    text = re.sub(r'(?<!\w)@\w+', '@user', text)
    return ' '.join(text.split())


def fusion_features(svm_probability, roberta_probability):
    """Four expert features: SVM log odds and three RoBERTa log scores."""
    p = np.clip(np.asarray(svm_probability), 1e-6, 1 - 1e-6)
    r = np.clip(np.asarray(roberta_probability), 1e-6, 1)
    return np.column_stack([np.log(p / (1 - p)), np.log(r)])


class RobertaExpert:
    def __init__(self, directory):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.set_num_threads(4)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(directory), local_files_only=True, trust_remote_code=False).eval()
        if self.model.config.num_labels != 3:
            raise ValueError('Expected Cardiff three-class sentiment checkpoint')
        self.lock = RLock()

    def probabilities(self, texts, *, batch_size=16, progress=None):
        """Average scores across overlapping 512-token windows for long inputs."""
        result = []
        with self.lock, self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = [roberta_text(t) for t in texts[start:start + batch_size]]
                encoded = self.tokenizer(batch, padding=True, truncation=True,
                    max_length=512, stride=64, return_overflowing_tokens=True, return_tensors='pt')
                mapping = encoded.pop('overflow_to_sample_mapping').numpy()
                scores = []
                for offset in range(0, len(mapping), batch_size):
                    inputs = {k: v[offset:offset + batch_size] for k, v in encoded.items()}
                    scores.append(self.model(**inputs).logits.softmax(dim=-1).cpu().numpy())
                scores = np.concatenate(scores)
                for index in range(len(batch)):
                    result.append(scores[mapping == index].mean(axis=0))
                if progress:
                    progress(min(start + batch_size, len(texts)), len(texts))
        return np.asarray(result, dtype=np.float64)


def verify_manifest(directory, metadata):
    for name, expected in metadata['files_sha256'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid model artifact path')
        digest = hashlib.sha256()
        with (directory / path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f'Fusion artifact checksum mismatch: {name}')


class FusionModel:
    classes_ = np.array([0, 1, 2])

    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        verify_manifest(self.directory, metadata)
        from .inference import _load
        self.svm, _ = _load(str((self.directory / 'svm').resolve()))
        self.roberta = RobertaExpert(self.directory / 'roberta')
        self.head = joblib.load(self.directory / 'fusion_head.joblib')
        if list(self.head.classes_) != [0, 1, 2]:
            raise ValueError('Fusion head must include negative, neutral and positive')

    def predict_proba(self, texts):
        from .svm import positive_probabilities
        svm = positive_probabilities(self.svm, texts)
        roberta = self.roberta.probabilities(texts)
        return self.head.predict_proba(fusion_features(svm, roberta))

    def predict_many(self, texts):
        probabilities = self.predict_proba(texts)
        return [score_result(dict(zip(LABELS, map(float, row)))) for row in probabilities]


def score_result(scores):
    if set(scores) != set(LABELS) or not all(np.isfinite(p) and 0 <= p <= 1 for p in scores.values()):
        raise ValueError('Expected finite negative/neutral/positive probabilities')
    if not np.isclose(sum(scores.values()), 1, atol=1e-6):
        raise ValueError('Class probabilities must sum to one')
    # Tie breaks toward neutral, without assigning a fabricated neutral score.
    label = max(('neutral', 'negative', 'positive'), key=lambda name: scores[name])
    return {'sentiment': label, 'confidence': scores[label], 'scores': scores}
