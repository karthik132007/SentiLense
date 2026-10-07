"""Reusable linear SVM training with disjoint probability calibration."""

import time
import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from .preprocessing import normalize_batch, tokenize


def fit_svm(train_text, train_y, calibration_text, calibration_y, validation_text,
            validation_y, *, word_features=300000, char_features=50000, min_df=3,
            seed=42, c_values=(0.5, 1.0, 2.0)):
    """Fit TF-IDF/SVM on train, sigmoid on calibration, choose C on validation.

    Callers must supply disjoint partitions. Test text is never passed here.
    Character features help with spelling variations without adding word lists
    or hand-written polarity rules to inference.
    """
    features = Pipeline([
        ('preprocessing', FunctionTransformer(normalize_batch, validate=False)),
        ('tfidf', FeatureUnion([
            ('word', TfidfVectorizer(tokenizer=tokenize, token_pattern=None,
                lowercase=False, ngram_range=(1, 2), min_df=min_df,
                max_features=word_features, sublinear_tf=True, dtype=np.float32)),
            ('character', TfidfVectorizer(analyzer='char_wb', lowercase=False,
                ngram_range=(4, 5), min_df=min_df, max_features=char_features,
                sublinear_tf=True, dtype=np.float32)),
        ])),
    ])
    print('Fitting word/character TF-IDF on train only', flush=True)
    started = time.monotonic()
    train_x = features.fit_transform(train_text)
    calibration_x = features.transform(calibration_text)
    validation_x = features.transform(validation_text)
    print(f'Features ready: {train_x.shape}, {train_x.nnz:,} nonzero values', flush=True)
    diagnostics, best = [], None
    with threadpool_limits(limits=4):
        for c in c_values:
            estimator = LinearSVC(C=c, dual='auto', max_iter=3000, tol=1e-4,
                                  random_state=seed)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always', ConvergenceWarning)
                estimator.fit(train_x, train_y)
            convergence = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
            if convergence:
                raise RuntimeError(f'SVM did not converge at C={c}: {convergence}')
            classifier = CalibratedClassifierCV(FrozenEstimator(estimator), method='sigmoid')
            classifier.fit(calibration_x, calibration_y)
            probability = classifier.predict_proba(validation_x)[:, 1]
            probability[np.asarray(validation_x.getnnz(axis=1)) == 0] = 0.5
            row = {'C': c, 'accuracy': float(accuracy_score(validation_y, probability >= 0.5)),
                   'f1': float(f1_score(validation_y, probability >= 0.5)),
                   'brier_score': float(brier_score_loss(validation_y, probability)),
                   'iterations': int(estimator.n_iter_), 'convergence_warnings': convergence}
            diagnostics.append(row)
            print('SVM validation:', row, flush=True)
            rank = (row['accuracy'], -row['brier_score'])
            if best is None or rank > best[0]:
                best = rank, classifier, estimator, probability, c
    _, classifier, estimator, validation_probability, c = best
    pipeline = Pipeline(features.steps + [('classifier', classifier)])
    vocabularies = {name: len(transformer.vocabulary_) for name, transformer in features['tfidf'].transformer_list}
    return pipeline, validation_probability, {
        'selected_C': c, 'validation_candidates': diagnostics,
        'classifier_parameters': estimator.get_params(),
        'classifier_iterations': [int(estimator.n_iter_)], 'convergence_warnings': [],
        'vocabulary_size': sum(vocabularies.values()), 'vocabulary_sizes': vocabularies,
        'feature_parameters': {'word_ngram_range': [1, 2], 'character_ngram_range': [4, 5],
            'character_analyzer': 'char_wb', 'min_df': min_df,
            'max_word_features': word_features, 'max_character_features': char_features,
            'sublinear_tf': True},
        'calibration': {'method': 'sigmoid', 'records': len(calibration_y),
                        'base_estimator_frozen': True, 'disjoint_from_train_validation_test': True},
        'fit_seconds': time.monotonic() - started,
    }


def positive_probabilities(pipeline, texts, *, batch_size=10000):
    """Match runtime's unknown-vocabulary policy during held-out evaluation."""
    batches = []
    for start in range(0, len(texts), batch_size):
        matrix = pipeline[:-1].transform(texts[start:start + batch_size])
        probability = pipeline[-1].predict_proba(matrix)[:, 1]
        probability[np.asarray(matrix.getnnz(axis=1)) == 0] = 0.5
        batches.append(probability)
    return np.concatenate(batches)
