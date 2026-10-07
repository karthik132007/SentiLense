#!/usr/bin/env python3
"""Train a calibrated linear SVM document model on Stanford's IMDb splits."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from sentiment_model.preprocessing import normalize_text
from sentiment_model.svm import fit_svm, positive_probabilities
from scripts.train import binary_metrics, select_thresholds, abstention_metrics


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/aclImdb_v1.tar.gz')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    started = time.monotonic()
    directory = args.output_dir / 'model/reviews'
    artifacts = args.output_dir / 'artifacts/reviews'
    directory.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    records = {'train': [], 'test': []}
    seen = {'train': set(), 'test': set()}
    duplicate_counts = {'train': 0, 'test': 0, 'test_matching_train': 0}
    raw_records = {'train': [], 'test': []}
    # Stream gzip once; sorting tar members and seeking backwards repeatedly
    # would decompress the archive thousands of times.
    with tarfile.open(args.data, 'r|gz') as archive:
        for member in archive:
            parts = member.name.split('/')
            if not member.isfile() or len(parts) != 4 or parts[0] != 'aclImdb' or parts[1] not in raw_records or parts[2] not in {'neg', 'pos'} or not member.name.endswith('.txt'):
                continue
            text = archive.extractfile(member).read().decode('utf-8')
            raw_records[parts[1]].append((member.name, text.replace('<br />', ' ').replace('<br>', ' '), int(parts[2] == 'pos')))
    for split in ['train', 'test']:
        for name, text, label in sorted(raw_records[split]):
            fingerprint = hashlib.sha256(normalize_text(text).encode()).hexdigest()
            if fingerprint in seen[split]:
                duplicate_counts[split] += 1
                continue
            if split == 'test' and fingerprint in seen['train']:
                duplicate_counts['test_matching_train'] += 1
                continue
            seen[split].add(fingerprint)
            records[split].append((name, text, label))
    texts = [row[1] for row in records['train']]
    labels = np.array([row[2] for row in records['train']], dtype=np.uint8)
    train, validation = train_test_split(np.arange(len(labels)), test_size=0.2,
                                         random_state=42, stratify=labels)
    calibration, validation = train_test_split(validation, test_size=0.5,
                                               random_state=42, stratify=labels[validation])
    print(f'Review splits: {len(train)} train, {len(calibration)} calibration, {len(validation)} validation', flush=True)
    pipeline, validation_p, svm_details = fit_svm(
        [texts[i] for i in train], labels[train], [texts[i] for i in calibration], labels[calibration],
        [texts[i] for i in validation], labels[validation], word_features=200000, min_df=2)
    lower, upper, selection = select_thresholds(labels[validation], validation_p, 0.90)
    # The official test is transformed only after model/threshold selection.
    test_text = [row[1] for row in records['test']]
    test_labels = np.array([row[2] for row in records['test']], dtype=np.uint8)
    test_p = positive_probabilities(pipeline, test_text, batch_size=1000)
    report = {'dataset': 'Stanford Large Movie Review Dataset (IMDb)',
              'source': 'https://ai.stanford.edu/~amaas/data/sentiment/',
              'protocol': 'Deduplicate normalized reviews; remove test matches to train. Stratified 80/10/10 train/calibration/validation split of official train, seed 42. Fit features/SVM on train only; calibrate on disjoint calibration; select C and uncertainty thresholds on validation only; evaluate once on official test. No refit.',
              'duplicates_removed': duplicate_counts,
              'svm_training': svm_details,
              'validation': binary_metrics(labels[validation], validation_p),
              'test': binary_metrics(test_labels, test_p),
              'selective_test': abstention_metrics(test_labels, test_p, lower, upper),
              'negative_threshold': lower, 'positive_threshold': upper}
    save(artifacts / 'evaluation.json', report)
    save(artifacts / 'neutral_threshold_selection.json', selection)
    save(artifacts / 'split_files.json', {
        'train': [records['train'][i][0] for i in train],
        'calibration': [records['train'][i][0] for i in calibration],
        'validation': [records['train'][i][0] for i in validation],
        'test': [row[0] for row in records['test']]})
    np.savez_compressed(artifacts / 'heldout_predictions.npz', test_labels=test_labels,
                        test_positive_probability=test_p, validation_labels=labels[validation],
                        validation_positive_probability=validation_p)
    path = directory / 'sentiment_pipeline.joblib'
    joblib.dump(pipeline, path, compress=3)
    reloaded = joblib.load(path)
    np.testing.assert_array_equal(reloaded.predict_proba(test_text[:100]), pipeline.predict_proba(test_text[:100]))
    metadata = {
        'model': 'Review word/character TF-IDF + calibrated LinearSVC (linear SVM)',
        'model_family': 'svm',
        'dataset': report['dataset'], 'dataset_source': report['source'],
        'dataset_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'records_used': len(labels) + len(test_labels), 'records_trained': len(train),
        'calibration_records': len(calibration), 'validation_records': len(validation), 'test_records': len(test_labels),
        'accuracy': report['test']['accuracy'], 'f1': report['test']['f1'],
        'roc_auc': report['test']['roc_auc'], **svm_details,
        'negative_threshold': lower, 'positive_threshold': upper,
        'pipeline_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'preprocessing': 'Shared sentiment_model normalization/tokenizer; HTML br tags replaced with whitespace before training.',
        'inference_unit': 'Full extracted review document including its title; segments are explanatory only.',
        'confidence_definition': 'max(sigmoid-calibrated binary positive probability, binary negative probability); not neutral probability or guaranteed certainty',
        'neutral_policy': selection['meaning'], 'evaluation_protocol': report['protocol'],
        'sklearn_version': sklearn.__version__, 'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'total_seconds': time.monotonic() - started,
        'limitations': ['Binary English movie reviews; no true neutral training class.',
                       'Validation review split is not movie-disjoint.',
                       'Movie-review benchmark accuracy does not establish news, webpage, or sarcasm accuracy.',
                       'Reported failing article was not added to training or threshold selection.']}
    save(directory / 'model_metadata.json', metadata)
    print(json.dumps({k: metadata[k] for k in ['accuracy', 'f1', 'roc_auc', 'negative_threshold', 'positive_threshold', 'fit_seconds']}, indent=2))


if __name__ == '__main__':
    main()
