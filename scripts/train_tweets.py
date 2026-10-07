#!/usr/bin/env python3
"""Train the deployed binary SVM on gold TweetEval sentiment annotations.

Official neutral rows are excluded: runtime neutral remains an abstention band.
Use --download once to obtain the pinned official files. All fitting, calibration
and selection precede test evaluation; no user complaints are training examples.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import train_test_split
from sentiment_model.preprocessing import normalize_text
from sentiment_model.svm import fit_svm, positive_probabilities
from scripts.train import save_json, sha256_file, binary_metrics, select_thresholds, abstention_metrics

REVISION = '4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66'
SOURCE = 'https://github.com/cardiffnlp/tweeteval'
# Verified against the pinned upstream files; reject accidental data changes.
CHECKSUMS = {
    'train_text.txt': '368f01052ea6fd8ffc408a2a2e6ac9669e31542581a0396ef16591ea26eb98a6',
    'train_labels.txt': '122bfb1732fb6995b0e5c5f726c0ba457c469c3b6e60513007ce5037f23e65d4',
    'val_text.txt': 'e5b021e6fc45064c260b09814b803d8f56cada519c4d952d72f43d48a350a964',
    'val_labels.txt': 'b4566926c72e2e4e2916c864def94e76c4cdde52446af2c7ba4fc2006e057e51',
    'test_text.txt': '09a93a55c63fd93f97485ef7302889d7edb4091cd49733aa37da094f0bfa0675',
    'test_labels.txt': '6afb4afe9374d1f983bcf9a7c79b108d0f37fdf020a83f30488309bed215db9d',
}


def download(directory):
    import httpx
    directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for name, checksum in CHECKSUMS.items():
            path = directory / name
            if path.exists() and sha256_file(path) == checksum:
                continue
            url = f'https://raw.githubusercontent.com/cardiffnlp/tweeteval/{REVISION}/datasets/sentiment/{name}'
            response = client.get(url)
            response.raise_for_status()
            import hashlib
            if hashlib.sha256(response.content).hexdigest() != checksum:
                raise ValueError(f'Upstream checksum mismatch: {name}')
            path.write_bytes(response.content)


def prepare(directory):
    """Remove normalized conflicts/duplicates, preserving official partitions.

    Held-out rows take precedence over training duplicates. IDs are global
    original line offsets, allowing independent partition-overlap checks.
    """
    raw, labels_by_text, offset = {}, defaultdict(set), 0
    for split in ('train', 'val', 'test'):
        texts = (directory / f'{split}_text.txt').read_text().splitlines()
        labels = (directory / f'{split}_labels.txt').read_text().splitlines()
        rows = []
        for line, (text, label) in enumerate(zip(texts, labels, strict=True)):
            if label not in {'0', '1', '2'}:
                raise ValueError(f'Invalid upstream label: {label}')
            normalized = normalize_text(text)
            labels_by_text[normalized].add(label)
            rows.append((text, label, offset + line, normalized))
        raw[split] = rows
        offset += len(rows)
    conflicts = {text for text, labels in labels_by_text.items() if len(labels) > 1}
    seen, cleaned, discarded = set(), {}, {}
    for split in ('test', 'val', 'train'):
        cleaned[split], discarded[split] = [], Counter()
        for text, label, index, normalized in raw[split]:
            if not normalized:
                discarded[split]['empty'] += 1
            elif normalized in conflicts:
                discarded[split]['conflicting_label'] += 1
            elif normalized in seen:
                discarded[split]['duplicate'] += 1
            else:
                seen.add(normalized)
                if label == '1':
                    discarded[split]['neutral_excluded_from_binary_model'] += 1
                else:
                    cleaned[split].append((text, int(label == '2'), index))
    inspection = {
        'source': SOURCE, 'revision': REVISION,
        'files': {name: {'sha256': sha256_file(directory / name),
            'url': f'https://raw.githubusercontent.com/cardiffnlp/tweeteval/{REVISION}/datasets/sentiment/{name}'}
            for name in CHECKSUMS},
        'rows': offset,
        'official_label_mapping': {'0': 'negative', '1': 'neutral', '2': 'positive'},
        'binary_label_mapping': {'0': 'negative', '1': 'positive'},
        'deduplication': 'Remove all normalized conflicting-label groups. Collapse duplicates with test > validation > train priority; preserve official split assignment.',
        'conflicting_normalized_groups': len(conflicts),
        'source_splits': {name: {'records': len(rows), 'original_labels': dict(Counter(r[1] for r in rows)),
            'retained_binary_records': len(cleaned[name]), 'excluded': dict(discarded[name])}
            for name, rows in raw.items()},
    }
    return cleaned, inspection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/tweeteval')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--baseline-dir', type=Path, help='Compare a frozen old model on the same held-out rows')
    args = parser.parse_args()
    started = time.monotonic()
    if args.download:
        download(args.data)
    for name, expected in CHECKSUMS.items():
        if sha256_file(args.data / name) != expected:
            raise ValueError(f'Dataset checksum mismatch: {name}')
    groups, inspection = prepare(args.data)
    train_rows = groups['train']
    train_y = np.array([r[1] for r in train_rows], dtype=np.uint8)
    train, calibration = train_test_split(np.arange(len(train_rows)), test_size=.15,
        stratify=train_y, random_state=42)
    rows = {'train': [train_rows[i] for i in train], 'calibration': [train_rows[i] for i in calibration],
            'validation': groups['val'], 'test': groups['test']}
    texts = {s: [r[0] for r in items] for s, items in rows.items()}
    labels = {s: np.array([r[1] for r in items], dtype=np.uint8) for s, items in rows.items()}
    ids = {s: np.array([r[2] for r in items], dtype=np.int64) for s, items in rows.items()}
    splits = {s: {'records': len(items), 'negative': int((labels[s] == 0).sum()),
                 'positive': int((labels[s] == 1).sum())} for s, items in rows.items()}
    print('Gold-label splits:', splits, flush=True)
    pipeline, validation_p, details = fit_svm(texts['train'], labels['train'],
        texts['calibration'], labels['calibration'], texts['validation'], labels['validation'],
        min_df=2, word_features=150000, char_features=50000, c_values=(.1, .5, 1., 2.))
    lower, upper, thresholds = select_thresholds(labels['validation'], validation_p, .90)
    # All model and policy choices are frozen before accessing test features.
    test_p = positive_probabilities(pipeline, texts['test'])
    report = binary_metrics(labels['test'], test_p)
    evaluation = {
        'test': report, 'validation': binary_metrics(labels['validation'], validation_p),
        'uncertainty_policy': {'negative_threshold': lower, 'positive_threshold': upper,
            'validation': abstention_metrics(labels['validation'], validation_p, lower, upper),
            'test': abstention_metrics(labels['test'], test_p, lower, upper)},
        'splits': splits, 'seed': 42, 'svm_training': details,
        'evaluation_protocol': 'Official TweetEval split assignments retained after normalized deduplication. Calibration is 15% of official train. Fit features/SVM on train, calibrator on calibration, select C/thresholds on official validation only. Test is the negative/positive subset of official test, not the complete three-class benchmark. No refit on held-out data.',
    }
    if args.baseline_dir:
        baseline = joblib.load(args.baseline_dir / 'sentiment_pipeline.joblib')
        baseline_p = positive_probabilities(baseline, texts['test'])
        baseline_meta = json.loads((args.baseline_dir / 'model_metadata.json').read_text())
        evaluation['same_test_baseline'] = {'pipeline_sha256': sha256_file(args.baseline_dir / 'sentiment_pipeline.joblib'),
            'dataset': baseline_meta['dataset'], 'test': binary_metrics(labels['test'], baseline_p),
            'uncertainty_policy': abstention_metrics(labels['test'], baseline_p,
                baseline_meta['negative_threshold'], baseline_meta['positive_threshold'])}
    model_dir, artifacts = args.output_dir / 'model', args.output_dir / 'artifacts'
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / 'sentiment_pipeline.joblib'
    joblib.dump(pipeline, model_path, compress=3)
    metadata = {
        'model': 'Word/character TF-IDF + calibrated LinearSVC (linear SVM)', 'model_family': 'svm',
        'dataset': 'TweetEval sentiment (gold negative/positive annotations)', 'analysis_model': 'tweeteval',
        'dataset_source': SOURCE, 'dataset_revision': REVISION,
        'source_records': inspection['rows'], 'records_used': sum(len(r) for r in rows.values()),
        'records_trained': len(train), 'splits': splits,
        **{k: report[k] for k in ('accuracy','precision','recall','f1','f1_macro','roc_auc')},
        'negative_threshold': lower, 'positive_threshold': upper,
        'neutral_policy': thresholds['meaning'], 'threshold_selection': thresholds['objective'],
        'target_validation_precision': .90,
        'confidence_definition': 'max(P(positive), P(negative)); binary polarity confidence, never a learned neutral probability',
        'zero_vocabulary_policy': 'Return neutral with both binary scores 0.5 when no features are recognized',
        'classifier_classes': [0, 1], 'label_mapping': {'0': 'negative', '2': 'positive'},
        'preprocessing': 'Shared NFC/ftfy/HTML/lowercase/URL/mention normalization; preserve negation and punctuation, no stopword removal',
        'deduplication': inspection['deduplication'], 'random_seed': 42, **details,
        'versions': {'python': sys.version.split()[0], 'scikit-learn': sklearn.__version__,
                     'numpy': np.__version__, 'joblib': joblib.__version__},
        'trained_at_utc': datetime.now(timezone.utc).isoformat(), 'pipeline_sha256': sha256_file(model_path),
        'total_seconds': time.monotonic() - started,
        'limitations': ['Official neutral training/test rows excluded; this is binary polarity, not a three-class TweetEval score',
            'Validation class balance differs from official test; class probabilities/selected precision may shift',
            'English tweets do not establish accuracy/calibration for webpages, stance, sarcasm or other languages'],
    }
    save_json(model_dir / 'model_metadata.json', metadata)
    for name, value in [('evaluation', evaluation), ('dataset_inspection', inspection), ('neutral_threshold_selection', thresholds)]:
        save_json(artifacts / f'{name}.json', value)
    np.savez_compressed(artifacts / 'split_indices.npz', **ids)
    np.savez_compressed(artifacts / 'heldout_predictions.npz', validation_source_rows=ids['validation'],
        validation_labels=labels['validation'], validation_positive_probability=validation_p,
        test_source_rows=ids['test'], test_labels=labels['test'], test_positive_probability=test_p)
    np.savetxt(artifacts / 'confusion_matrix.csv', report['confusion_matrix'], delimiter=',', fmt='%d')
    (artifacts / 'classification_report.txt').write_text(classification_report(labels['test'], test_p >= .5,
        target_names=['negative', 'positive'], digits=6))
    fpr, tpr, roc_thresholds = roc_curve(labels['test'], test_p)
    np.savez_compressed(artifacts / 'roc_curve.npz', false_positive_rate=fpr, true_positive_rate=tpr, thresholds=roc_thresholds)
    loaded = joblib.load(model_path)
    np.testing.assert_array_equal(pipeline.predict_proba(texts['test'][:100]), loaded.predict_proba(texts['test'][:100]))
    from sentiment_model import predict_sentiment
    examples = [{'text': t, 'prediction': predict_sentiment(t, model_dir=model_dir)} for t in
        ['I absolutely love this!', 'I hate this product.', 'The product arrived yesterday.']]
    save_json(artifacts / 'inference_examples.json', {'saved_pipeline_reload_matches': True, 'examples': examples})
    print(json.dumps({'test': report, 'negative_threshold': lower, 'positive_threshold': upper,
                      'examples': examples}, indent=2), flush=True)


if __name__ == '__main__':
    main()
