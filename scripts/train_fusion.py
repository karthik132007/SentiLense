#!/usr/bin/env python3
"""Train a three-class stacking head over frozen TweetEval SVM and Cardiff RoBERTa."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sentiment_model.fusion import MODEL_ID, REVISION, RobertaExpert, fusion_features
from sentiment_model.preprocessing import normalize_text
from sentiment_model.svm import positive_probabilities
from scripts.train import save_json, sha256_file
from scripts.train_tweets import CHECKSUMS, download as download_tweets


def download_roberta(directory):
    """Pin the requested public checkpoint; convert its weights to safetensors."""
    if (directory / 'model.safetensors').exists():
        return
    import httpx
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for name in ('config.json', 'merges.txt', 'vocab.json', 'special_tokens_map.json', 'pytorch_model.bin'):
            path = directory / name
            if path.exists():
                continue
            temporary = path.with_suffix(path.suffix + '.partial')
            url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
            with client.stream('GET', url) as response:
                response.raise_for_status()
                with temporary.open('wb') as stream:
                    for block in response.iter_bytes(1024 * 1024):
                        stream.write(block)
            temporary.replace(path)
            print('Downloaded', name, path.stat().st_size, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(directory),
        local_files_only=True, trust_remote_code=False, use_safetensors=False)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    (directory / 'pytorch_model.bin').unlink()


def datasets(directory):
    """Keep all three labels, drop normalized conflicts and cross-split duplicates."""
    raw, meanings, offset = {}, defaultdict(set), 0
    for split in ('train', 'val', 'test'):
        texts = (directory / f'{split}_text.txt').read_text().splitlines()
        labels = (directory / f'{split}_labels.txt').read_text().splitlines()
        raw[split] = []
        for index, (text, label) in enumerate(zip(texts, labels, strict=True)):
            normalized = normalize_text(text)
            meanings[normalized].add(int(label))
            raw[split].append((text, int(label), offset + index, normalized))
        offset += len(texts)
    conflicts = {t for t, y in meanings.items() if len(y) > 1}
    seen, kept = set(), {}
    for split in ('test', 'val', 'train'):
        kept[split] = []
        for text, label, index, normalized in raw[split]:
            if normalized and normalized not in conflicts and normalized not in seen:
                seen.add(normalized)
                kept[split].append((text, label, index))
    return kept


def metrics(y, p):
    predicted = np.argmax(p, axis=1)
    return {'records': len(y), 'accuracy': float(accuracy_score(y, predicted)),
        'macro_f1': float(f1_score(y, predicted, average='macro')),
        'log_loss': float(log_loss(y, p, labels=[0, 1, 2])),
        'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
        'labels': ['negative', 'neutral', 'positive'],
        'classification_report': classification_report(y, predicted, target_names=['negative', 'neutral', 'positive'], output_dict=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/tweeteval')
    parser.add_argument('--svm-dir', type=Path, default=ROOT / 'model/svm' if (ROOT / 'model/svm').exists() else ROOT / 'model')
    parser.add_argument('--output-dir', type=Path, required=True, help='Train in a separate staging directory')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    if args.download:
        download_tweets(args.data)
    for name, checksum in CHECKSUMS.items():
        if sha256_file(args.data / name) != checksum:
            raise ValueError(f'Dataset checksum mismatch: {name}')
    model_dir, artifacts = args.output_dir / 'model', args.output_dir / 'artifacts/fusion'
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    svm_dir = model_dir / 'svm'
    if args.svm_dir.resolve() == model_dir.resolve():
        raise ValueError('Train into a separate output directory first; --svm-dir must contain the frozen binary expert')
    svm_dir.mkdir(exist_ok=True)
    for name in ('sentiment_pipeline.joblib', 'model_metadata.json'):
        shutil.copy2(args.svm_dir / name, svm_dir / name)
    source_evidence = (args.svm_dir.parent.parent / 'artifacts/svm' if args.svm_dir.name == 'svm'
                       else args.svm_dir.parent / 'artifacts')
    if source_evidence.exists():
        (args.output_dir / 'artifacts/svm').mkdir(parents=True, exist_ok=True)
        for name in ('evaluation.json', 'dataset_inspection.json', 'neutral_threshold_selection.json',
                     'heldout_predictions.npz', 'split_indices.npz', 'roc_curve.npz', 'inference_examples.json'):
            if (source_evidence / name).exists():
                shutil.copy2(source_evidence / name, args.output_dir / 'artifacts/svm' / name)
    if args.download:
        download_roberta(model_dir / 'roberta')
    from sentiment_model.inference import _load
    svm, _ = _load(str(svm_dir.resolve()))
    expert = RobertaExpert(model_dir / 'roberta')
    source = datasets(args.data)
    val = source['val']; vy = np.array([r[1] for r in val]); indices = np.arange(len(val))
    train, rest = train_test_split(indices, test_size=.5, stratify=vy, random_state=42)
    calibration, selection = train_test_split(rest, test_size=.5, stratify=vy[rest], random_state=42)
    def progress(done, total):
        if done % 256 == 0 or done == total:
            print(f'RoBERTa {done}/{total}; elapsed {time.monotonic()-started:.1f}s', flush=True)
    def expert_scores(name, rows):
        cache = artifacts / f'{name}_expert_scores.npz'
        texts = [r[0] for r in rows]; ids = np.array([r[2] for r in rows])
        identity = sha256_file(svm_dir / 'sentiment_pipeline.joblib') + sha256_file(model_dir / 'roberta/model.safetensors')
        if cache.exists():
            with np.load(cache) as old:
                if str(old['identity']) != identity or not np.array_equal(old['source_ids'], ids):
                    raise ValueError('Stale expert cache; remove it before rerunning')
                return old['svm_positive'], old['roberta']
        sp = positive_probabilities(svm, texts)
        rp = expert.probabilities(texts, progress=progress)
        np.savez_compressed(cache, identity=identity, source_ids=ids, svm_positive=sp, roberta=rp)
        return sp, rp
    sp, rp = expert_scores('validation', val)
    x = fusion_features(sp, rp)
    best, candidates = None, []
    for c in (.1, 1., 10.):
        head = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000, random_state=42))
        head.fit(x[train], vy[train])
        calibrated = CalibratedClassifierCV(FrozenEstimator(head), method='sigmoid')
        calibrated.fit(x[calibration], vy[calibration])
        probability = calibrated.predict_proba(x[selection])
        row = {'C': c, **metrics(vy[selection], probability)}
        candidates.append(row)
        rank = (row['macro_f1'], -row['log_loss'])
        if best is None or rank > best[0]:
            best = rank, calibrated, c
        print('Fusion selection:', c, row['accuracy'], row['macro_f1'], flush=True)
    _, head, selected_c = best
    joblib.dump(head, model_dir / 'fusion_head.joblib', compress=3)
    # No test features or labels influence the head, calibration or C selection.
    test = source['test']; test_y = np.array([r[1] for r in test])
    test_sp, test_rp = expert_scores('test', test)
    test_probability = head.predict_proba(fusion_features(test_sp, test_rp))
    report = {'test': metrics(test_y, test_probability), 'roberta_same_test': metrics(test_y, test_rp),
        'selection': metrics(vy[selection], head.predict_proba(x[selection])),
        'candidates': candidates, 'selected_C': selected_c,
        'protocol': 'Frozen SVM fitted on official train only; frozen upstream RoBERTa was finetuned for TweetEval. Split official validation 50/25/25 for fusion-head train/calibration/selection. No refit. Evaluate once on cleaned official three-class test. Upstream model selection may have used official validation; this is not a new unseen dataset for RoBERTa.',
        'test_neutral_included': True,
    }
    np.savez_compressed(artifacts / 'heldout_predictions.npz', test_labels=test_y,
        test_probabilities=test_probability, roberta_probabilities=test_rp,
        test_source_rows=np.array([r[2] for r in test]))
    np.savez_compressed(artifacts / 'split_indices.npz',
        train=np.array([val[i][2] for i in train]), calibration=np.array([val[i][2] for i in calibration]),
        validation=np.array([val[i][2] for i in selection]), test=np.array([r[2] for r in test]),
        svm_source_train=np.array([r[2] for r in source['train']]))
    save_json(artifacts / 'evaluation.json', report)
    metadata = {'model_family': 'svm_roberta_fusion',
        'model': 'Learned calibrated stacking of TF-IDF LinearSVC and Cardiff Twitter RoBERTa',
        'analysis_model': 'svm_roberta_fusion', 'dataset': 'TweetEval sentiment, all three classes for fusion',
        'roberta_model_id': MODEL_ID, 'roberta_revision': REVISION, 'roberta_frozen': True,
        'svm_frozen': True, 'feature_names': ['svm_log_odds', 'roberta_log_negative', 'roberta_log_neutral', 'roberta_log_positive'],
        'roberta_runtime': 'CPU float32; identical feature extraction during head training and deployment',
        'fusion_head': 'StandardScaler + multinomial LogisticRegression, disjoint sigmoid calibration',
        'selected_C': selected_c, 'classifier_classes': [0, 1, 2],
        'label_mapping': {'0': 'negative', '1': 'neutral', '2': 'positive'},
        'neutral_policy': 'Learned neutral class; label is argmax of three calibrated class probabilities',
        'confidence_definition': 'Probability of the selected class; no guarantee of correctness',
        'long_text_policy': 'Average RoBERTa probabilities over 512-token windows with 64-token overlap; SVM uses full text',
        'splits': {name: len(indices) for name, indices in [('train', train), ('calibration', calibration), ('validation', selection), ('test', test)]},
        'dataset_revision': '4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66', 'dataset_sha256': CHECKSUMS,
        'accuracy': report['test']['accuracy'], 'macro_f1': report['test']['macro_f1'],
        'trained_at_utc': datetime.now(timezone.utc).isoformat(), 'total_seconds': time.monotonic()-started,
        'files_sha256': {str(p.relative_to(model_dir)): sha256_file(p) for folder in ('svm','roberta')
            for p in (model_dir / folder).iterdir() if p.is_file()},
        'limitations': ['Frozen upstream model and SVM share TweetEval training provenance; experts are correlated',
            'Head calibration/selection uses small subsets of official validation; score calibration can shift by domain',
            'Tweet benchmark does not establish webpage/topic stance/other-language accuracy'],
    }
    metadata['files_sha256']['fusion_head.joblib'] = sha256_file(model_dir / 'fusion_head.joblib')
    import torch, transformers
    metadata['versions'] = {'torch': torch.__version__, 'transformers': transformers.__version__,
                            'python': sys.version.split()[0]}
    save_json(model_dir / 'model_metadata.json', metadata)
    print(json.dumps({'test': report['test'], 'roberta_same_test': report['roberta_same_test'], 'model_dir': str(model_dir)},indent=2),flush=True)


if __name__ == '__main__':
    main()
