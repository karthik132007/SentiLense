#!/usr/bin/env python3
"""Verify a staged fusion and preserve the prior deployment before replacing it."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sentiment_model.inference import _load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trained-root', type=Path, required=True)
    args = parser.parse_args()
    source = (args.trained_root / 'model').resolve()
    target = ROOT / 'model'
    if source == target.resolve():
        parser.error('Use a separate staging root')
    model, metadata = _load(str(source))
    if metadata.get('model_family') != 'svm_roberta_fusion':
        raise ValueError('Staged model must be a fusion')
    evaluation_path = args.trained_root / 'artifacts/fusion/evaluation.json'
    evaluation = json.loads(evaluation_path.read_text())
    if evaluation['test']['records'] != metadata['splits']['test']:
        raise ValueError('Model and evaluation must belong to the same training run')
    # Check actual inference, not just manifest files. No label is forced here.
    probe = model.predict_many(['I absolutely love this!', 'I hate this product.'])
    if [p['sentiment'] for p in probe] != ['positive', 'negative']:
        raise ValueError('Staged fusion failed basic deployment probes')
    # Preserve audit evidence for the frozen binary expert independently of
    # the active three-class fusion's report.
    staged_svm_evidence = args.trained_root / 'artifacts/svm'
    binary_evidence = ROOT / 'artifacts/svm'
    if staged_svm_evidence.exists():
        shutil.copytree(staged_svm_evidence, binary_evidence, dirs_exist_ok=True)
    elif json.loads((target / 'model_metadata.json').read_text()).get('model_family') == 'svm':
        binary_evidence.mkdir(parents=True, exist_ok=True)
        for name in ('evaluation.json', 'dataset_inspection.json', 'neutral_threshold_selection.json',
                     'heldout_predictions.npz', 'split_indices.npz', 'roc_curve.npz', 'inference_examples.json'):
            if (ROOT / 'artifacts' / name).exists():
                shutil.copy2(ROOT / 'artifacts' / name, binary_evidence / name)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup = ROOT / 'artifacts' / f'baseline-before-fusion-{stamp}'
    replacement = ROOT / f'model.next-{stamp}'
    shutil.copytree(source, replacement)
    if (target / 'README.md').exists():
        shutil.copy2(target / 'README.md', replacement / 'README.md')
    if (target / 'reviews').exists():
        shutil.copytree(target / 'reviews', replacement / 'reviews')
    backup.mkdir(parents=True)
    target.rename(backup / 'model')
    try:
        replacement.rename(target)
    except BaseException:
        (backup / 'model').rename(target)
        raise
    for name in ('evaluation.json', 'dataset_inspection.json', 'neutral_threshold_selection.json',
                 'heldout_predictions.npz', 'split_indices.npz', 'inference_examples.json', 'sentiment_regressions.json'):
        path = ROOT / 'artifacts' / name
        if path.exists():
            (backup / 'artifacts').mkdir(exist_ok=True)
            shutil.copy2(path, backup / 'artifacts' / name)
    evidence = ROOT / 'artifacts/fusion'
    if evidence.exists():
        evidence.rename(backup / 'fusion-evidence')
    shutil.copytree(args.trained_root / 'artifacts/fusion', evidence)
    for name in ('evaluation.json', 'heldout_predictions.npz', 'split_indices.npz'):
        shutil.copy2(evidence / name, ROOT / 'artifacts' / name)
    print(json.dumps({'installed': str(target), 'previous_model': str(backup / 'model'),
                      'restart_required': True}, indent=2))


if __name__ == '__main__':
    main()
