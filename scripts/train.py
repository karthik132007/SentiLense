#!/usr/bin/env python3
"""Inspect, deduplicate, train, calibrate, evaluate, and save a linear SVM."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import scipy
import sklearn
import ftfy
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, brier_score_loss)
from sklearn.model_selection import train_test_split

from sentiment_model.preprocessing import normalize_text
from sentiment_model.svm import fit_svm, positive_probabilities

COLUMNS = ["target", "id", "date", "flag", "user", "text"]


def log(message):
    print(message, flush=True)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


@contextmanager
def open_bytes(path, member=None):
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            with archive.open(member) as stream:
                yield stream
    else:
        with path.open("rb") as stream:
            yield stream


def discover_source(requested):
    source = Path(requested).resolve()
    if source.is_dir():
        candidates = sorted(p for p in source.iterdir() if p.suffix.lower() in {".csv", ".zip"})
        if len(candidates) != 1:
            raise ValueError(f"Choose a CSV/ZIP explicitly; found {candidates}")
        source = candidates[0]
    if not source.is_file():
        raise FileNotFoundError(source)
    member = None
    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            candidates = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(candidates) != 1:
                raise ValueError(f"Expected one CSV in archive; found {candidates}")
            member = candidates[0]
    return source, member


def inspect_encoding(source, member):
    import codecs
    checks = {}
    for encoding in ["utf-8", "cp1252"]:
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        offset = 0
        try:
            with open_bytes(source, member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    decoder.decode(block)
                    offset += len(block)
                decoder.decode(b"", final=True)
            checks[encoding] = {"valid": True, "bytes_checked": offset}
        except UnicodeDecodeError as exc:
            checks[encoding] = {"valid": False, "first_invalid_byte_approximately": offset + exc.start,
                                "reason": exc.reason}
    encoding = "utf-8" if checks["utf-8"]["valid"] else "iso-8859-1"
    return encoding, checks


def inspect_and_prepare(source, member, encoding):
    import csv
    missing = Counter({column: 0 for column in COLUMNS})
    label_counts, field_counts, flags = Counter(), Counter(), Counter()
    unique_rows, raw_texts = set(), set()
    # normalized text -> (label, original text, source row). None means conflict.
    records = {}
    conflict_counts = Counter()
    duplicate_rows = duplicate_texts = normalized_duplicates = 0
    changed = empty_normalized = missing_binary_text = excluded_other_labels = 0
    invalid_ids = invalid_dates = total = 0
    date_pattern = re.compile(r"^[A-Za-z]{3} [A-Za-z]{3} \d{2} \d{2}:\d{2}:\d{2} [A-Z]+ \d{4}$")
    first_row = None
    with open_bytes(source, member) as stream:
        reader = csv.reader(io.TextIOWrapper(stream, encoding=encoding, errors="strict", newline=""))
        for row_index, row in enumerate(reader):
            if first_row is None:
                first_row = row[:5] + ["<tweet text present>" if len(row) == 6 and row[5] else "<missing>"]
                if [value.lower() for value in row] == COLUMNS:
                    raise ValueError("This script expects verified headerless Sentiment140 input")
            total += 1
            field_counts[len(row)] += 1
            if len(row) != 6:
                raise ValueError(f"Source row {row_index} has {len(row)} fields, expected 6")
            label, identifier, date, flag, user, text = row
            label_counts[label] += 1
            flags[flag] += 1
            invalid_ids += not identifier.isdigit()
            invalid_dates += not bool(date_pattern.fullmatch(date))
            for column, value in zip(COLUMNS, row):
                missing[column] += not value.strip()
            digest = hashlib.blake2b(json.dumps(row, ensure_ascii=False).encode("utf-8"), digest_size=16).digest()
            duplicate_rows += digest in unique_rows
            unique_rows.add(digest)
            duplicate_texts += text in raw_texts
            raw_texts.add(text)
            if label not in {"0", "4"}:
                excluded_other_labels += 1
                continue
            if not text.strip():
                missing_binary_text += 1
                continue
            normalized = normalize_text(text)
            changed += normalized != text
            if not normalized:
                empty_normalized += 1
                continue
            if normalized in records:
                normalized_duplicates += 1
                previous = records[normalized]
                if previous is None:
                    conflict_counts[normalized] += 1
                elif previous[0] != label:
                    records[normalized] = None
                    conflict_counts[normalized] = 2
            else:
                records[normalized] = (label, text, row_index)
            if total % 200000 == 0:
                log(f"Inspected/preprocessed {total:,} source records")
    if invalid_ids or invalid_dates:
        raise ValueError(f"Unexpected schema: {invalid_ids} invalid IDs, {invalid_dates} invalid dates")
    kept = [record for record in records.values() if record is not None]
    cleaned_labels = Counter(record[0] for record in kept)
    inspection = {
        "rows": total, "header_present": False, "columns": COLUMNS,
        "column_interpretation": "Semantic names inferred from all rows' six-field structure, numeric labels/IDs, date format, and flag/user/text positions; names are not stored in a header.",
        "first_row_non_text_fields": first_row, "field_count_distribution": dict(field_counts),
        "invalid_numeric_ids": invalid_ids, "invalid_date_formats": invalid_dates,
        "labels": sorted(label_counts), "label_distribution": dict(label_counts),
        "label_percentages": {k: v / total for k, v in label_counts.items()},
        "missing_values": dict(missing), "missing_definition": "Empty or whitespace-only fields; literal NA-like tweet strings are retained.",
        "flag_distribution": dict(flags), "exact_duplicate_rows": duplicate_rows,
        "duplicate_raw_text_rows": duplicate_texts,
        "duplicate_normalized_text_rows": normalized_duplicates,
        "conflicting_normalized_text_groups": len(conflict_counts),
        "normalization_changed_rows": changed,
        "excluded_nonbinary_label_rows": excluded_other_labels,
        "excluded_missing_binary_text_rows": missing_binary_text,
        "excluded_empty_normalized_rows": empty_normalized,
        "records_after_cleaning": len(kept), "records_removed": total - len(kept),
        "cleaned_label_distribution": dict(cleaned_labels),
        "deduplication": "Keep first occurrence of identical normalized text with consistent labels; remove every normalized group with conflicting labels BEFORE splitting.",
    }
    return [r[1] for r in kept], np.array([0 if r[0] == "0" else 1 for r in kept], dtype=np.uint8), np.array([r[2] for r in kept], dtype=np.int64), inspection


def binary_metrics(y, probability):
    prediction = (probability >= 0.5).astype(np.uint8)
    return {
        "records": len(y), "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction)),
        "recall": float(recall_score(y, prediction)),
        "f1": float(f1_score(y, prediction)), "f1_macro": float(f1_score(y, prediction, average="macro")),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "metric_positive_class": "positive (1)", "binary_decision_threshold": 0.5,
        "confusion_matrix_labels": ["negative", "positive"],
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, prediction, labels=[0, 1],
                                target_names=["negative", "positive"], output_dict=True, zero_division=0),
    }


def select_thresholds(y, p, target_precision):
    candidates = []
    minimum = max(100, int(0.05 * len(y)))
    for threshold in np.arange(0.01, 1.0, 0.01):
        threshold = round(float(threshold), 2)
        positive = threshold > 0.5
        if threshold == 0.5:
            continue
        selected = p >= threshold if positive else p <= threshold
        count = int(selected.sum())
        precision = float(np.mean(y[selected] == int(positive))) if count else 0.0
        candidates.append({"class": "positive" if positive else "negative", "threshold": threshold,
                           "accepted_records": count, "precision": precision,
                           "eligible": count >= minimum and precision >= target_precision})
    choices = {}
    for label in ["negative", "positive"]:
        feasible = [r for r in candidates if r["class"] == label and r["eligible"]]
        if not feasible:
            raise ValueError(f"No {label} threshold reaches validation precision {target_precision} with {minimum} predictions; choose an attainable --target-precision")
        choices[label] = max(feasible, key=lambda row: row["accepted_records"])
    return choices["negative"]["threshold"], choices["positive"]["threshold"], {
        "selection_dataset": "validation only", "target_precision_per_accepted_class": target_precision,
        "minimum_accepted_records_per_class": minimum,
        "objective": "Maximize accepted count for each class subject to validation precision >= target; search 0.01 probability increments, negative < 0.5 and positive > 0.5.",
        "selected": choices, "candidates": candidates,
        "meaning": "Neutral is abstention/uncertainty, not a trained neutral label. Binary data cannot measure true neutral precision/recall/F1. Binary probability estimates are not guaranteed certainty.",
    }


def abstention_metrics(y, p, lower, upper):
    negative, positive = p <= lower, p >= upper
    accepted = negative | positive
    return {
        "records": len(y), "negative_records": int(negative.sum()), "positive_records": int(positive.sum()),
        "neutral_uncertain_records": int((~accepted).sum()), "coverage": float(accepted.mean()),
        "accepted_accuracy": float(np.mean((p[accepted] >= 0.5) == y[accepted])) if accepted.any() else None,
        "negative_precision": float(np.mean(y[negative] == 0)) if negative.any() else None,
        "positive_precision": float(np.mean(y[positive] == 1)) if positive.any() else None,
        "neutral_is_abstention": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="/data" if Path("/data").exists() else str(ROOT / "data"))
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--max-records", type=int, default=None, help="Optional reproducible stratified smoke-test subset")
    parser.add_argument("--max-features", type=int, default=300000)
    parser.add_argument("--char-features", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-precision", type=float, default=0.90)
    args = parser.parse_args()
    if not 0.5 < args.target_precision < 1:
        parser.error("--target-precision must be between 0.5 and 1")
    if args.max_records is not None and args.max_records < 10000:
        parser.error("--max-records must be >= 10000 for meaningful threshold selection")
    started = time.monotonic()
    model_dir, artifacts = args.output_dir / "model", args.output_dir / "artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    source, member = discover_source(args.data)
    encoding, encoding_checks = inspect_encoding(source, member)
    log(f"Source: {source}, archive member: {member}, encoding: {encoding}")
    texts, y, source_rows, inspection = inspect_and_prepare(source, member, encoding)
    inspection.update({"source_path": str(source), "archive_member": member, "encoding": encoding,
                       "encoding_checks": encoding_checks, "source_sha256": sha256_file(source),
                       "encoding_note": "Strict UTF-8 validation across the stream; use lossless ISO-8859-1 for non-UTF-8 input, repairing mojibake and C1 controls downstream. Single-byte encodings cannot always be uniquely distinguished from bytes alone."})
    save_json(artifacts / "dataset_inspection.json", inspection)
    log(f"Cleaned data: {len(y):,} records; labels {inspection['cleaned_label_distribution']}")
    indices = np.arange(len(y))
    if args.max_records and args.max_records < len(y):
        indices, _ = train_test_split(indices, train_size=args.max_records, stratify=y, random_state=args.seed)
    train_validation, test = train_test_split(indices, test_size=0.20, stratify=y[indices], random_state=args.seed)
    train, validation = train_test_split(train_validation, test_size=0.125, stratify=y[train_validation], random_state=args.seed)
    calibration, validation = train_test_split(validation, test_size=0.5, stratify=y[validation], random_state=args.seed)
    # Unique normalized texts established before split; row indices permit audit.
    np.savez_compressed(artifacts / "split_indices.npz", train=source_rows[train], calibration=source_rows[calibration], validation=source_rows[validation], test=source_rows[test])
    splits = {name: {"records": len(split), "negative": int((y[split] == 0).sum()), "positive": int((y[split] == 1).sum())}
              for name, split in [("train", train), ("calibration", calibration), ("validation", validation), ("test", test)]}
    log(f"Stratified 70/5/5/20 train/calibration/validation/test splits: {splits}")
    pipeline, validation_p, svm_details = fit_svm(
        [texts[i] for i in train], y[train], [texts[i] for i in calibration], y[calibration],
        [texts[i] for i in validation], y[validation], word_features=args.max_features,
        char_features=args.char_features, seed=args.seed)
    fit_seconds = svm_details['fit_seconds']
    lower, upper, threshold_report = select_thresholds(y[validation], validation_p, args.target_precision)
    save_json(artifacts / "neutral_threshold_selection.json", threshold_report)
    log(f"Validation-selected thresholds: negative <= {lower}, positive >= {upper}")
    test_p = positive_probabilities(pipeline, [texts[i] for i in test])
    test_metrics = binary_metrics(y[test], test_p)
    evaluation = {
        "test": test_metrics, "validation": binary_metrics(y[validation], validation_p),
        "uncertainty_policy": {"negative_threshold": lower, "positive_threshold": upper,
                               "validation": abstention_metrics(y[validation], validation_p, lower, upper),
                               "test": abstention_metrics(y[test], test_p, lower, upper)},
        "splits": splits, "seed": args.seed,
        "evaluation_protocol": "Deduplicate before stratified 70/5/5/20 split. Fit vocabulary, IDF, and SVM only on train. Fit sigmoid only on calibration. Select C and uncertainty thresholds only on validation. Evaluate once on untouched test. No refit on held-out records.",
        "svm_training": svm_details,
    }
    save_json(artifacts / "evaluation.json", evaluation)
    (artifacts / "classification_report.txt").write_text(classification_report(y[test], test_p >= 0.5,
        target_names=["negative", "positive"], digits=6, zero_division=0))
    np.savetxt(artifacts / "confusion_matrix.csv", test_metrics["confusion_matrix"], delimiter=",", fmt="%d",
               header="rows=actual negative/positive; columns=predicted negative/positive", comments="# ")
    np.savez_compressed(artifacts / "heldout_predictions.npz", validation_source_rows=source_rows[validation],
        validation_labels=y[validation], validation_positive_probability=validation_p,
        test_source_rows=source_rows[test], test_labels=y[test], test_positive_probability=test_p)
    fpr, tpr, thresholds = roc_curve(y[test], test_p)
    # Store complete ROC coordinates for plotting/recomputing without reloading tweets.
    np.savez_compressed(artifacts / "roc_curve.npz", false_positive_rate=fpr, true_positive_rate=tpr, thresholds=thresholds)
    model_path = model_dir / "sentiment_pipeline.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    metadata = {
        "model": "Word/character TF-IDF + calibrated LinearSVC (linear SVM)",
        "model_family": "svm",
        "dataset": "Sentiment140", "dataset_file": member or source.name,
        "dataset_archive": str(source), "dataset_sha256": inspection["source_sha256"], "encoding": encoding,
        "source_records": inspection["rows"], "cleaned_records_available": len(y),
        "records_used": len(indices), "records_trained": len(train), "splits": splits,
        "accuracy": test_metrics["accuracy"], "precision": test_metrics["precision"], "recall": test_metrics["recall"],
        "f1": test_metrics["f1"], "f1_macro": test_metrics["f1_macro"], "roc_auc": test_metrics["roc_auc"],
        "negative_threshold": lower, "positive_threshold": upper,
        "neutral_policy": threshold_report["meaning"], "threshold_selection": threshold_report["objective"],
        "target_validation_precision": args.target_precision,
        "confidence_definition": "max(P(positive), P(negative)); binary polarity confidence, never a learned neutral probability",
        "zero_vocabulary_policy": "Return neutral with both binary scores 0.5 when no features are recognized",
        "label_mapping": {"0": "negative", "4": "positive"}, "classifier_classes": [0, 1],
        "preprocessing": "NFC + ftfy repair, HTML unescape, lowercase, URL/mention placeholders, whitespace/control cleanup; preserve negation, emoticons, emojis, hashtags, punctuation; no stopword removal",
        "deduplication": inspection["deduplication"],
        "random_seed": args.seed, **svm_details,
        "tfidf_parameters": svm_details['feature_parameters'],
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__, "joblib": joblib.__version__, "ftfy": ftfy.__version__},
        "compute": "CPU; no local CUDA GPU or callable VS Code/Colab kernel-control tool available",
        "fit_seconds": fit_seconds, "total_seconds": time.monotonic() - started,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_sha256": sha256_file(model_path),
        "limitations": ["Binary, weakly labeled historical English tweets; no true neutral training/evaluation class",
                        "Random tweet split does not estimate user-disjoint or chronological generalization",
                        "Sigmoid calibration on held-out tweets does not establish calibration on articles or other domains",
                        "Domain shift, sarcasm, and out-of-domain texts can still yield confident errors"],
    }
    save_json(model_dir / "model_metadata.json", metadata)
    loaded = joblib.load(model_path)
    probe = [texts[i] for i in test[:100]]
    np.testing.assert_allclose(loaded.predict_proba(probe), pipeline.predict_proba(probe), rtol=0, atol=0)
    from sentiment_model.inference import predict_sentiment, _load
    _load.cache_clear()
    examples = [{"text": text, "prediction": predict_sentiment(text, model_dir=model_dir)} for text in
                ["I absolutely love this!", "I hate this product.", "The product arrived yesterday."]]
    save_json(artifacts / "inference_examples.json", {"saved_pipeline_reload_matches": True, "examples": examples})
    log(json.dumps({"test_metrics": {k: v for k, v in test_metrics.items() if k != "classification_report"},
                    "uncertainty_policy_test": evaluation["uncertainty_policy"]["test"], "examples": examples}, indent=2))
    log(f"Saved pipeline: {model_path}; total elapsed {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
