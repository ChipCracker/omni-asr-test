#!/usr/bin/env python3
"""Analyze ASR evaluation results from JSON files."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_results(file_path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_header(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    print()
    print(char * 60)
    print(title)
    print(char * 60)


def print_overview(data: Dict[str, Any]) -> None:
    """Print evaluation overview."""
    print_header("OVERVIEW")
    print(f"Model:           {data.get('model', 'N/A')}")
    print(f"Dataset:         {data.get('dataset', 'N/A')}")
    print(f"Language:        {data.get('language', 'N/A')}")
    print(f"Timestamp:       {data.get('timestamp', 'N/A')}")
    print(f"Total Samples:   {data.get('num_samples', 0)}")
    print(f"Skipped:         {data.get('num_skipped', 0)}")

    print_header("AGGREGATE METRICS", "-")
    results = data.get("results", {})

    dialect_res = results.get("dialect_reference", {})
    if dialect_res:
        print("Dialect Reference:")
        print(f"  WER:           {dialect_res.get('wer', 0):.2%}")
        print(f"  CER:           {dialect_res.get('cer', 0):.2%}")
        print(f"  Substitutions: {dialect_res.get('substitutions', 0)}")
        print(f"  Deletions:     {dialect_res.get('deletions', 0)}")
        print(f"  Insertions:    {dialect_res.get('insertions', 0)}")

    ort_res = results.get("ort_reference", {})
    if ort_res and ort_res.get("wer") is not None:
        print()
        print("Standard Orthography Reference:")
        print(f"  WER:           {ort_res.get('wer', 0):.2%}")
        print(f"  CER:           {ort_res.get('cer', 0):.2%}")
        print(f"  Substitutions: {ort_res.get('substitutions', 0)}")
        print(f"  Deletions:     {ort_res.get('deletions', 0)}")
        print(f"  Insertions:    {ort_res.get('insertions', 0)}")


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute descriptive statistics for a list of values."""
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}

    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0,
    }


def print_distribution(per_sample: List[Dict[str, Any]]) -> None:
    """Print WER/CER distribution statistics."""
    print_header("DISTRIBUTION STATISTICS")

    dialect_wers = [s["dialect_wer"] for s in per_sample if s.get("dialect_wer") is not None]
    dialect_cers = [s["dialect_cer"] for s in per_sample if s.get("dialect_cer") is not None]

    if dialect_wers:
        stats = compute_statistics(dialect_wers)
        print("Dialect WER Distribution:")
        print(f"  Min:           {stats['min']:.2%}")
        print(f"  Max:           {stats['max']:.2%}")
        print(f"  Mean:          {stats['mean']:.2%}")
        print(f"  Median:        {stats['median']:.2%}")
        print(f"  Std Dev:       {stats['std']:.2%}")

    if dialect_cers:
        print()
        stats = compute_statistics(dialect_cers)
        print("Dialect CER Distribution:")
        print(f"  Min:           {stats['min']:.2%}")
        print(f"  Max:           {stats['max']:.2%}")
        print(f"  Mean:          {stats['mean']:.2%}")
        print(f"  Median:        {stats['median']:.2%}")
        print(f"  Std Dev:       {stats['std']:.2%}")

    # ORT statistics if available
    ort_wers = [s["ort_wer"] for s in per_sample if s.get("ort_wer") is not None]
    ort_cers = [s["ort_cer"] for s in per_sample if s.get("ort_cer") is not None]

    if ort_wers:
        print()
        stats = compute_statistics(ort_wers)
        print("ORT WER Distribution:")
        print(f"  Min:           {stats['min']:.2%}")
        print(f"  Max:           {stats['max']:.2%}")
        print(f"  Mean:          {stats['mean']:.2%}")
        print(f"  Median:        {stats['median']:.2%}")
        print(f"  Std Dev:       {stats['std']:.2%}")

    if ort_cers:
        print()
        stats = compute_statistics(ort_cers)
        print("ORT CER Distribution:")
        print(f"  Min:           {stats['min']:.2%}")
        print(f"  Max:           {stats['max']:.2%}")
        print(f"  Mean:          {stats['mean']:.2%}")
        print(f"  Median:        {stats['median']:.2%}")
        print(f"  Std Dev:       {stats['std']:.2%}")


def print_top_samples(per_sample: List[Dict[str, Any]], top_n: int) -> None:
    """Print best and worst performing samples."""
    print_header(f"TOP {top_n} BEST SAMPLES (Lowest WER)")

    sorted_by_wer = sorted(
        [s for s in per_sample if s.get("dialect_wer") is not None],
        key=lambda x: x["dialect_wer"]
    )

    for i, sample in enumerate(sorted_by_wer[:top_n], 1):
        print(f"\n{i}. Index: {sample.get('index', 'N/A')}")
        print(f"   WER: {sample['dialect_wer']:.2%}, CER: {sample.get('dialect_cer', 0):.2%}")
        print(f"   Duration: {sample.get('duration', 0):.2f}s")
        if sample.get("speaker_id"):
            print(f"   Speaker: {sample['speaker_id']}")
        audio_path = sample.get("audio_path", "")
        if audio_path:
            print(f"   Audio: {Path(audio_path).name}")

    print_header(f"TOP {top_n} WORST SAMPLES (Highest WER)")

    for i, sample in enumerate(sorted_by_wer[-top_n:][::-1], 1):
        print(f"\n{i}. Index: {sample.get('index', 'N/A')}")
        print(f"   WER: {sample['dialect_wer']:.2%}, CER: {sample.get('dialect_cer', 0):.2%}")
        print(f"   Duration: {sample.get('duration', 0):.2f}s")
        if sample.get("speaker_id"):
            print(f"   Speaker: {sample['speaker_id']}")
        audio_path = sample.get("audio_path", "")
        if audio_path:
            print(f"   Audio: {Path(audio_path).name}")


def print_speaker_analysis(per_sample: List[Dict[str, Any]]) -> None:
    """Print WER statistics per speaker."""
    print_header("SPEAKER ANALYSIS")

    # Group samples by speaker
    speaker_samples: Dict[str, List[Dict[str, Any]]] = {}
    for sample in per_sample:
        speaker_id = sample.get("speaker_id")
        if speaker_id:
            if speaker_id not in speaker_samples:
                speaker_samples[speaker_id] = []
            speaker_samples[speaker_id].append(sample)

    if not speaker_samples:
        print("No speaker information available.")
        return

    # Compute per-speaker statistics
    speaker_stats: List[Dict[str, Any]] = []
    for speaker_id, samples in speaker_samples.items():
        wers = [s["dialect_wer"] for s in samples if s.get("dialect_wer") is not None]
        cers = [s["dialect_cer"] for s in samples if s.get("dialect_cer") is not None]

        if wers:
            speaker_stats.append({
                "speaker_id": speaker_id,
                "num_samples": len(samples),
                "mean_wer": statistics.mean(wers),
                "mean_cer": statistics.mean(cers) if cers else 0,
                "min_wer": min(wers),
                "max_wer": max(wers),
            })

    # Sort by mean WER
    speaker_stats.sort(key=lambda x: x["mean_wer"])

    print(f"\n{'Speaker':<20} {'Samples':>8} {'Mean WER':>10} {'Mean CER':>10} {'Min WER':>10} {'Max WER':>10}")
    print("-" * 70)

    for stats in speaker_stats:
        print(
            f"{stats['speaker_id']:<20} "
            f"{stats['num_samples']:>8} "
            f"{stats['mean_wer']:>9.2%} "
            f"{stats['mean_cer']:>9.2%} "
            f"{stats['min_wer']:>9.2%} "
            f"{stats['max_wer']:>9.2%}"
        )

    print("-" * 70)
    print(f"Total speakers: {len(speaker_stats)}")


def print_examples(per_sample: List[Dict[str, Any]], top_n: int) -> None:
    """Print detailed examples of transcriptions."""
    print_header("EXAMPLE TRANSCRIPTIONS")

    sorted_by_wer = sorted(
        [s for s in per_sample if s.get("dialect_wer") is not None],
        key=lambda x: x["dialect_wer"]
    )

    # Best examples
    print("\n--- BEST TRANSCRIPTIONS ---")
    for i, sample in enumerate(sorted_by_wer[:top_n], 1):
        print(f"\n[{i}] Index: {sample.get('index', 'N/A')} | WER: {sample['dialect_wer']:.2%}")
        print(f"REF: {sample.get('dialect_reference', '')}")
        print(f"HYP: {sample.get('hypothesis', '')}")

    # Worst examples
    print("\n--- WORST TRANSCRIPTIONS ---")
    for i, sample in enumerate(sorted_by_wer[-top_n:][::-1], 1):
        print(f"\n[{i}] Index: {sample.get('index', 'N/A')} | WER: {sample['dialect_wer']:.2%}")
        print(f"REF: {sample.get('dialect_reference', '')}")
        print(f"HYP: {sample.get('hypothesis', '')}")

    # Median examples (representative)
    print("\n--- MEDIAN TRANSCRIPTIONS ---")
    mid_idx = len(sorted_by_wer) // 2
    start_idx = max(0, mid_idx - top_n // 2)
    end_idx = min(len(sorted_by_wer), start_idx + top_n)

    for i, sample in enumerate(sorted_by_wer[start_idx:end_idx], 1):
        print(f"\n[{i}] Index: {sample.get('index', 'N/A')} | WER: {sample['dialect_wer']:.2%}")
        print(f"REF: {sample.get('dialect_reference', '')}")
        print(f"HYP: {sample.get('hypothesis', '')}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze ASR evaluation results from JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "result_file",
        type=Path,
        help="Path to the evaluation results JSON file",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of best/worst samples to display",
    )
    parser.add_argument(
        "--show-speakers",
        action="store_true",
        help="Show per-speaker statistics",
    )
    parser.add_argument(
        "--show-examples",
        action="store_true",
        help="Show example transcriptions (reference vs. hypothesis)",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if not args.result_file.exists():
        print(f"Error: File not found: {args.result_file}", file=sys.stderr)
        return 1

    try:
        data = load_results(args.result_file)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        return 1

    per_sample = data.get("per_sample", [])

    # Always show overview
    print_overview(data)

    # Always show distribution statistics
    if per_sample:
        print_distribution(per_sample)
        print_top_samples(per_sample, args.top_n)

    # Optional: speaker analysis
    if args.show_speakers:
        print_speaker_analysis(per_sample)

    # Optional: example transcriptions
    if args.show_examples:
        print_examples(per_sample, args.top_n)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
