#!/usr/bin/env python3
"""Paraphrase note-level ``PatientDemographics`` text reproducibly.

This is the portable, replication-ready copy of ``3_paraphrasing.py``. It is
compatible with ``2_gen_pan_ehr_label_replication.py`` and has no hard-coded
data path, GPU index, or import-time model execution.

The default model and core generation settings preserve the original experiment:
``humarin/chatgpt_paraphraser_on_T5_base``, sampling enabled, three beams,
temperature 0.30, top-p 0.90, and top-k 50. Because sampled GPU generation can
vary across hardware/library versions, use ``--no-do-sample`` when exact
deterministic decoding is more important than matching the original sampling
method. This copy paraphrases each unique demographics string once by default
for efficiency; pass ``--no-deduplicate-text --batch-size 1 --shuffle`` to
mirror the original row-by-row shuffled execution. Every run writes a JSON
file containing the complete configuration.

Dependencies
------------
Python 3.10+, pandas, torch, transformers, and sentencepiece. The Hugging Face
model is downloaded on first use unless it is already cached.

Examples
--------
Original sampling method on the first visible GPU:

python 3_paraphrasing_replication.py \
  --input replication_output/CN_EHR_pancreas_min2_labeled.csv \
  --output replication_output/CN_EHR_pancreas_min2_paraphrase.csv \
  --device cuda:0 --batch-size 16

Deterministic beam-search replication:

python 3_paraphrasing_replication.py \
  --input replication_output/CN_EHR_pancreas_min2_labeled.csv \
  --output replication_output/CN_EHR_pancreas_min2_paraphrase.csv \
  --device cuda:0 --no-do-sample --num-return-sequences 1

Resume an interrupted run from its last atomic checkpoint with ``--resume``.
Use ``--copy-input-text`` only for dependency-free pipeline smoke testing; it
does not run the paraphrasing model and is recorded in the run metadata.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"
DEFAULT_BAD_TERMS = (
    "skin",
    "Caucase",
    "hetero",
    "homo",
    "hippie",
    "diaper",
    "substance",
    "non-native",
    "non-dairy",
    "Rainy",
)
DEMOGRAPHIC_FIELDS = (
    ("tobacco_group", "Tobacco smoking status"),
    ("BMI_group", "BMI number"),
    ("depression_group", "Depression status"),
    ("alcohol_group", "Alcohol consumption description"),
    ("nutrition_group", "Nutrition description"),
    ("physical_activity_group", "Physical activity description"),
)
PREFERRED_RESUME_KEYS = (
    "CLINICAL_DOCUMENT_FPK",
    "PATIENT_CLINIC_NUMBER",
    "CLINICAL_DOCUMENT_ORIGINAL_DTM",
    "PatientDemographics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paraphrase PatientDemographics with a T5 paraphraser.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Labeled input CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV. Defaults to INPUT_STEM_paraphrase.csv beside the input.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--text-column", default="PatientDemographics")
    parser.add_argument(
        "--output-column",
        help="Defaults to ParaphraseDemographics_TEMPERATURE.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        help="Optional CUDA_VISIBLE_DEVICES value set before importing torch.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.30)
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--num-return-sequences", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--max-input-length", type=int, default=400)
    parser.add_argument("--max-output-length", type=int, default=400)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample tokens (original method) or use deterministic beam search.",
    )
    parser.add_argument(
        "--deduplicate-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate one paraphrase per unique demographics string and reuse it.",
    )
    parser.add_argument(
        "--rebuild-text",
        action="store_true",
        help="Rebuild PatientDemographics from the six group columns.",
    )
    parser.add_argument(
        "--bad-term",
        action="append",
        default=None,
        help="Additional forbidden output term; may be repeated.",
    )
    parser.add_argument("--no-default-bad-terms", action="store_true")
    parser.add_argument(
        "--checkpoint-every-batches",
        type=int,
        default=100,
        help="Atomically save progress every N model batches; 0 disables checkpoints.",
    )
    parser.add_argument("--preview-rows", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Optional row limit for a test run.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle rows using --seed.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output CSV.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-input-text",
        action="store_true",
        help="Smoke-test mode: copy source text instead of loading the model.",
    )
    return parser.parse_args()


def clean_value(value: object, default: str = "unknown") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def build_demographics_text(frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
    missing = [column for column, _label in DEMOGRAPHIC_FIELDS if column not in frame]
    if missing:
        raise ValueError(
            f"Cannot build {text_column!r}; input is missing group columns: {missing}"
        )
    out = frame.copy()
    text = pd.Series("Patient information: ", index=out.index, dtype="string")
    for column, label in DEMOGRAPHIC_FIELDS:
        values = out[column].map(clean_value)
        text = text + f"{label}: " + values + ". "
    out[text_column] = text.str.strip()
    return out


def prepare_input(args: argparse.Namespace) -> pd.DataFrame:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    frame = pd.read_csv(args.input, dtype=str, low_memory=False)
    if args.rebuild_text or args.text_column not in frame:
        frame = build_demographics_text(frame, args.text_column)
    frame[args.text_column] = frame[args.text_column].map(
        lambda value: clean_value(value, default="")
    )
    if args.shuffle:
        frame = frame.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be nonnegative")
        frame = frame.head(args.limit).copy()
    return frame


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output
    return args.input.with_name(f"{args.input.stem}_paraphrase.csv")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def load_resume_values(
    frame: pd.DataFrame,
    output_path: Path,
    output_column: str,
) -> pd.Series:
    previous = pd.read_csv(output_path, dtype=str, low_memory=False)
    if len(previous) != len(frame):
        raise ValueError(
            f"Cannot resume: input has {len(frame)} rows but existing output has {len(previous)}."
        )
    if output_column not in previous:
        raise ValueError(f"Cannot resume: existing output lacks {output_column!r}.")
    keys = [key for key in PREFERRED_RESUME_KEYS if key in frame and key in previous]
    if not keys:
        raise ValueError(
            "Cannot safely resume because input/output share none of the expected identity columns."
        )
    for key in keys:
        left = frame[key].fillna("").astype(str).reset_index(drop=True)
        right = previous[key].fillna("").astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"Cannot resume: row identity/order differs in column {key!r}.")
    return previous[output_column].fillna("").astype(str).reset_index(drop=True)


def make_bad_words_ids(tokenizer: Any, terms: list[str]) -> list[list[int]]:
    return [
        token_ids
        for token_ids in (
            tokenizer.encode(term, add_special_tokens=False) for term in terms
        )
        if token_ids
    ]


def resolve_device(requested: str, torch_module: Any) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
    return requested


def load_model(model_name: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def generate_batch(
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: str,
    args: argparse.Namespace,
    bad_words_ids: list[list[int]],
) -> list[str]:
    import torch

    prompts = [f"Rephrase: {text}" for text in texts]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_input_length,
    ).to(device)
    generation: dict[str, Any] = {
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "max_length": args.max_output_length,
        "bad_words_ids": bad_words_ids or None,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation.update({
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        })
    with torch.inference_mode():
        generated = model.generate(**encoded, **generation)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    step = args.num_return_sequences
    return [decoded[index * step] for index in range(len(texts))]


def pending_generation_items(
    frame: pd.DataFrame,
    text_column: str,
    output_column: str,
    deduplicate: bool,
) -> list[tuple[list[int], str]]:
    pending = frame[output_column].fillna("").astype(str).str.strip().eq("")
    pending &= frame[text_column].fillna("").astype(str).str.strip().ne("")
    if deduplicate:
        grouped: dict[str, list[int]] = {}
        for index, text in frame.loc[pending, text_column].items():
            grouped.setdefault(str(text), []).append(index)
        return [(indices, text) for text, indices in grouped.items()]
    return [([index], str(text)) for index, text in frame.loc[pending, text_column].items()]


def validate_generation_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_beams < 1 or args.num_return_sequences < 1:
        raise ValueError("Beam and return-sequence counts must be positive")
    if not args.do_sample and args.num_return_sequences > args.num_beams:
        raise ValueError(
            "Deterministic beam search requires --num-return-sequences <= --num-beams."
        )
    if args.num_beams > 1 and args.num_return_sequences > args.num_beams:
        raise ValueError(
            "Beam generation requires --num-return-sequences <= --num-beams."
        )
    if args.max_input_length < 1 or args.max_output_length < 1:
        raise ValueError("Maximum sequence lengths must be positive")
    if args.checkpoint_every_batches < 0:
        raise ValueError("--checkpoint-every-batches cannot be negative")


def main() -> None:
    args = parse_args()
    validate_generation_args(args)
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    output_path = resolve_output_path(args)
    if output_path.exists() and not (args.resume or args.overwrite):
        raise FileExistsError(
            f"{output_path} already exists; use --resume, --overwrite, or another --output."
        )
    if args.resume and args.overwrite:
        raise ValueError("Choose either --resume or --overwrite, not both.")

    frame = prepare_input(args)
    output_column = args.output_column or f"ParaphraseDemographics_{args.temperature:.2f}"
    frame[output_column] = ""
    resumed_rows = 0
    if args.resume:
        if not output_path.is_file():
            raise FileNotFoundError(f"--resume requested but output does not exist: {output_path}")
        frame[output_column] = load_resume_values(frame, output_path, output_column)
        resumed_rows = int(frame[output_column].str.strip().ne("").sum())

    items = pending_generation_items(
        frame, args.text_column, output_column, args.deduplicate_text
    )
    print(f"Input rows: {len(frame):,}")
    print(f"Previously completed rows: {resumed_rows:,}")
    print(f"Generation items: {len(items):,}")
    print(f"Output: {output_path.resolve()}")

    device = "not_used"
    model_versions: dict[str, str] = {}
    if args.copy_input_text:
        for indices, text in items:
            frame.loc[indices, output_column] = text
    elif items:
        import torch
        import transformers
        from transformers import set_seed

        set_seed(args.seed)
        device = resolve_device(args.device, torch)
        tokenizer, model = load_model(args.model, device)
        terms = [] if args.no_default_bad_terms else list(DEFAULT_BAD_TERMS)
        terms.extend(args.bad_term or [])
        bad_words_ids = make_bad_words_ids(tokenizer, terms)
        model_versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        print(f"Loaded {args.model} on {device}")

        total_batches = (len(items) + args.batch_size - 1) // args.batch_size
        for batch_number, start in enumerate(range(0, len(items), args.batch_size), start=1):
            batch = items[start:start + args.batch_size]
            generated = generate_batch(
                [text for _indices, text in batch],
                tokenizer,
                model,
                device,
                args,
                bad_words_ids,
            )
            for (indices, _text), paraphrase in zip(batch, generated):
                frame.loc[indices, output_column] = paraphrase
            print(f"Batch {batch_number:,}/{total_batches:,} complete", flush=True)
            if (
                args.checkpoint_every_batches
                and batch_number % args.checkpoint_every_batches == 0
            ):
                atomic_write_csv(frame, output_path)
                print(f"Checkpoint saved: {output_path}", flush=True)

    empty_source = frame[args.text_column].fillna("").astype(str).str.strip().eq("")
    frame.loc[empty_source, output_column] = ""
    atomic_write_csv(frame, output_path)

    run_info = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "input": str(args.input.resolve()),
        "output": str(output_path.resolve()),
        "input_rows": len(frame),
        "output_column": output_column,
        "model": args.model,
        "device": device,
        "cuda_visible_devices": args.cuda_visible_devices,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "shuffle": args.shuffle,
        "limit": args.limit,
        "generation": {
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_input_length": args.max_input_length,
            "max_output_length": args.max_output_length,
            "top_p": args.top_p if args.do_sample else None,
            "top_k": args.top_k if args.do_sample else None,
            "deduplicate_text": args.deduplicate_text,
            "bad_terms": (
                ([] if args.no_default_bad_terms else list(DEFAULT_BAD_TERMS))
                + (args.bad_term or [])
            ),
        },
        "resume": args.resume,
        "resumed_rows": resumed_rows,
        "copy_input_text_smoke_test": args.copy_input_text,
        "versions": model_versions,
    }
    run_path = output_path.with_suffix(".run.json")
    temporary_run = run_path.with_suffix(run_path.suffix + ".tmp")
    temporary_run.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_run, run_path)

    preview = frame.loc[
        frame[output_column].fillna("").astype(str).str.strip().ne("")
    ].head(max(args.preview_rows, 0))
    for _index, row in preview.iterrows():
        print(f"Original: {row[args.text_column]}")
        print(f"Paraphrase: {row[output_column]}")
        print("-" * 75)
    completed = int(frame[output_column].fillna("").astype(str).str.strip().ne("").sum())
    print(f"Completed paraphrases: {completed:,}/{len(frame):,}")
    print(f"Saved CSV: {output_path.resolve()}")
    print(f"Saved run metadata: {run_path.resolve()}")


if __name__ == "__main__":
    main()
