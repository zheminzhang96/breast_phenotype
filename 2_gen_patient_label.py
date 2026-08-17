#!/usr/bin/env python3
"""Add standardized EHR labels to a combined clinical-note CSV.

This is the portable Python-script version of ``1_gen_ehr_label.ipynb``. The
notebook is preserved unchanged. This script removes notebook-state and
institution-specific path dependencies, processes large CSVs in chunks, and
writes reproducibility metadata and patient-level audit tables.

Added row-level columns
-----------------------
``tobacco_group``
    smoker, non-smoker, or unknown
``alcohol_group``
    alcoholic drinker, social drinker, never drink, or unknown
``employment_group``
    employment, unemployment, or unknown
``depression_group``
    low risk, medium risk, high risk, or unknown
``insurance_group``
    Medicare, Commercial, Self-Paid, Uninsurance, or Other/Unknown
``BMI_group``
    Desirable, Intermediate, High, or Invalid

Required columns
----------------
PATIENT_CLINIC_NUMBER, soc__Tobacco Use, soc__Alcohol Use, soc__Employment,
soc__Depression, ins__FINANCIAL_CLASS_NAME, and bmi__BMI_NUM.

The notebook contains an unreachable depression text-label branch after an
early ``return``. This copy intentionally fixes it: ``NOT AT RISK`` and
``NONE TO MINIMAL DEPRESSION SEVERITY`` map to low risk, and ``AT RISK`` maps
to high risk. This correction is recorded in the run JSON.

Dependencies
------------
Python 3.10+ and pandas.

Example
-------
python 1_gen_ehr_label_replication.py \
  --input CN_and_ehr_breast.csv \
  --output CN_and_ehr_breast_labeled.csv

Run the command once per cohort. If ``--output`` is omitted, ``_labeled`` is
added to the input filename. Existing outputs are protected unless
``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_PATIENT_ID_COLUMN = "PATIENT_CLINIC_NUMBER"
SOURCE_COLUMNS = {
    "tobacco_group": "soc__Tobacco Use",
    "alcohol_group": "soc__Alcohol Use",
    "employment_group": "soc__Employment",
    "depression_group": "soc__Depression",
    "insurance_group": "ins__FINANCIAL_CLASS_NAME",
    "BMI_group": "bmi__BMI_NUM",
}
PATIENT_OUTPUT_COLUMNS = {
    "tobacco_group": "SMOKING_GROUP",
    "alcohol_group": "ALCOHOL_GROUP",
    "employment_group": "EMPLOYMENT_GROUP",
    "depression_group": "DEPRESSION_RISK_GROUP",
    "insurance_group": "INSURANCE_GROUP",
    "BMI_group": "BMI_GROUP",
}
PATIENT_GROUP_PRIORITY = {
    "tobacco_group": ("smoker", "non-smoker", "unknown"),
    "alcohol_group": (
        "alcoholic drinker",
        "social drinker",
        "never drink",
        "unknown",
    ),
    "employment_group": ("employment", "unemployment", "unknown"),
    "depression_group": ("high risk", "medium risk", "low risk", "unknown"),
    "BMI_group": ("High", "Intermediate", "Desirable", "Invalid"),
}
INSURANCE_TIE_PRIORITY = (
    "Medicare",
    "Commercial",
    "Self-Paid",
    "Uninsurance",
    "Other/Unknown",
)

SMOKER_ANSWERS = {
    "EVERY DAY",
    "CURRENT",
    "SOME DAYS",
    "LIGHT SMOKER",
    "HEAVY SMOKER",
    "YES",
    "FORMER",
    "PAST",
    "SMOKER, CURRENT STATUS UNKNOWN",
}
NON_SMOKER_ANSWERS = {"NEVER", "PASSIVE SMOKE EXPOSURE - NEVER SMOKER"}

ALCOHOL_NEVER_ANSWERS = {"NEVER", "PATIENT DOES NOT DRINK"}
ALCOHOL_SOCIAL_ANSWERS = {
    "1 OR 2",
    "3 OR 4",
    "MONTHLY OR LESS",
    "LESS THAN MONTHLY",
    "2-4 TIMES A MONTH",
    "MONTHLY",
    "WEEKLY",
    "2-3 TIMES A WEEK",
}
ALCOHOL_HEAVY_ANSWERS = {
    "4 OR MORE TIMES A WEEK",
    "DAILY OR ALMOST DAILY",
    "5 OR 6",
    "7 TO 9",
    "10 OR MORE",
}

EMPLOYED_ANSWERS = {
    "EMPLOYED AND ACTIVELY WORKING WITHOUT RESTRICTIONS",
    "WORKING WITH TEMPORARY RESTRICTIONS",
    "EMPLOYED BUT NOT WORKING DUE TO ILLNESS OR INJURY",
    "EMPLOYED BUT NOT WORKING DUE TO FURLOUGH",
}
UNEMPLOYED_ANSWERS = {
    "RETIRED",
    "PERMANENTLY DISABLED",
    "TEMPORARILY DISABLED",
    "UNEMPLOYED/NOT IN THE PAID WORKFORCE AND NOT SEEKING EMPLOYMENT",
    "UNEMPLOYED/NOT IN THE PAID WORKFORCE BUT SEEKING EMPLOYMENT",
}

DEPRESSION_LOW_TEXT = {"NOT AT RISK", "NONE TO MINIMAL DEPRESSION SEVERITY"}
DEPRESSION_HIGH_TEXT = {"AT RISK"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add notebook-compatible EHR category labels to a note CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Combined note + EHR CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Labeled CSV. Defaults to INPUT_STEM_labeled.csv beside the input.",
    )
    parser.add_argument("--patient-id-column", default=DEFAULT_PATIENT_ID_COLUMN)
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100_000,
        help="Rows per read/write chunk; keeps the breast cohort memory bounded.",
    )
    parser.add_argument(
        "--allow-missing-columns",
        action="store_true",
        help="Fill a missing EHR source column with unknown/Invalid instead of failing.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_patient_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalize_answer(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


def classify_tobacco(series: pd.Series) -> pd.Series:
    answer = normalize_answer(series)
    result = pd.Series("unknown", index=series.index, dtype="string")
    result.loc[answer.isin(SMOKER_ANSWERS)] = "smoker"
    result.loc[answer.isin(NON_SMOKER_ANSWERS)] = "non-smoker"
    return result


def classify_alcohol(series: pd.Series) -> pd.Series:
    answer = normalize_answer(series)
    result = pd.Series("unknown", index=series.index, dtype="string")
    result.loc[answer.isin(ALCOHOL_NEVER_ANSWERS)] = "never drink"
    result.loc[answer.isin(ALCOHOL_SOCIAL_ANSWERS)] = "social drinker"
    result.loc[answer.isin(ALCOHOL_HEAVY_ANSWERS)] = "alcoholic drinker"
    return result


def classify_employment(series: pd.Series) -> pd.Series:
    answer = normalize_answer(series)
    result = pd.Series("unknown", index=series.index, dtype="string")
    result.loc[answer.isin(EMPLOYED_ANSWERS)] = "employment"
    result.loc[answer.isin(UNEMPLOYED_ANSWERS)] = "unemployment"
    return result


def classify_depression(series: pd.Series) -> pd.Series:
    answer = normalize_answer(series)
    score = pd.to_numeric(answer, errors="coerce")
    result = pd.Series("unknown", index=series.index, dtype="string")
    result.loc[score.eq(0)] = "low risk"
    result.loc[score.between(1, 2, inclusive="both")] = "medium risk"
    result.loc[score.ge(3)] = "high risk"
    result.loc[answer.isin(DEPRESSION_LOW_TEXT)] = "low risk"
    result.loc[answer.isin(DEPRESSION_HIGH_TEXT)] = "high risk"
    return result


def classify_insurance(series: pd.Series) -> pd.Series:
    answer = series.astype("string").fillna("").str.strip().str.lower()
    result = pd.Series("Other/Unknown", index=series.index, dtype="string")
    result.loc[answer.str.contains("uninsured|uninsurance|no insurance", regex=True)] = (
        "Uninsurance"
    )
    result.loc[
        answer.str.contains("self pay|self-pay|selfpay|cash pay", regex=True)
    ] = "Self-Paid"
    result.loc[answer.str.contains("commercial", regex=False)] = "Commercial"
    # Preserve notebook precedence when a string contains multiple keywords.
    result.loc[answer.str.contains("medicare", regex=False)] = "Medicare"
    return result


def classify_bmi(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series("Invalid", index=series.index, dtype="string")
    valid = numeric.between(15, 40, inclusive="both")
    result.loc[valid & numeric.lt(25)] = "Desirable"
    result.loc[valid & numeric.between(25, 30, inclusive="both")] = "Intermediate"
    result.loc[valid & numeric.gt(30)] = "High"
    text = series.astype("string").str.strip().str.lower()
    result.loc[numeric.isna() & text.eq("desirable")] = "Desirable"
    result.loc[numeric.isna() & text.eq("intermediate")] = "Intermediate"
    result.loc[numeric.isna() & text.eq("high")] = "High"
    return result


CLASSIFIERS = {
    "tobacco_group": classify_tobacco,
    "alcohol_group": classify_alcohol,
    "employment_group": classify_employment,
    "depression_group": classify_depression,
    "insurance_group": classify_insurance,
    "BMI_group": classify_bmi,
}


def required_columns(patient_id_column: str) -> set[str]:
    return {patient_id_column, *SOURCE_COLUMNS.values()}


def validate_header(
    input_path: Path,
    patient_id_column: str,
    allow_missing: bool,
) -> list[str]:
    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    if patient_id_column not in header:
        raise ValueError(f"{input_path} is missing patient ID column {patient_id_column!r}.")
    missing = sorted(required_columns(patient_id_column) - set(header))
    if missing and not allow_missing:
        raise ValueError(
            f"{input_path} is missing required EHR columns: {missing}. "
            "Use --allow-missing-columns only when unknown labels are intended."
        )
    return missing


def label_chunk(chunk: pd.DataFrame, missing_source_columns: Iterable[str]) -> pd.DataFrame:
    out = chunk.copy()
    for column in missing_source_columns:
        out[column] = pd.NA
    for output_column, source_column in SOURCE_COLUMNS.items():
        out[output_column] = CLASSIFIERS[output_column](out[source_column])
    return out


def update_patient_counts(
    accumulators: dict[str, dict[str, Counter[str]]],
    chunk: pd.DataFrame,
    patient_id_column: str,
) -> None:
    patient_ids = chunk[patient_id_column].map(clean_patient_id)
    for row_label in SOURCE_COLUMNS:
        working = pd.DataFrame({
            "patient_id": patient_ids,
            "group": chunk[row_label].astype(str),
        })
        working = working.loc[working["patient_id"].ne("")]
        grouped = working.groupby(["patient_id", "group"], sort=False).size()
        for (patient_id, group), count in grouped.items():
            accumulators[row_label][patient_id][group] += int(count)


def priority_group(counts: Counter[str], priority: Iterable[str]) -> str:
    return next((group for group in priority if counts[group] > 0), next(iter(priority)))


def most_frequent_insurance(counts: Counter[str]) -> str:
    maximum = max(counts.values(), default=0)
    if maximum == 0:
        return "Other/Unknown"
    tied = {group for group, count in counts.items() if count == maximum}
    return next(group for group in INSURANCE_TIE_PRIORITY if group in tied)


def build_patient_audits(
    accumulators: dict[str, dict[str, Counter[str]]],
    patient_id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_ids = sorted({
        patient_id
        for feature_counts in accumulators.values()
        for patient_id in feature_counts
    })
    patient_rows: list[dict[str, object]] = []
    for patient_id in patient_ids:
        row: dict[str, object] = {patient_id_column: patient_id}
        for row_label, patient_column in PATIENT_OUTPUT_COLUMNS.items():
            counts = accumulators[row_label][patient_id]
            if row_label == "insurance_group":
                group = most_frequent_insurance(counts)
            else:
                group = priority_group(counts, PATIENT_GROUP_PRIORITY[row_label])
            row[patient_column] = group
        patient_rows.append(row)
    patient_groups = pd.DataFrame(
        patient_rows,
        columns=[patient_id_column, *PATIENT_OUTPUT_COLUMNS.values()],
    )

    count_rows: list[dict[str, object]] = []
    for row_label, patient_column in PATIENT_OUTPUT_COLUMNS.items():
        counts = patient_groups[patient_column].value_counts(dropna=False)
        for group, count in counts.items():
            count_rows.append({
                "feature": row_label,
                "patient_group_column": patient_column,
                "group": group,
                "unique_patient_count": int(count),
                "percent": 100.0 * count / len(patient_groups) if len(patient_groups) else 0.0,
            })
    return patient_groups, pd.DataFrame(count_rows)


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output
    return args.input.with_name(f"{args.input.stem}_labeled.csv")


def related_output_paths(labeled_path: Path) -> dict[str, Path]:
    stem = labeled_path.with_suffix("")
    return {
        "labeled": labeled_path,
        "patient_groups": stem.with_name(stem.name + ".patient_groups.csv"),
        "label_counts": stem.with_name(stem.name + ".label_counts.csv"),
        "run": stem.with_name(stem.name + ".run.json"),
    }


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.chunksize < 1:
        raise ValueError("--chunksize must be positive")
    output_path = resolve_output_path(args)
    paths = related_output_paths(output_path)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; choose another --output or pass --overwrite:\n  "
            + "\n  ".join(map(str, existing))
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    missing_columns = validate_header(
        args.input, args.patient_id_column, args.allow_missing_columns
    )
    accumulators: dict[str, dict[str, Counter[str]]] = {
        row_label: defaultdict(Counter) for row_label in SOURCE_COLUMNS
    }
    temporary_labeled = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_labeled.exists():
        temporary_labeled.unlink()

    total_rows = 0
    chunks = 0
    try:
        for chunks, chunk in enumerate(
            pd.read_csv(
                args.input,
                dtype=str,
                low_memory=False,
                chunksize=args.chunksize,
            ),
            start=1,
        ):
            labeled = label_chunk(chunk, missing_columns)
            update_patient_counts(accumulators, labeled, args.patient_id_column)
            labeled.to_csv(
                temporary_labeled,
                mode="w" if chunks == 1 else "a",
                header=chunks == 1,
                index=False,
            )
            total_rows += len(labeled)
            print(
                f"Chunk {chunks:,}: wrote {len(labeled):,} rows "
                f"({total_rows:,} total)",
                flush=True,
            )
        if chunks == 0:
            raise ValueError(f"Input CSV has no data rows: {args.input}")
        os.replace(temporary_labeled, output_path)
    except BaseException:
        temporary_labeled.unlink(missing_ok=True)
        raise

    patient_groups, label_counts = build_patient_audits(
        accumulators, args.patient_id_column
    )
    atomic_write_csv(patient_groups, paths["patient_groups"])
    atomic_write_csv(label_counts, paths["label_counts"])
    run_info = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "source_notebook": "1_gen_ehr_label.ipynb",
        "input": str(args.input.resolve()),
        "outputs": {name: str(path.resolve()) for name, path in paths.items() if name != "run"},
        "rows": total_rows,
        "unique_patients": len(patient_groups),
        "patient_id_column": args.patient_id_column,
        "chunksize": args.chunksize,
        "allow_missing_columns": args.allow_missing_columns,
        "missing_columns_filled_as_unknown": missing_columns,
        "depression_text_label_bug_fixed": True,
        "insurance_patient_tie_priority": list(INSURANCE_TIE_PRIORITY),
        "patient_group_priorities": {
            name: list(priority) for name, priority in PATIENT_GROUP_PRIORITY.items()
        },
    }
    temporary_run = paths["run"].with_suffix(paths["run"].suffix + ".tmp")
    temporary_run.write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_run, paths["run"])

    print(f"Labeled rows: {total_rows:,}")
    print(f"Unique patients in audits: {len(patient_groups):,}")
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
