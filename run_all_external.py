#!/usr/bin/env python3
"""Build frozen patient embeddings for an external institution in two stages.

Stage 1 creates patient-level temporal representations from note embeddings:
simple mean, time-decay mean, and optionally a frozen temporal Transformer.

Stage 2 is optional. It applies frozen metric-learning adapters that were
trained with ICD comorbidity similarity as supervision. The deployed adapters
consume only Stage 1 patient embeddings; external ICD codes are not inputs.

No model is fitted, tuned, or selected on external data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .run_all import (
    OBJECTIVES,
    EpisodeNotes,
    aggregate_pooling,
    apply_adapter,
    atomic_json,
    evaluate_representation,
    fixed_episode_selection,
    infer_transformer,
    load_transformer,
    normalized,
    prepare_output,
    require_files,
    resolve_device,
    selection_sha256,
    sha256_file,
    validate_representation,
)


EMBEDDING_DIM = 768
DEFAULT_OUTPUT = Path("external_patient_embeddings")
DEFAULT_MISSING_LABELS = ("", "unknown", "nan", "na", "n/a", "not available")
TRANSFORMER_NAMES = (
    "transformer_stage_only",
    "transformer_reconstruction_only",
    "transformer_combined",
)


@dataclass(frozen=True)
class ExternalData:
    """Validated inputs aligned by run-local episode and patient identifiers."""

    notes: pd.DataFrame
    manifest: pd.DataFrame
    output_metadata: pd.DataFrame
    cohort_flow: pd.DataFrame
    endpoint_mappings: dict[str, dict[str, str]]
    audit: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    inputs = parser.add_argument_group("external inputs")
    inputs.add_argument("--note-emb-path", type=Path)
    inputs.add_argument("--note-meta-path", type=Path)
    inputs.add_argument("--episode-meta-path", type=Path)
    inputs.add_argument(
        "--note-run-path",
        type=Path,
        help="Optional JSON provenance file for the note embeddings.",
    )
    inputs.add_argument(
        "--endpoints",
        nargs="*",
        default=[],
        help="Episode columns used for optional retrieval evaluation.",
    )
    inputs.add_argument(
        "--missing-label-values",
        nargs="+",
        default=list(DEFAULT_MISSING_LABELS),
        help="Case-insensitive endpoint values treated as missing.",
    )
    inputs.add_argument(
        "--input-protocol-status",
        choices=("matched", "unmatched", "unknown"),
        default="unknown",
        help="Whether note construction and embedding match model development.",
    )

    stage1 = parser.add_argument_group("stage 1: temporal patient embeddings")
    stage1.add_argument(
        "--transformer-checkpoint",
        type=Path,
        help="Optional frozen Transformer best.pt; omit to produce pooling only.",
    )
    stage1.add_argument(
        "--transformer-name",
        choices=TRANSFORMER_NAMES,
        default="transformer_combined",
        help="Output name and matching DML base name for the Transformer.",
    )
    stage1.add_argument("--min-pre-notes", type=int, default=2)
    stage1.add_argument("--max-pre-notes", type=int, default=128)
    stage1.add_argument("--max-dx-notes", type=int, default=64)
    stage1.add_argument("--max-post-notes", type=int, default=128)
    stage1.add_argument("--half-life-days", type=float, default=30.0)
    stage1.add_argument("--batch-size", type=int, default=8)

    stage2 = parser.add_argument_group("stage 2: optional ICD-supervised DML")
    stage2.add_argument(
        "--run-dml",
        action="store_true",
        help="Apply frozen ICD-supervised adapters to every Stage 1 representation.",
    )
    stage2.add_argument(
        "--dml-root",
        type=Path,
        help="Root containing <base>/<objective>/best.pt and run_config.json.",
    )
    stage2.add_argument(
        "--dml-objectives",
        nargs="+",
        choices=list(OBJECTIVES),
        default=list(OBJECTIVES),
    )
    stage2.add_argument("--adapter-batch-size", type=int, default=512)

    runtime = parser.add_argument_group("runtime and outputs")
    runtime.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    runtime.add_argument("--device", default="auto")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    runtime.add_argument("--bootstrap-samples", type=int, default=1000)
    runtime.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate inputs and cohort construction without model inference.",
    )
    runtime.add_argument("--overwrite", action="store_true")
    runtime.add_argument(
        "--describe-formats",
        action="store_true",
        help="Print the input and output data contract and exit.",
    )
    return parser.parse_args()


def format_contract() -> dict[str, Any]:
    return {
        "inputs": {
            "note_embeddings_npy": {
                "shape": ["N_notes", EMBEDDING_DIM],
                "row_alignment": "row i matches row i of note_metadata.csv",
            },
            "note_metadata_csv": {
                "required_columns": [
                    "episode_id",
                    "patient_id",
                    "days_from_diagnosis",
                ],
                "one_row_per": "note embedding",
                "time_definition": "negative=pre-diagnosis, 0=diagnosis day, positive=post",
            },
            "episode_metadata_csv": {
                "required_columns": ["episode_id", "patient_id"],
                "optional_columns": [
                    "eligible (boolean; defaults to true)",
                    "columns named by --endpoints",
                ],
                "one_row_per": "diagnosis episode",
            },
        },
        "outputs": {
            "episode_metadata.csv": (
                "one row per included episode; row order matches every embedding array"
            ),
            "representations/mean/episode.emb.npy": ["N_episodes", EMBEDDING_DIM],
            "representations/time_decay/episode.emb.npy": [
                "N_episodes",
                EMBEDDING_DIM,
            ],
            "representations/<transformer-name>/episode.emb.npy": (
                "written when a checkpoint is supplied"
            ),
            "representations/<base>_icd_<objective>/episode.emb.npy": (
                "written only when --run-dml is enabled"
            ),
        },
        "privacy": (
            "Input IDs must already be deidentified. patient_id is removed from "
            "outputs; episode_id is retained for graph-feature joins."
        ),
    }


def require_runtime_args(args: argparse.Namespace) -> None:
    required = {
        "--note-emb-path": args.note_emb_path,
        "--note-meta-path": args.note_meta_path,
        "--episode-meta-path": args.episode_meta_path,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    if args.run_dml and args.dml_root is None:
        raise ValueError("--dml-root is required when --run-dml is enabled.")


def validate_arguments(args: argparse.Namespace) -> list[int]:
    if args.batch_size < 1 or args.adapter_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if args.min_pre_notes < 0 or min(
        args.max_pre_notes, args.max_dx_notes, args.max_post_notes
    ) < 0:
        raise ValueError("Note eligibility and phase limits must be non-negative.")
    if args.half_life_days <= 0:
        raise ValueError("--half-life-days must be positive.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative.")
    if len(set(args.endpoints)) != len(args.endpoints):
        raise ValueError("--endpoints contains duplicates.")
    if len(set(args.dml_objectives)) != len(args.dml_objectives):
        raise ValueError("--dml-objectives contains duplicates.")
    k_values = sorted(set(args.k))
    if not k_values or min(k_values) < 1:
        raise ValueError("Every K must be positive.")
    return k_values


def clean_ids(values: pd.Series, *, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} contains missing IDs.")
    cleaned = values.astype("string").str.strip()
    if cleaned.eq("").any():
        raise ValueError(f"{name} contains blank IDs.")
    return cleaned


def true_mask(values: pd.Series, *, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    text = values.astype("string").str.strip().str.lower().fillna("")
    accepted = {"1", "1.0", "true", "t", "yes", "y"}
    rejected = {"0", "0.0", "false", "f", "no", "n", ""}
    unexpected = sorted(set(text).difference(accepted | rejected))
    if unexpected:
        raise ValueError(f"{name} has unrecognized boolean values: {unexpected[:5]}")
    return text.isin(accepted)


def encode_endpoint(
    values: pd.Series,
    *,
    missing_labels: set[str],
) -> tuple[pd.Series, dict[str, str]]:
    """Encode labels for exact-match retrieval while retaining a reversible map."""
    text = values.astype("string").str.strip()
    missing = values.isna() | text.str.lower().isin(missing_labels)
    labels = sorted(text.loc[~missing].unique().tolist())
    code_by_label = {label: code for code, label in enumerate(labels)}
    encoded = text.map(code_by_label).astype("float64")
    encoded.loc[missing] = np.nan
    return encoded, {str(code): label for label, code in code_by_label.items()}


def read_external_data(args: argparse.Namespace) -> ExternalData:
    """Read the canonical CSVs and create privacy-safe evaluation indices."""
    note_meta = pd.read_csv(
        args.note_meta_path,
        dtype={"episode_id": "string", "patient_id": "string"},
    )
    episode_meta = pd.read_csv(
        args.episode_meta_path,
        dtype={"episode_id": "string", "patient_id": "string"},
    )
    note_required = {"episode_id", "patient_id", "days_from_diagnosis"}
    episode_required = {"episode_id", "patient_id", *args.endpoints}
    missing_notes = sorted(note_required.difference(note_meta.columns))
    missing_episodes = sorted(episode_required.difference(episode_meta.columns))
    if missing_notes:
        raise KeyError(f"Note metadata is missing columns: {missing_notes}")
    if missing_episodes:
        raise KeyError(f"Episode metadata is missing columns: {missing_episodes}")

    note_meta = note_meta.copy()
    episode_meta = episode_meta.copy()
    for frame in (note_meta, episode_meta):
        frame["episode_id"] = clean_ids(frame["episode_id"], name="episode_id")
        frame["patient_id"] = clean_ids(frame["patient_id"], name="patient_id")
    if episode_meta["episode_id"].duplicated().any():
        raise ValueError("Episode metadata must contain one row per episode_id.")

    unknown_note_episodes = set(note_meta["episode_id"]).difference(
        episode_meta["episode_id"]
    )
    if unknown_note_episodes:
        raise ValueError(
            f"Notes reference {len(unknown_note_episodes)} episodes absent from "
            "episode metadata."
        )
    patient_by_episode = episode_meta.set_index("episode_id")["patient_id"]
    expected_patient = note_meta["episode_id"].map(patient_by_episode)
    patient_mismatch = note_meta["patient_id"].ne(expected_patient)
    if patient_mismatch.any():
        raise ValueError(
            f"{int(patient_mismatch.sum())} notes disagree with episode patient_id."
        )

    days = pd.to_numeric(note_meta["days_from_diagnosis"], errors="raise")
    if days.isna().any() or not np.isfinite(days.to_numpy(dtype=np.float64)).all():
        raise ValueError("days_from_diagnosis must contain finite numeric values.")

    episode_meta["evaluation_episode_id"] = np.arange(
        1, len(episode_meta) + 1, dtype=np.int64
    )
    patient_codes, patient_values = pd.factorize(episode_meta["patient_id"], sort=True)
    episode_meta["patient_group_id"] = patient_codes.astype(np.int64) + 1
    episode_index = episode_meta.set_index("episode_id")["evaluation_episode_id"]
    patient_index = episode_meta.set_index("episode_id")["patient_group_id"]

    notes = pd.DataFrame(
        {
            "source_note_row": np.arange(len(note_meta), dtype=np.int64),
            "emb_row_idx": np.arange(len(note_meta), dtype=np.int64),
            "episode_id": note_meta["episode_id"].map(episode_index).astype("int64"),
            "patient_id": note_meta["episode_id"].map(patient_index).astype("int64"),
            "dt_days": days.to_numpy(dtype=np.float32),
        }
    )
    counts = (
        notes.groupby("episode_id")
        .agg(
            n_notes=("emb_row_idx", "size"),
            n_pre_notes=("dt_days", lambda values: int((values < 0).sum())),
            n_dx_notes=("dt_days", lambda values: int((values == 0).sum())),
            n_post_notes=("dt_days", lambda values: int((values > 0).sum())),
        )
        .reset_index()
    )
    manifest = episode_meta[
        ["evaluation_episode_id", "patient_group_id"]
    ].rename(columns={"evaluation_episode_id": "episode_id"})
    manifest = manifest.merge(
        counts, on="episode_id", how="left", validate="one_to_one"
    )
    count_columns = ["n_notes", "n_pre_notes", "n_dx_notes", "n_post_notes"]
    manifest[count_columns] = manifest[count_columns].fillna(0).astype("int64")

    missing_labels = {value.strip().lower() for value in args.missing_label_values}
    endpoint_mappings: dict[str, dict[str, str]] = {}
    for endpoint in args.endpoints:
        manifest[endpoint], endpoint_mappings[endpoint] = encode_endpoint(
            episode_meta[endpoint], missing_labels=missing_labels
        )

    institution_eligible = (
        true_mask(episode_meta["eligible"], name="eligible")
        if "eligible" in episode_meta
        else pd.Series(True, index=episode_meta.index)
    )
    manifest["eligible"] = (
        institution_eligible.to_numpy()
        & manifest["n_pre_notes"].ge(args.min_pre_notes)
    )

    output_metadata = episode_meta[
        ["episode_id", "evaluation_episode_id", "patient_group_id", *args.endpoints]
    ].merge(
        manifest[["episode_id", *count_columns]].rename(
            columns={"episode_id": "evaluation_episode_id"}
        ),
        on="evaluation_episode_id",
        validate="one_to_one",
    )
    output_metadata = output_metadata[
        [
            "episode_id",
            "evaluation_episode_id",
            "patient_group_id",
            *count_columns,
            *args.endpoints,
        ]
    ]
    cohort_flow = output_metadata.copy()
    cohort_flow["eligible"] = manifest["eligible"].to_numpy()

    return ExternalData(
        notes=notes,
        manifest=manifest,
        output_metadata=output_metadata,
        cohort_flow=cohort_flow,
        endpoint_mappings=endpoint_mappings,
        audit={
            "n_notes": len(notes),
            "n_episodes": len(manifest),
            "n_patients": int(len(patient_values)),
            "n_eligible_episodes": int(manifest["eligible"].sum()),
            "n_eligible_patients": int(
                manifest.loc[manifest["eligible"], "patient_group_id"].nunique()
            ),
            "minimum_pre_diagnosis_notes": args.min_pre_notes,
            "input_episode_ids_retained": True,
            "input_patient_ids_written": False,
        },
    )


def load_note_embeddings(
    args: argparse.Namespace,
    *,
    expected_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.load(args.note_emb_path, mmap_mode="r")
    if matrix.shape != (expected_rows, EMBEDDING_DIM):
        raise ValueError(
            f"Expected note embeddings with shape ({expected_rows}, {EMBEDDING_DIM}); "
            f"got {matrix.shape}."
        )
    sample_rows = np.unique(
        np.linspace(0, len(matrix) - 1, min(1024, len(matrix))).astype(int)
    )
    sample = np.asarray(matrix[sample_rows], dtype=np.float32)
    if not np.isfinite(sample).all():
        raise ValueError("Note embeddings contain non-finite sampled values.")

    provenance = json.loads(args.note_run_path.read_text()) if args.note_run_path else {}
    warning = (
        None
        if args.input_protocol_status == "matched"
        else "Note construction and embedding preprocessing are not confirmed to "
        "match the frozen model-development protocol."
    )
    return matrix, {
        "shape": list(matrix.shape),
        "model": provenance.get("model"),
        "text_fields": provenance.get("text_fields", provenance.get("summary_col")),
        "provenance_file": args.note_run_path.name if args.note_run_path else None,
        "provenance_sha256": (
            sha256_file(args.note_run_path) if args.note_run_path else None
        ),
        "input_protocol_status": args.input_protocol_status,
        "input_protocol_warning": warning,
        "sample_max_l2_norm_error": float(
            np.max(np.abs(np.linalg.norm(sample, axis=1) - 1.0))
        ),
    }


def stage_1_temporal_embeddings(
    args: argparse.Namespace,
    note_embeddings: np.ndarray,
    episodes: list[EpisodeNotes],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    """Stage 1: mean, time-decay, and optional Transformer representations."""
    print("Stage 1/2: temporal patient embeddings", flush=True)
    mean, time_decay = aggregate_pooling(
        note_embeddings, episodes, half_life_days=args.half_life_days
    )
    representations = {"mean": mean, "time_decay": time_decay}
    predictions: dict[str, np.ndarray] = {}
    checkpoint_hashes: dict[str, str] = {}

    if args.transformer_checkpoint is None:
        print("  Transformer skipped: no checkpoint supplied", flush=True)
        return representations, predictions, checkpoint_hashes

    print(f"  Transformer: {args.transformer_name}", flush=True)
    model, checkpoint = load_transformer(args.transformer_checkpoint, device)
    config = checkpoint["model_config"]
    expected_limits = (
        int(config["max_pre_notes"]),
        int(config["max_dx_notes"]),
        int(config["max_post_notes"]),
    )
    requested_limits = (
        args.max_pre_notes,
        args.max_dx_notes,
        args.max_post_notes,
    )
    if expected_limits != requested_limits:
        raise ValueError(
            f"Transformer expects phase limits {expected_limits}, got {requested_limits}."
        )
    values, stage_predictions = infer_transformer(
        model,
        note_embeddings,
        episodes,
        device=device,
        batch_size=args.batch_size,
    )
    representations[args.transformer_name] = normalized(values)
    predictions[args.transformer_name] = stage_predictions
    checkpoint_hashes[args.transformer_name] = sha256_file(
        args.transformer_checkpoint
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return representations, predictions, checkpoint_hashes


def validate_dml_artifacts(
    args: argparse.Namespace,
    base_names: list[str],
) -> None:
    if not args.run_dml:
        return
    required: list[Path] = []
    for base in base_names:
        for objective in args.dml_objectives:
            directory = args.dml_root / base / objective
            required.extend([directory / "best.pt", directory / "run_config.json"])
    require_files(required)


def stage_2_dml_embeddings(
    args: argparse.Namespace,
    stage_1: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Stage 2: optionally apply adapters trained with ICD supervision."""
    if not args.run_dml:
        print("Stage 2/2: ICD-supervised DML skipped", flush=True)
        return {}, {}

    print("Stage 2/2: frozen ICD-supervised DML adapters", flush=True)
    adapted: dict[str, np.ndarray] = {}
    checkpoint_hashes: dict[str, str] = {}
    for base, base_values in stage_1.items():
        for objective in args.dml_objectives:
            slug = f"{base}_icd_{objective}"
            directory = args.dml_root / base / objective
            config = json.loads((directory / "run_config.json").read_text())
            if config.get("architecture") != "base_embedding_only_residual_mlp_v1":
                raise ValueError(
                    f"{slug} is not an embedding-only deployable adapter."
                )
            if config.get("base_name") != base:
                raise ValueError(f"{slug} run configuration base mismatch.")
            print(f"  Adapter: {slug}", flush=True)
            checkpoint = directory / "best.pt"
            adapted[slug] = apply_adapter(
                base_values,
                checkpoint,
                device=device,
                batch_size=args.adapter_batch_size,
            )
            checkpoint_hashes[slug] = sha256_file(checkpoint)
    return adapted, checkpoint_hashes


def dml_incremental(summary: pd.DataFrame) -> pd.DataFrame:
    adapted = summary.loc[summary["method"].str.contains("_icd_")].copy()
    if adapted.empty:
        return pd.DataFrame()
    adapted["base_method"] = adapted["method"].str.split("_icd_", n=1).str[0]
    baseline = summary.rename(
        columns={
            "method": "base_method",
            "mean_precision_at_k": "base_mean_precision_at_k",
            "macro_target_precision_at_k": "base_macro_target_precision_at_k",
        }
    )[
        [
            "base_method",
            "endpoint",
            "k",
            "base_mean_precision_at_k",
            "base_macro_target_precision_at_k",
        ]
    ]
    merged = adapted.merge(
        baseline,
        on=["base_method", "endpoint", "k"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("A DML representation is missing its Stage 1 baseline.")
    merged = merged.drop(columns="_merge")
    merged["dml_minus_base_precision_at_k"] = (
        merged["mean_precision_at_k"] - merged["base_mean_precision_at_k"]
    )
    merged["dml_minus_base_macro_target_precision_at_k"] = (
        merged["macro_target_precision_at_k"]
        - merged["base_macro_target_precision_at_k"]
    )
    return merged


def write_representations(
    output_dir: Path,
    representations: dict[str, np.ndarray],
    *,
    n_episodes: int,
) -> None:
    for name, values in representations.items():
        validate_representation(values, n_episodes, name)
        path = output_dir / "representations" / name / "episode.emb.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values)


def evaluate_and_write(
    args: argparse.Namespace,
    representations: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    k_values: list[int],
) -> None:
    if not args.endpoints:
        print("Retrieval evaluation skipped: no --endpoints supplied", flush=True)
        return
    summaries: list[pd.DataFrame] = []
    per_queries: list[pd.DataFrame] = []
    neighbors: list[pd.DataFrame] = []
    for name, values in representations.items():
        print(f"Evaluating {name}", flush=True)
        summary, per_query, neighbor = evaluate_representation(
            name,
            values,
            manifest,
            endpoints=args.endpoints,
            k_values=k_values,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        summaries.append(summary)
        per_queries.append(per_query)
        neighbors.append(neighbor)

    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(args.output_dir / "retrieval_summary.csv", index=False)
    pd.concat(per_queries, ignore_index=True).to_csv(
        args.output_dir / "retrieval_per_query.csv", index=False
    )
    pd.concat(neighbors, ignore_index=True).to_csv(
        args.output_dir / "retrieval_neighbors.csv", index=False
    )
    if 5 in k_values:
        summary.loc[summary["k"].eq(5)].pivot(
            index="method", columns="endpoint", values="mean_precision_at_k"
        ).to_csv(args.output_dir / "retrieval_p5_wide.csv")
    incremental = dml_incremental(summary)
    if not incremental.empty:
        incremental.to_csv(
            args.output_dir / "retrieval_dml_incremental.csv", index=False
        )


def main() -> None:
    args = parse_args()
    if args.describe_formats:
        print(json.dumps(format_contract(), indent=2))
        return
    require_runtime_args(args)
    k_values = validate_arguments(args)

    input_paths = [args.note_emb_path, args.note_meta_path, args.episode_meta_path]
    if args.note_run_path:
        input_paths.append(args.note_run_path)
    if args.transformer_checkpoint:
        input_paths.append(args.transformer_checkpoint)
    require_files(input_paths)
    prepare_output(args.output_dir, overwrite=args.overwrite)

    external = read_external_data(args)
    note_embeddings, embedding_audit = load_note_embeddings(
        args, expected_rows=len(external.notes)
    )
    eligible_manifest = (
        external.manifest.loc[external.manifest["eligible"]]
        .sort_values("episode_id")
        .reset_index(drop=True)
    )
    if eligible_manifest.empty:
        raise ValueError("No episodes satisfy the note-history eligibility rule.")
    episodes, selection = fixed_episode_selection(
        external.notes,
        eligible_manifest,
        seed=args.seed,
        max_pre_notes=args.max_pre_notes,
        max_dx_notes=args.max_dx_notes,
        max_post_notes=args.max_post_notes,
    )
    if args.endpoints:
        largest_patient = int(
            eligible_manifest["patient_group_id"].value_counts().max()
        )
        if len(eligible_manifest) - largest_patient < max(k_values):
            raise ValueError("Too few other-patient episodes for the requested K.")

    eligible_ids = set(eligible_manifest["episode_id"])
    output_metadata = external.output_metadata.loc[
        external.output_metadata["evaluation_episode_id"].isin(eligible_ids)
    ].sort_values("evaluation_episode_id").reset_index(drop=True)
    selection_output = selection.rename(
        columns={"episode_id": "evaluation_episode_id"}
    ).merge(
        external.output_metadata[["episode_id", "evaluation_episode_id"]],
        on="evaluation_episode_id",
        validate="many_to_one",
    )
    selection_output = selection_output[
        [
            "episode_id",
            "evaluation_episode_id",
            "note_sequence_position",
            "phase",
            "emb_row_idx",
            "dt_days",
        ]
    ]

    selection_hash = selection_sha256(selection)
    external.cohort_flow.to_csv(args.output_dir / "cohort_flow.csv", index=False)
    output_metadata.to_csv(args.output_dir / "episode_metadata.csv", index=False)
    selection_output.to_csv(args.output_dir / "note_selection.csv", index=False)
    audit = {
        "cohort": external.audit,
        "note_embeddings": embedding_audit,
        "selection": {
            "sha256": selection_hash,
            "n_selected_notes": len(selection),
            "max_pre_notes": args.max_pre_notes,
            "max_dx_notes": args.max_dx_notes,
            "max_post_notes": args.max_post_notes,
            "seed": args.seed,
        },
        "endpoint_label_codes": external.endpoint_mappings,
        "stage_1": {
            "always": ["mean", "time_decay"],
            "transformer": (
                args.transformer_name if args.transformer_checkpoint else None
            ),
        },
        "stage_2": {
            "enabled": args.run_dml,
            "external_icd_codes_required": False,
            "icd_role": "training-only supervision for the frozen adapters",
            "objectives": args.dml_objectives if args.run_dml else [],
        },
    }
    atomic_json(args.output_dir / "data_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return

    device = resolve_device(args.device)
    stage_1, predictions, stage_1_hashes = stage_1_temporal_embeddings(
        args, note_embeddings, episodes, device
    )
    validate_dml_artifacts(args, list(stage_1))
    stage_2, stage_2_hashes = stage_2_dml_embeddings(args, stage_1, device)
    representations = {**stage_1, **stage_2}
    write_representations(
        args.output_dir, representations, n_episodes=len(eligible_manifest)
    )

    prediction_output = output_metadata[
        ["episode_id", "evaluation_episode_id", "patient_group_id"]
    ].copy()
    for name, values in predictions.items():
        prediction_output[f"{name}_predicted_stage_0_to_4"] = values
    prediction_output.to_csv(
        args.output_dir / "transformer_stage_predictions.csv", index=False
    )
    evaluate_and_write(args, representations, eligible_manifest, k_values)

    report = {
        "status": "complete",
        "stage_1_temporal_representations": list(stage_1),
        "stage_2_dml_enabled": args.run_dml,
        "stage_2_dml_representations": list(stage_2),
        "dml_inference_contract": (
            "The adapters consume patient embeddings only. ICD comorbidity codes "
            "were training supervision and are not external inference inputs."
        ),
        "n_episodes": len(eligible_manifest),
        "endpoints_evaluated": args.endpoints,
        "checkpoint_sha256": {**stage_1_hashes, **stage_2_hashes},
        "selection_sha256": selection_hash,
        "note_embedding_sha256": sha256_file(args.note_emb_path),
        "input_protocol_status": args.input_protocol_status,
        "limitations": [
            warning
            for warning in (
                embedding_audit["input_protocol_warning"],
                (
                    "Within-institution retrieval evaluates cohort structure; it is "
                    "not an external-query-to-frozen-internal-gallery estimate."
                    if args.endpoints
                    else None
                ),
            )
            if warning is not None
        ],
    }
    atomic_json(args.output_dir / "run_config.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
