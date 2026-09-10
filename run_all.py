#!/usr/bin/env python3
"""Run the frozen Stage 02/03 representation matrix on SEER-linked episodes.

This runner never fits or selects a model on SEER.  It creates one fixed note
selection for every eligible diagnosis episode, exports the five Stage 02
representations, applies each matched frozen Stage 03 adapter, and evaluates
all 20 representations by leave-one-patient-out retrieval within SEER.

Stage 04 is audited but deliberately not inferred: the frozen graph requires
31 Ki-67/radiology indicators that are absent from the linked SEER table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from zhemin.pipeline.stage_02_temporal.model import TemporalNoteTransformer
from zhemin.pipeline.stage_03_dml.model import ResidualMetricAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEER_ROOT = PROJECT_ROOT / "zhemin" / "seer"
EXPERIMENT_ROOT = PROJECT_ROOT / "zhemin" / "experiments"
DEFAULT_NOTE_EMB = (
    SEER_ROOT / "emb_data/note_embeddings/medembed/breast.note.emb.npy"
)
DEFAULT_NOTE_META = (
    SEER_ROOT / "emb_data/note_embeddings/medembed/breast.note.meta.csv"
)
DEFAULT_NOTE_RUN = (
    SEER_ROOT / "emb_data/note_embeddings/medembed/breast.note.run.json"
)
DEFAULT_LABELS = SEER_ROOT / "notes_labels_dx.csv"
DEFAULT_STAGE02_ROOT = EXPERIMENT_ROOT / "02_patient_embeddings"
DEFAULT_STAGE03_ROOT = EXPERIMENT_ROOT / "03_icd_dml"
DEFAULT_STAGE04_ROOT = EXPERIMENT_ROOT / "04_phenotype_gnn_no_stage_edges"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "05_seer_external"

BASES = (
    "mean",
    "time_decay",
    "transformer_stage_only",
    "transformer_reconstruction_only",
    "transformer_combined",
)
TRANSFORMER_DIRS = {
    "transformer_stage_only": "transformer_ablations/stage_only",
    "transformer_reconstruction_only": "transformer_ablations/reconstruction_only",
    "transformer_combined": "transformer_ablations/combined",
}
OBJECTIVES = ("triplet", "supcon", "fastap")
LABEL_COLUMNS = (
    "label_row_id",
    "patient_clinic_key",
    "dx_date",
    "age",
    "sex",
    "histology",
    "grade",
    "proliferation",
    "er",
    "pr",
    "her2",
    "t",
    "n",
    "m",
    "stage",
    "race",
    "local_recurrence",
    "distant_recurrence",
    "regional_recurrence",
)
DEFAULT_ENDPOINTS = (
    "stage",
    "histology",
    "grade",
    "er",
    "pr",
    "her2",
    "recurrence_any",
)


@dataclass(frozen=True)
class EpisodeNotes:
    episode_id: int
    patient_id: int
    rows: np.ndarray
    days: np.ndarray
    source_rows: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note-emb-path", type=Path, default=DEFAULT_NOTE_EMB)
    parser.add_argument("--note-meta-path", type=Path, default=DEFAULT_NOTE_META)
    parser.add_argument("--note-run-path", type=Path, default=DEFAULT_NOTE_RUN)
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--stage02-root", type=Path, default=DEFAULT_STAGE02_ROOT)
    parser.add_argument("--stage03-root", type=Path, default=DEFAULT_STAGE03_ROOT)
    parser.add_argument("--stage04-root", type=Path, default=DEFAULT_STAGE04_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--adapter-batch-size", type=int, default=512)
    parser.add_argument("--min-pre-notes", type=int, default=2)
    parser.add_argument("--max-pre-notes", type=int, default=128)
    parser.add_argument("--max-dx-notes", type=int, default=64)
    parser.add_argument("--max-post-notes", type=int, default=128)
    parser.add_argument("--half-life-days", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--endpoints", nargs="+", default=list(DEFAULT_ENDPOINTS))
    parser.add_argument(
        "--include-development-overlap",
        action="store_true",
        help="Include patients also present in the internal model-development cohort.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate inputs and write the eligibility audit without model inference.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def stable_seed(*parts: object) -> int:
    text = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little")


def select_indices(
    indices: np.ndarray,
    limit: int,
    *,
    seed: int,
) -> np.ndarray:
    if limit < 0:
        raise ValueError("Note limits must be non-negative.")
    if len(indices) <= limit:
        return np.arange(len(indices), dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(indices), size=limit, replace=False))


def fixed_episode_selection(
    notes: pd.DataFrame,
    eligible: pd.DataFrame,
    *,
    seed: int,
    max_pre_notes: int,
    max_dx_notes: int,
    max_post_notes: int,
) -> tuple[list[EpisodeNotes], pd.DataFrame]:
    """Select the same deterministic phase-stratified notes for every method."""
    wanted = set(eligible["episode_id"].astype(int))
    selected_records: list[dict[str, Any]] = []
    episode_notes: list[EpisodeNotes] = []
    for episode_id, frame in notes.loc[notes["episode_id"].isin(wanted)].groupby(
        "episode_id", sort=True
    ):
        frame = frame.sort_values(["dt_days", "emb_row_idx"], kind="stable")
        output_positions: list[int] = []
        for phase_code, mask, limit in (
            (0, frame["dt_days"].lt(0).to_numpy(), max_pre_notes),
            (1, frame["dt_days"].eq(0).to_numpy(), max_dx_notes),
            (2, frame["dt_days"].gt(0).to_numpy(), max_post_notes),
        ):
            phase_positions = np.flatnonzero(mask)
            local = select_indices(
                phase_positions,
                limit,
                seed=stable_seed(seed, int(episode_id), phase_code),
            )
            output_positions.extend(phase_positions[local].tolist())
        chosen = frame.iloc[output_positions].sort_values(
            ["dt_days", "emb_row_idx"], kind="stable"
        )
        patient_values = chosen["patient_id"].unique()
        if len(patient_values) != 1:
            raise ValueError(f"Episode {episode_id} maps to multiple patients.")
        rows = chosen["emb_row_idx"].to_numpy(dtype=np.int64)
        days = chosen["dt_days"].to_numpy(dtype=np.float32)
        source_rows = chosen["source_note_row"].to_numpy(dtype=np.int64)
        episode_notes.append(
            EpisodeNotes(
                episode_id=int(episode_id),
                patient_id=int(patient_values[0]),
                rows=rows,
                days=days,
                source_rows=source_rows,
            )
        )
        for position, (row, day, source_row) in enumerate(
            zip(rows, days, source_rows, strict=True)
        ):
            selected_records.append(
                {
                    "episode_id": int(episode_id),
                    "note_sequence_position": position,
                    "phase": "pre" if day < 0 else "dx" if day == 0 else "post",
                    "emb_row_idx": int(row),
                    "source_note_row": int(source_row),
                    "dt_days": float(day),
                }
            )
    observed = [record.episode_id for record in episode_notes]
    expected = sorted(wanted)
    if observed != expected:
        raise ValueError("Fixed selection does not cover every eligible episode.")
    return episode_notes, pd.DataFrame.from_records(selected_records)


def combine_recurrence(labels: pd.DataFrame) -> pd.Series:
    columns = ["local_recurrence", "distant_recurrence", "regional_recurrence"]
    values = labels[columns].apply(pd.to_numeric, errors="coerce")
    result = pd.Series(np.nan, index=labels.index, dtype=np.float64)
    result.loc[values.eq(1).any(axis=1)] = 1.0
    result.loc[values.eq(0).all(axis=1)] = 0.0
    return result


def load_cohort(
    note_meta_path: Path,
    labels_path: Path,
    internal_meta_path: Path,
    *,
    min_pre_notes: int,
    include_overlap: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    note_columns = [
        "source_note_row",
        "patient_clinic_key",
        "label_row_id",
        "days_from_dx",
        "emb_row_idx",
    ]
    notes = pd.read_csv(note_meta_path, usecols=note_columns).rename(
        columns={
            "patient_clinic_key": "patient_id",
            "label_row_id": "episode_id",
            "days_from_dx": "dt_days",
        }
    )
    for column in ("source_note_row", "patient_id", "episode_id", "emb_row_idx"):
        numeric = pd.to_numeric(notes[column], errors="raise")
        if numeric.isna().any() or (numeric % 1).ne(0).any():
            raise ValueError(f"{column} must contain integer values.")
        notes[column] = numeric.astype("int64")
    notes["dt_days"] = pd.to_numeric(notes["dt_days"], errors="raise")
    if notes["emb_row_idx"].duplicated().any():
        raise ValueError("SEER note metadata has duplicate embedding row indices.")
    episode_patient_counts = notes.groupby("episode_id")["patient_id"].nunique()
    if episode_patient_counts.ne(1).any():
        raise ValueError("A SEER episode maps to more than one patient.")

    raw_labels = pd.read_csv(labels_path, usecols=list(LABEL_COLUMNS), low_memory=False)
    consistency = raw_labels.groupby("label_row_id", dropna=False)[
        [column for column in LABEL_COLUMNS if column != "label_row_id"]
    ].nunique(dropna=False)
    if consistency.gt(1).any().any():
        bad = int(consistency.gt(1).any(axis=1).sum())
        raise ValueError(f"{bad} episodes have inconsistent repeated label rows.")
    labels = raw_labels.drop_duplicates("label_row_id", keep="first").rename(
        columns={"label_row_id": "episode_id", "patient_clinic_key": "patient_id"}
    )
    labels["episode_id"] = pd.to_numeric(labels["episode_id"], errors="raise").astype("int64")
    labels["patient_id"] = pd.to_numeric(labels["patient_id"], errors="raise").astype("int64")
    labels["recurrence_any"] = combine_recurrence(labels)

    episode_counts = notes.groupby("episode_id").agg(
        patient_id=("patient_id", "first"),
        n_notes=("emb_row_idx", "size"),
        n_unique_source_notes=("source_note_row", "nunique"),
        n_pre_notes=("dt_days", lambda values: int((values < 0).sum())),
        n_dx_notes=("dt_days", lambda values: int((values == 0).sum())),
        n_post_notes=("dt_days", lambda values: int((values > 0).sum())),
    ).reset_index()
    manifest = episode_counts.merge(
        labels,
        on=["episode_id", "patient_id"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not manifest["_merge"].eq("both").all():
        raise ValueError("Some note episodes do not have matching label rows.")
    manifest = manifest.drop(columns="_merge")

    internal = pd.read_csv(internal_meta_path, usecols=["PATIENT_CLINIC_NUMBER"])
    internal_ids = set(
        pd.to_numeric(internal["PATIENT_CLINIC_NUMBER"], errors="raise").astype("int64")
    )
    manifest["development_overlap"] = manifest["patient_id"].isin(internal_ids)
    manifest["eligible_note_history"] = manifest["n_pre_notes"].ge(min_pre_notes)
    manifest["included"] = manifest["eligible_note_history"] & (
        include_overlap | ~manifest["development_overlap"]
    )

    patient_ids = sorted(manifest["patient_id"].unique())
    anonymous_patient = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    manifest["patient_group_id"] = manifest["patient_id"].map(anonymous_patient).astype(int)
    audit = {
        "n_note_rows": int(len(notes)),
        "n_episodes": int(len(manifest)),
        "n_patients": int(manifest["patient_id"].nunique()),
        "n_multi_episode_patients": int(
            manifest.groupby("patient_id").size().gt(1).sum()
        ),
        "n_development_overlap_episodes": int(manifest["development_overlap"].sum()),
        "n_development_overlap_patients": int(
            manifest.loc[manifest["development_overlap"], "patient_id"].nunique()
        ),
        "n_note_eligible_episodes": int(manifest["eligible_note_history"].sum()),
        "n_included_episodes": int(manifest["included"].sum()),
        "n_included_patients": int(
            manifest.loc[manifest["included"], "patient_id"].nunique()
        ),
        "min_pre_notes": int(min_pre_notes),
        "development_overlap_included": bool(include_overlap),
    }
    return notes, manifest, audit


def safe_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Drop the source patient identifier before writing cohort artifacts."""
    return manifest.drop(columns=["patient_id"])


def selection_sha256(selection: pd.DataFrame) -> str:
    columns = ["episode_id", "note_sequence_position", "emb_row_idx", "dt_days"]
    payload = selection[columns].to_csv(index=False, float_format="%.6g").encode()
    return hashlib.sha256(payload).hexdigest()


def validate_embedding_source(
    note_emb_path: Path,
    note_run_path: Path,
    notes: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.load(note_emb_path, mmap_mode="r")
    if matrix.ndim != 2 or matrix.shape[1] != 768:
        raise ValueError(f"Expected SEER note embeddings with shape (N, 768); got {matrix.shape}.")
    if len(matrix) != len(notes):
        raise ValueError("SEER note embedding and metadata row counts differ.")
    expected_rows = np.arange(len(notes), dtype=np.int64)
    if not np.array_equal(np.sort(notes["emb_row_idx"].to_numpy()), expected_rows):
        raise ValueError("SEER embedding row indices are not exactly 0..N-1.")
    sample_rows = np.unique(np.linspace(0, len(matrix) - 1, min(1024, len(matrix))).astype(int))
    sample = np.asarray(matrix[sample_rows], dtype=np.float32)
    if not np.isfinite(sample).all():
        raise ValueError("SEER note embeddings contain non-finite sampled values.")
    run = json.loads(note_run_path.read_text())
    if run.get("summary_col") != "summary":
        raise ValueError("Expected SEER embeddings generated from the summary column.")
    audit = {
        "shape": list(matrix.shape),
        "model": run.get("model"),
        "summary_col": run.get("summary_col"),
        "sample_max_l2_norm_error": float(
            np.max(np.abs(np.linalg.norm(sample, axis=1) - 1.0))
        ),
        "input_protocol_match_to_internal": False,
        "input_protocol_warning": (
            "SEER vectors use summary only; the frozen internal models were developed "
            "with summary plus paraphrased demographics. Results are exploratory until "
            "the note input protocol is matched."
        ),
    }
    return matrix, audit


def aggregate_pooling(
    note_embeddings: np.ndarray,
    episodes: list[EpisodeNotes],
    *,
    half_life_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    mean_rows: list[np.ndarray] = []
    decay_rows: list[np.ndarray] = []
    alpha = math.log(2.0) / half_life_days
    for episode in episodes:
        notes = np.asarray(note_embeddings[episode.rows], dtype=np.float32)
        weights = np.exp(-alpha * np.abs(episode.days.astype(np.float64)))
        mean_rows.append(notes.mean(axis=0))
        decay_rows.append(np.average(notes, axis=0, weights=weights))
    return normalized(np.stack(mean_rows)), normalized(np.stack(decay_rows))


def load_transformer(checkpoint_path: Path, device: torch.device) -> tuple[TemporalNoteTransformer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    architecture = checkpoint.get("architecture_version")
    if architecture != "continuous_time_reconstruction_v2":
        raise ValueError(f"Unsupported frozen Transformer architecture: {architecture!r}.")
    model = TemporalNoteTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint


def infer_transformer(
    model: TemporalNoteTransformer,
    note_embeddings: np.ndarray,
    episodes: list[EpisodeNotes],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    outputs: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    capacity = model.max_note_tokens
    with torch.inference_mode():
        for start in range(0, len(episodes), batch_size):
            batch = episodes[start : start + batch_size]
            vectors = np.zeros((len(batch), capacity, 768), dtype=np.float32)
            days = np.zeros((len(batch), capacity), dtype=np.float32)
            padding = np.ones((len(batch), capacity), dtype=bool)
            for row, episode in enumerate(batch):
                count = len(episode.rows)
                if count > capacity:
                    raise ValueError(f"Episode {episode.episode_id} exceeds model capacity.")
                vectors[row, :count] = np.asarray(
                    note_embeddings[episode.rows], dtype=np.float32
                )
                days[row, :count] = episode.days
                padding[row, :count] = False
            logits, embeddings = model(
                torch.from_numpy(vectors).to(device),
                torch.from_numpy(days).to(device),
                torch.from_numpy(padding).to(device),
            )
            outputs.append(embeddings.cpu().numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32), np.concatenate(predictions)


def apply_adapter(
    base: np.ndarray,
    checkpoint_path: Path,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ResidualMetricAdapter(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(base), batch_size):
            values = torch.from_numpy(base[start : start + batch_size]).to(device)
            outputs.append(model(values).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def validate_representation(matrix: np.ndarray, n_episodes: int, slug: str) -> None:
    if matrix.shape != (n_episodes, 768):
        raise ValueError(f"{slug} has unexpected shape {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{slug} contains non-finite values.")
    error = float(np.max(np.abs(np.linalg.norm(matrix, axis=1) - 1.0)))
    if error > 1e-4:
        raise ValueError(f"{slug} maximum L2 norm error is {error}.")


def rank_leave_patient_out(
    embeddings: np.ndarray,
    patient_groups: np.ndarray,
    *,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    cosine = normalized(embeddings) @ normalized(embeddings).T
    same_patient = patient_groups[:, None] == patient_groups[None, :]
    cosine[same_patient] = -np.inf
    available = (~same_patient).sum(axis=1)
    if np.any(available < max_k):
        raise ValueError("Too few other-patient gallery episodes for requested K.")
    top = np.argsort(-cosine, axis=1, kind="stable")[:, :max_k]
    return top, np.take_along_axis(cosine, top, axis=1)


def clustered_interval(
    values: np.ndarray,
    patient_groups: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples == 0 or len(values) == 0:
        return float("nan"), float("nan")
    frame = pd.DataFrame({"value": values, "patient": patient_groups})
    grouped = frame.groupby("patient", sort=True)["value"].agg(["sum", "count"])
    patient_sums = grouped["sum"].to_numpy()
    patient_counts = grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    sampled_groups = rng.integers(
        0, len(patient_sums), size=(samples, len(patient_sums))
    )
    draws = patient_sums[sampled_groups].sum(axis=1) / patient_counts[
        sampled_groups
    ].sum(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def evaluate_representation(
    slug: str,
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    *,
    endpoints: list[str],
    k_values: list[int],
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_groups = manifest["patient_group_id"].to_numpy(dtype=np.int64)
    episode_ids = manifest["episode_id"].to_numpy(dtype=np.int64)
    top, similarities = rank_leave_patient_out(
        embeddings, patient_groups, max_k=max(k_values)
    )
    neighbor_records: list[dict[str, Any]] = []
    for query_row in range(len(manifest)):
        for rank, neighbor_row in enumerate(top[query_row], start=1):
            neighbor_records.append(
                {
                    "method": slug,
                    "query_episode_id": int(episode_ids[query_row]),
                    "rank": rank,
                    "neighbor_episode_id": int(episode_ids[neighbor_row]),
                    "cosine_similarity": float(similarities[query_row, rank - 1]),
                }
            )

    per_query: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for endpoint in endpoints:
        if endpoint not in manifest:
            raise KeyError(f"Unknown endpoint column: {endpoint}")
        values = pd.to_numeric(manifest[endpoint], errors="coerce").to_numpy()
        valid_queries = np.flatnonzero(np.isfinite(values))
        for k in k_values:
            observed: list[float] = []
            chances: list[float] = []
            groups: list[int] = []
            targets: list[float] = []
            for query_row in valid_queries:
                neighbors = top[query_row, :k]
                precision = float(np.equal(values[neighbors], values[query_row]).sum() / k)
                gallery_mask = patient_groups != patient_groups[query_row]
                gallery_values = values[gallery_mask]
                chance = float(
                    np.equal(gallery_values, values[query_row]).mean()
                ) if len(gallery_values) else float("nan")
                observed.append(precision)
                chances.append(chance)
                groups.append(int(patient_groups[query_row]))
                targets.append(float(values[query_row]))
                per_query.append(
                    {
                        "method": slug,
                        "query_episode_id": int(episode_ids[query_row]),
                        "patient_group_id": int(patient_groups[query_row]),
                        "endpoint": endpoint,
                        "target_code": float(values[query_row]),
                        "k": k,
                        "precision_at_k": precision,
                        "prevalence_chance": chance,
                    }
                )
            observed_array = np.asarray(observed, dtype=float)
            chance_array = np.asarray(chances, dtype=float)
            group_array = np.asarray(groups, dtype=np.int64)
            target_array = np.asarray(targets, dtype=float)
            low, high = clustered_interval(
                observed_array,
                group_array,
                samples=bootstrap_samples,
                seed=stable_seed(seed, slug, endpoint, k, "precision"),
            )
            class_means = pd.DataFrame(
                {"target": target_array, "precision": observed_array}
            ).groupby("target")["precision"].mean()
            mean_precision = float(observed_array.mean()) if len(observed_array) else float("nan")
            mean_chance = float(chance_array.mean()) if len(chance_array) else float("nan")
            summary.append(
                {
                    "method": slug,
                    "endpoint": endpoint,
                    "k": k,
                    "n_queries": int(len(observed_array)),
                    "n_patients": int(len(np.unique(group_array))),
                    "n_target_codes": int(len(class_means)),
                    "mean_precision_at_k": mean_precision,
                    "ci95_low": low,
                    "ci95_high": high,
                    "macro_target_precision_at_k": (
                        float(class_means.mean()) if len(class_means) else float("nan")
                    ),
                    "prevalence_chance": mean_chance,
                    "normalized_lift": (
                        (mean_precision - mean_chance) / (1.0 - mean_chance)
                        if np.isfinite(mean_chance) and mean_chance < 1.0
                        else float("nan")
                    ),
                }
            )
    return (
        pd.DataFrame.from_records(summary),
        pd.DataFrame.from_records(per_query),
        pd.DataFrame.from_records(neighbor_records),
    )


def comparison_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create method-level P@5 ranks and matched DML-minus-base differences."""
    p5 = summary.loc[summary["k"].eq(5)]
    method_summary = (
        p5.groupby("method", sort=False)
        .agg(
            endpoint_macro_precision_at_5=("mean_precision_at_k", "mean"),
            endpoint_macro_target_precision_at_5=(
                "macro_target_precision_at_k",
                "mean",
            ),
            endpoint_macro_normalized_lift=("normalized_lift", "mean"),
            n_endpoints=("endpoint", "nunique"),
        )
        .reset_index()
        .sort_values("endpoint_macro_precision_at_5", ascending=False)
    )
    adapted = summary.loc[summary["method"].str.contains("_icd_")].copy()
    adapted["base_method"] = adapted["method"].str.split("_icd_", n=1).str[0]
    base = summary.rename(
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
    incremental = adapted.merge(
        base,
        on=["base_method", "endpoint", "k"],
        how="left",
        validate="many_to_one",
    )
    if incremental["base_mean_precision_at_k"].isna().any():
        raise ValueError("A Stage 03 method is missing its matched Stage 02 baseline.")
    incremental["dml_minus_base_precision_at_k"] = (
        incremental["mean_precision_at_k"]
        - incremental["base_mean_precision_at_k"]
    )
    incremental["dml_minus_base_macro_target_precision_at_k"] = (
        incremental["macro_target_precision_at_k"]
        - incremental["base_macro_target_precision_at_k"]
    )
    return method_summary, incremental


def stage04_audit(stage04_root: Path, labels: pd.DataFrame) -> dict[str, Any]:
    config_path = stage04_root / "transformer_combined/dmon_3/run_config.json"
    if not config_path.is_file():
        return {
            "status": "unavailable",
            "reason": f"Reference Stage 04 run configuration is missing: {config_path}",
        }
    config = json.loads(config_path.read_text())
    required = config["graph_construction"]["feature_columns"]
    available = [column for column in required if column in labels.columns]
    missing = [column for column in required if column not in labels.columns]
    return {
        "status": "unavailable" if missing else "available",
        "frozen_inference_performed": False,
        "reference_run_config": str(config_path.resolve()),
        "required_graph_feature_count": len(required),
        "available_graph_feature_count": len(available),
        "missing_graph_feature_count": len(missing),
        "missing_graph_features": missing,
        "reason": (
            "Frozen Stage 04 incoming-edge inference requires the same 31 "
            "Ki-67/radiology indicators used internally; they are absent from "
            "the SEER-linked label table."
            if missing
            else "Graph features exist, but external inference is not implemented here."
        ),
    }


def prepare_output(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_dir}; pass --overwrite."
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.adapter_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if args.min_pre_notes < 0 or min(
        args.max_pre_notes, args.max_dx_notes, args.max_post_notes
    ) < 0:
        raise ValueError("Note eligibility and phase limits must be non-negative.")
    if args.half_life_days <= 0:
        raise ValueError("--half-life-days must be positive.")
    k_values = sorted(set(args.k))
    if not k_values or min(k_values) < 1:
        raise ValueError("Every K must be positive.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative.")

    internal_meta = args.stage02_root / "mean/patient.meta.csv"
    required = [
        args.note_emb_path,
        args.note_meta_path,
        args.note_run_path,
        args.labels_path,
        internal_meta,
    ]
    for relative in TRANSFORMER_DIRS.values():
        required.append(args.stage02_root / relative / "best.pt")
    for base in BASES:
        for objective in OBJECTIVES:
            required.append(args.stage03_root / base / objective / "best.pt")
            required.append(args.stage03_root / base / objective / "run_config.json")
    require_files(required)
    prepare_output(args.output_dir, overwrite=args.overwrite)

    notes, manifest, cohort_audit = load_cohort(
        args.note_meta_path,
        args.labels_path,
        internal_meta,
        min_pre_notes=args.min_pre_notes,
        include_overlap=args.include_development_overlap,
    )
    note_embeddings, embedding_audit = validate_embedding_source(
        args.note_emb_path, args.note_run_path, notes
    )
    eligible = manifest.loc[manifest["included"]].sort_values("episode_id").reset_index(drop=True)
    episodes, selection = fixed_episode_selection(
        notes,
        eligible,
        seed=args.seed,
        max_pre_notes=args.max_pre_notes,
        max_dx_notes=args.max_dx_notes,
        max_post_notes=args.max_post_notes,
    )
    if len(eligible) <= max(k_values):
        raise ValueError("Too few eligible episodes for the requested retrieval K.")
    selection_hash = selection_sha256(selection)
    eligible_safe = safe_manifest(eligible)
    full_safe = safe_manifest(manifest)
    full_safe.to_csv(args.output_dir / "cohort_flow.csv", index=False)
    eligible_safe.to_csv(args.output_dir / "episode_metadata.csv", index=False)
    selection.to_csv(args.output_dir / "note_selection.csv", index=False)
    stage04 = stage04_audit(args.stage04_root, manifest)
    atomic_json(args.output_dir / "stage04_availability.json", stage04)
    audit = {
        "cohort": cohort_audit,
        "note_embeddings": embedding_audit,
        "selection": {
            "sha256": selection_hash,
            "n_selected_notes": int(len(selection)),
            "max_pre_notes": args.max_pre_notes,
            "max_dx_notes": args.max_dx_notes,
            "max_post_notes": args.max_post_notes,
            "strategy": "deterministic random without replacement per episode and phase",
            "seed": args.seed,
        },
        "stage04": stage04,
    }
    atomic_json(args.output_dir / "data_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return

    device = resolve_device(args.device)
    print(f"Frozen inference device: {device}", flush=True)
    representations: dict[str, np.ndarray] = {}
    transformer_predictions: dict[str, np.ndarray] = {}
    mean, decay = aggregate_pooling(
        note_embeddings, episodes, half_life_days=args.half_life_days
    )
    representations["mean"] = mean
    representations["time_decay"] = decay

    checkpoint_hashes: dict[str, str] = {}
    for slug, relative in TRANSFORMER_DIRS.items():
        path = args.stage02_root / relative / "best.pt"
        print(f"Inferring {slug} ...", flush=True)
        model, checkpoint = load_transformer(path, device)
        limits = checkpoint["model_config"]
        expected_limits = (
            int(limits["max_pre_notes"]),
            int(limits["max_dx_notes"]),
            int(limits["max_post_notes"]),
        )
        requested_limits = (
            args.max_pre_notes,
            args.max_dx_notes,
            args.max_post_notes,
        )
        if expected_limits != requested_limits:
            raise ValueError(
                f"{slug} expects phase limits {expected_limits}, got {requested_limits}."
            )
        embeddings, predictions = infer_transformer(
            model,
            note_embeddings,
            episodes,
            device=device,
            batch_size=args.batch_size,
        )
        representations[slug] = embeddings
        transformer_predictions[slug] = predictions
        checkpoint_hashes[slug] = sha256_file(path)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for base in BASES:
        for objective in OBJECTIVES:
            slug = f"{base}_icd_{objective}"
            model_dir = args.stage03_root / base / objective
            run_config = json.loads((model_dir / "run_config.json").read_text())
            if run_config.get("architecture") != "base_embedding_only_residual_mlp_v1":
                raise ValueError(f"{slug} is not deployable without ICD input.")
            if run_config.get("base_name") != base:
                raise ValueError(f"{slug} run configuration base mismatch.")
            path = model_dir / "best.pt"
            print(f"Inferring {slug} ...", flush=True)
            representations[slug] = apply_adapter(
                representations[base],
                path,
                device=device,
                batch_size=args.adapter_batch_size,
            )
            checkpoint_hashes[slug] = sha256_file(path)

    for slug, matrix in representations.items():
        validate_representation(matrix, len(eligible), slug)
        path = args.output_dir / "representations" / slug / "episode.emb.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
    prediction_frame = eligible_safe[["episode_id", "patient_group_id"]].copy()
    for slug, values in transformer_predictions.items():
        prediction_frame[f"{slug}_predicted_stage_0_to_4"] = values
    prediction_frame.to_csv(args.output_dir / "transformer_stage_predictions.csv", index=False)

    summaries: list[pd.DataFrame] = []
    per_queries: list[pd.DataFrame] = []
    neighbors: list[pd.DataFrame] = []
    for slug, matrix in representations.items():
        print(f"Evaluating {slug} ...", flush=True)
        summary, per_query, neighbor = evaluate_representation(
            slug,
            matrix,
            eligible,
            endpoints=args.endpoints,
            k_values=k_values,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        summaries.append(summary)
        per_queries.append(per_query)
        neighbors.append(neighbor)
    summary = pd.concat(summaries, ignore_index=True)
    per_query = pd.concat(per_queries, ignore_index=True)
    neighbor = pd.concat(neighbors, ignore_index=True)
    summary.to_csv(args.output_dir / "retrieval_summary.csv", index=False)
    per_query.to_csv(args.output_dir / "retrieval_per_query.csv", index=False)
    neighbor.to_csv(args.output_dir / "retrieval_neighbors.csv", index=False)
    p5 = summary.loc[summary["k"].eq(5)].pivot(
        index="method", columns="endpoint", values="mean_precision_at_k"
    )
    p5.to_csv(args.output_dir / "retrieval_p5_wide.csv")
    method_summary, incremental = comparison_tables(summary)
    method_summary.to_csv(args.output_dir / "method_summary_p5.csv", index=False)
    incremental.to_csv(args.output_dir / "retrieval_dml_incremental.csv", index=False)

    report = {
        "status": "complete",
        "analysis_role": "exploratory frozen external-cohort evaluation",
        "unit": "diagnosis episode",
        "retrieval_design": (
            "within-SEER leave-one-patient-out; all episodes belonging to the query "
            "patient are excluded from its gallery"
        ),
        "n_methods": len(representations),
        "stage02_methods": list(BASES),
        "stage03_objectives_per_base": list(OBJECTIVES),
        "stage04": stage04,
        "endpoints": args.endpoints,
        "endpoint_semantics": (
            "Exact equality of raw registry codes; no unverified code-to-clinical-label "
            "mapping is applied. recurrence_any is 1 if any site is 1, 0 only if all "
            "three site indicators are observed and 0, otherwise missing."
        ),
        "k": k_values,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "unit": "patient_group_id (episodes clustered within patient)",
            "interval": "percentile 95%",
            "seed": args.seed,
        },
        "selection_sha256": selection_hash,
        "note_embedding_sha256": sha256_file(args.note_emb_path),
        "checkpoint_sha256": checkpoint_hashes,
        "input_protocol_match_to_internal": False,
        "limitations": [
            embedding_audit["input_protocol_warning"],
            "Within-SEER retrieval is secondary; a primary external-query to frozen "
            "internal-gallery analysis requires a prespecified registry-code harmonization.",
            stage04["reason"],
        ],
    }
    atomic_json(args.output_dir / "run_config.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
