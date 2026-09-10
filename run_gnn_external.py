#!/usr/bin/env python3
"""Run frozen stage+radiology GNN models on another institution's cohort.

This entry point is the portable counterpart to ``run_gnn.py``. It does not
read SEER files or parse local radiology narratives. The external institution
provides an episode table, a table of already-derived binary graph features,
and the representation arrays produced for those episodes.

No model is fitted or selected on the external cohort. External nodes receive
messages from frozen internal-training nodes only; external-to-external and
external-to-training edges are intentionally absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from zhemin.pipeline.stage_04_gnn.evaluate import (
    infer_gnn_outputs,
    load_trained_model,
)
from zhemin.pipeline.stage_04_gnn.train import (
    DMoN,  # noqa: F401 - required when loading full-object checkpoints
    DMoNModel,  # noqa: F401 - required when loading full-object checkpoints
    RADIOLOGY_FEATURE_COLUMNS,
    STAGE_FEATURE_COLUMNS,
)

from .run_all import (
    BASES,
    OBJECTIVES,
    atomic_json,
    evaluate_representation,
    normalized,
    resolve_device,
    sha256_file,
)


FEATURE_COLUMNS = STAGE_FEATURE_COLUMNS + RADIOLOGY_FEATURE_COLUMNS
DEFAULT_OUTPUT = Path("external_gnn_results")
DEFAULT_EMBEDDING_TEMPLATE = "representations/{pipeline}/episode.emb.npy"
DEFAULT_MISSING_LABELS = ("", "unknown", "nan", "na", "n/a", "not available")


@dataclass(frozen=True)
class ExternalCohort:
    """Validated and privacy-safe external data aligned to embedding rows."""

    manifest: pd.DataFrame
    graph_features: pd.DataFrame
    embedding_rows: np.ndarray
    endpoint_mappings: dict[str, dict[str, str]]
    audit: dict[str, Any]


def available_pipelines() -> list[str]:
    pipelines = list(BASES)
    pipelines.extend(
        f"{base}_icd_{objective}" for base in BASES for objective in OBJECTIVES
    )
    return pipelines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        help="Root containing <pipeline>/<model-subdir>/ frozen model artifacts.",
    )
    parser.add_argument(
        "--training-graph-features-path",
        type=Path,
        help=(
            "Deidentified .npy or .csv matrix of the 35 internal-training graph "
            "features, in the exact node order used by graph.bin."
        ),
    )
    parser.add_argument(
        "--representation-root",
        type=Path,
        help="Root containing the external representation arrays.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        help="Episode metadata in the same row order as every representation array.",
    )
    parser.add_argument(
        "--graph-features-path",
        type=Path,
        help="One-row-per-episode CSV containing the 35 binary graph features.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-map",
        type=Path,
        help=(
            "Optional JSON object mapping canonical feature names to institution "
            "column names. Unmapped features retain their canonical names."
        ),
    )
    parser.add_argument("--episode-id-column", default="episode_id")
    parser.add_argument(
        "--graph-episode-id-column",
        help="Episode ID in the graph-feature CSV; defaults to --episode-id-column.",
    )
    parser.add_argument("--patient-id-column", default="patient_id")
    parser.add_argument(
        "--eligibility-column",
        help="Optional metadata column; only rows with a true value are evaluated.",
    )
    parser.add_argument(
        "--require-radiology-report",
        action="store_true",
        help="Keep only episodes whose radiology-count column is greater than zero.",
    )
    parser.add_argument("--radiology-count-column", default="n_radiology_reports")
    parser.add_argument(
        "--endpoints",
        nargs="+",
        help="Metadata columns to evaluate by exact within-column label agreement.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=available_pipelines(),
        help="Frozen representation/GNN pipelines to evaluate.",
    )
    parser.add_argument("--model-subdir", default="dmon_3")
    parser.add_argument(
        "--embedding-template",
        default=DEFAULT_EMBEDDING_TEMPLATE,
        help="Path template below representation-root; must contain {pipeline}.",
    )
    parser.add_argument(
        "--missing-label-values",
        nargs="+",
        default=list(DEFAULT_MISSING_LABELS),
        help="Case-insensitive endpoint values treated as missing.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate and summarize external inputs without loading GNN models.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--print-required-columns",
        action="store_true",
        help="Print the canonical graph-feature schema and exit.",
    )
    return parser.parse_args()


def require_runtime_args(args: argparse.Namespace) -> None:
    required = {
        "--model-root": args.model_root,
        "--training-graph-features-path": args.training_graph_features_path,
        "--representation-root": args.representation_root,
        "--metadata-path": args.metadata_path,
        "--graph-features-path": args.graph_features_path,
        "--endpoints": args.endpoints,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    if "{pipeline}" not in args.embedding_template:
        raise ValueError("--embedding-template must contain {pipeline}.")


def graph_hash(source: np.ndarray, destination: np.ndarray, weights: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(source.astype(np.int64, copy=False).tobytes())
    digest.update(destination.astype(np.int64, copy=False).tobytes())
    digest.update(weights.astype(np.float32, copy=False).tobytes())
    return digest.hexdigest()


def build_frozen_inference_graph(
    training_features: np.ndarray,
    external_features: np.ndarray,
    *,
    threshold: float,
) -> tuple[dgl.DGLGraph, dict[str, Any]]:
    """Build train-to-train and train-to-external edges only."""
    combined = np.concatenate([training_features, external_features]).astype(np.float64)
    if not np.isin(combined, (0.0, 1.0)).all():
        raise ValueError("All stage+radiology graph features must be binary (0 or 1).")
    if np.any(np.linalg.norm(combined, axis=1) == 0):
        raise ValueError("Every graph node must have at least one active feature.")

    similarities = np.round(cosine_similarity(combined), 2)
    source, destination = np.nonzero(similarities >= threshold)

    # External nodes are inference targets, never message sources.
    keep = source < len(training_features)
    source = source[keep].astype(np.int64, copy=False)
    destination = destination[keep].astype(np.int64, copy=False)
    weights = similarities[source, destination].astype(np.float32)

    graph = dgl.graph(
        (torch.from_numpy(source), torch.from_numpy(destination)),
        num_nodes=len(combined),
    )
    graph.edata["weights"] = torch.from_numpy(weights[:, None])

    external_destinations = destination[destination >= len(training_features)]
    incoming = np.bincount(
        external_destinations - len(training_features),
        minlength=len(external_features),
    )
    if np.any(incoming == 0):
        count = int((incoming == 0).sum())
        raise ValueError(
            f"{count} external episodes have no incoming training-node edge. "
            "Check graph-feature harmonization and the frozen threshold."
        )

    return graph, {
        "n_training_nodes": len(training_features),
        "n_external_nodes": len(external_features),
        "n_training_to_training_edges": int(
            (destination < len(training_features)).sum()
        ),
        "n_training_to_external_edges": int(
            (destination >= len(training_features)).sum()
        ),
        "n_external_source_edges": 0,
        "minimum_external_in_degree": int(incoming.min()),
        "median_external_in_degree": float(np.median(incoming)),
        "maximum_external_in_degree": int(incoming.max()),
        "threshold": threshold,
        "rounding_decimals": 2,
        "topology_sha256": graph_hash(source, destination, weights),
    }


def verify_training_topology(
    graph: dgl.DGLGraph,
    saved_graph_path: Path,
    n_training: int,
) -> dict[str, Any]:
    """Confirm that adding external targets did not alter the training graph."""
    saved_graphs, _ = dgl.load_graphs(str(saved_graph_path))
    saved = saved_graphs[0]
    source, destination = graph.edges()
    keep = (source < n_training) & (destination < n_training)
    inferred_pairs = torch.stack([source[keep], destination[keep]], dim=1).numpy()
    saved_source, saved_destination = saved.edges()
    saved_pairs = torch.stack([saved_source, saved_destination], dim=1).numpy()
    if not np.array_equal(inferred_pairs, saved_pairs):
        raise ValueError(
            "The supplied training graph features do not reproduce graph.bin. "
            "Verify their row order and values."
        )
    return {
        "saved_training_edges": int(saved.num_edges()),
        "inference_training_edges": int(keep.sum()),
        "edge_order_and_topology_identical": True,
    }


def read_csv_with_string_id(path: Path, id_column: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    if id_column not in header.columns:
        raise KeyError(f"{path} is missing ID column {id_column!r}.")
    return pd.read_csv(path, dtype={id_column: "string"})


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
    normalized_values = values.astype("string").str.strip().str.lower().fillna("")
    accepted = {"1", "1.0", "true", "t", "yes", "y"}
    rejected = {"0", "0.0", "false", "f", "no", "n", "", "<na>"}
    unexpected = sorted(set(normalized_values).difference(accepted | rejected))
    if unexpected:
        raise ValueError(f"{name} has unrecognized boolean values: {unexpected[:5]}")
    return normalized_values.isin(accepted)


def read_feature_map(path: Path | None) -> dict[str, str]:
    mapping = {column: column for column in FEATURE_COLUMNS}
    if path is None:
        return mapping
    supplied = json.loads(path.read_text())
    if not isinstance(supplied, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in supplied.items()
    ):
        raise TypeError("--feature-map must contain one JSON string-to-string object.")
    unknown = sorted(set(supplied).difference(FEATURE_COLUMNS))
    if unknown:
        raise KeyError(f"Feature map contains unknown canonical features: {unknown}")
    mapping.update(supplied)
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Two canonical graph features map to the same source column.")
    return mapping


def encode_endpoint(
    values: pd.Series,
    *,
    missing_labels: set[str],
) -> tuple[pd.Series, dict[str, str]]:
    """Encode arbitrary external labels while preserving exact-match semantics."""
    as_text = values.astype("string").str.strip()
    missing = values.isna() | as_text.str.lower().isin(missing_labels)
    observed = sorted(as_text.loc[~missing].unique().tolist())
    code_by_label = {label: code for code, label in enumerate(observed)}
    encoded = as_text.map(code_by_label).astype("float64")
    encoded.loc[missing] = np.nan
    mapping = {str(code): label for label, code in code_by_label.items()}
    return encoded, mapping


def load_external_cohort(args: argparse.Namespace, *, max_k: int) -> ExternalCohort:
    graph_id_column = args.graph_episode_id_column or args.episode_id_column
    metadata = read_csv_with_string_id(args.metadata_path, args.episode_id_column)
    graph_source = read_csv_with_string_id(args.graph_features_path, graph_id_column)

    required_metadata = {args.episode_id_column, args.patient_id_column, *args.endpoints}
    if args.eligibility_column:
        required_metadata.add(args.eligibility_column)
    missing_metadata = sorted(required_metadata.difference(metadata.columns))
    if missing_metadata:
        raise KeyError(f"Metadata is missing columns: {missing_metadata}")

    metadata = metadata.copy()
    metadata["_join_id"] = clean_ids(
        metadata[args.episode_id_column], name=args.episode_id_column
    )
    if metadata["_join_id"].duplicated().any():
        raise ValueError("Metadata must contain exactly one row per episode ID.")
    metadata["_embedding_row"] = np.arange(len(metadata), dtype=np.int64)

    selected = pd.Series(True, index=metadata.index)
    if args.eligibility_column:
        selected &= true_mask(
            metadata[args.eligibility_column], name=args.eligibility_column
        )

    graph_source = graph_source.copy()
    graph_source["_join_id"] = clean_ids(
        graph_source[graph_id_column], name=graph_id_column
    )
    if graph_source["_join_id"].duplicated().any():
        raise ValueError("Graph features must contain exactly one row per episode ID.")

    feature_map = read_feature_map(args.feature_map)
    missing_features = sorted(set(feature_map.values()).difference(graph_source.columns))
    if missing_features:
        raise KeyError(f"Graph-feature CSV is missing columns: {missing_features}")

    if args.require_radiology_report:
        if args.radiology_count_column not in graph_source:
            raise KeyError(
                "--require-radiology-report needs graph column "
                f"{args.radiology_count_column!r}."
            )
        report_counts = pd.to_numeric(
            graph_source[args.radiology_count_column], errors="raise"
        )
        ids_with_reports = set(graph_source.loc[report_counts.gt(0), "_join_id"])
        selected &= metadata["_join_id"].isin(ids_with_reports)

    selected_metadata = metadata.loc[selected].reset_index(drop=True)
    if selected_metadata.empty:
        raise ValueError("No external episodes remain after eligibility filters.")

    graph_by_id = graph_source.set_index("_join_id", verify_integrity=True)
    missing_graph_ids = sorted(
        set(selected_metadata["_join_id"]).difference(graph_by_id.index)
    )
    if missing_graph_ids:
        raise ValueError(
            f"Graph features are missing {len(missing_graph_ids)} selected episodes."
        )
    aligned_graph = graph_by_id.loc[selected_metadata["_join_id"]]
    graph_features = pd.DataFrame(
        {
            canonical: pd.to_numeric(aligned_graph[source], errors="raise").to_numpy()
            for canonical, source in feature_map.items()
        }
    )
    if graph_features.isna().any().any():
        raise ValueError("Graph features cannot contain missing values.")
    graph_array = graph_features.to_numpy(dtype=np.float64)
    if not np.isin(graph_array, (0.0, 1.0)).all():
        raise ValueError("Graph features must contain only 0 and 1.")
    stage_sum = graph_features.loc[:, STAGE_FEATURE_COLUMNS].sum(axis=1)
    if not stage_sum.eq(1).all():
        count = int(stage_sum.ne(1).sum())
        raise ValueError(
            f"{count} episodes do not have exactly one active broad-stage feature."
        )

    patient_ids = clean_ids(
        selected_metadata[args.patient_id_column], name=args.patient_id_column
    )
    patient_codes, patient_values = pd.factorize(patient_ids, sort=True)
    patient_groups = patient_codes.astype(np.int64) + 1
    n_external = len(selected_metadata)
    smallest_gallery = n_external - pd.Series(patient_groups).value_counts().max()
    if smallest_gallery < max_k:
        raise ValueError(
            f"At least one patient has only {smallest_gallery} other-patient episodes "
            f"available, fewer than max K={max_k}."
        )

    manifest = pd.DataFrame(
        {
            "episode_id": np.arange(1, n_external + 1, dtype=np.int64),
            "patient_group_id": patient_groups,
        }
    )
    missing_labels = {value.strip().lower() for value in args.missing_label_values}
    endpoint_mappings: dict[str, dict[str, str]] = {}
    for endpoint in args.endpoints:
        manifest[endpoint], endpoint_mappings[endpoint] = encode_endpoint(
            selected_metadata[endpoint], missing_labels=missing_labels
        )

    return ExternalCohort(
        manifest=manifest,
        graph_features=graph_features,
        embedding_rows=selected_metadata["_embedding_row"].to_numpy(dtype=np.int64),
        endpoint_mappings=endpoint_mappings,
        audit={
            "n_metadata_rows": len(metadata),
            "n_selected_episodes": n_external,
            "n_selected_patients": int(len(patient_values)),
            "source_episode_ids_written": False,
            "source_patient_ids_written": False,
            "eligibility_column": args.eligibility_column,
            "radiology_report_required": args.require_radiology_report,
            "n_stage_features": len(STAGE_FEATURE_COLUMNS),
            "n_radiology_features": len(RADIOLOGY_FEATURE_COLUMNS),
            "endpoint_nonmissing_counts": {
                endpoint: int(manifest[endpoint].notna().sum())
                for endpoint in args.endpoints
            },
        },
    )


def load_training_graph_features(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        features = np.load(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        missing = sorted(set(FEATURE_COLUMNS).difference(frame.columns))
        if missing:
            raise KeyError(f"Training graph-feature CSV is missing columns: {missing}")
        features = frame.loc[:, FEATURE_COLUMNS].to_numpy()
    else:
        raise ValueError("Training graph features must be a .npy or .csv file.")
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            "Training graph features must have shape "
            f"(n_training_nodes, {len(FEATURE_COLUMNS)}); got {features.shape}."
        )
    if not np.isin(features, (0.0, 1.0)).all():
        raise ValueError("Training graph features must contain only 0 and 1.")
    return features


def model_directory(args: argparse.Namespace, pipeline: str) -> Path:
    return args.model_root / pipeline / args.model_subdir


def representation_path(args: argparse.Namespace, pipeline: str) -> Path:
    rendered = Path(args.embedding_template.format(pipeline=pipeline))
    return rendered if rendered.is_absolute() else args.representation_root / rendered


def validate_model_matrix(args: argparse.Namespace) -> None:
    required_names = (
        "model.pth",
        "run_config.json",
        "graph.bin",
        "embeddings_org.npy",
        "embeddings_gnn.npy",
    )
    missing: list[str] = []
    for pipeline in args.pipelines:
        directory = model_directory(args, pipeline)
        missing.extend(
            str(directory / name)
            for name in required_names
            if not (directory / name).is_file()
        )
        embedding = representation_path(args, pipeline)
        if not embedding.is_file():
            missing.append(str(embedding))
    if missing:
        preview = missing[:5]
        suffix = " ..." if len(missing) > len(preview) else ""
        raise FileNotFoundError(f"Missing frozen evaluation artifacts: {preview}{suffix}")


def gnn_incremental(summary: pd.DataFrame) -> pd.DataFrame:
    gnn = summary.loc[summary["method"].str.startswith("gnn__")].copy()
    gnn["pipeline"] = gnn["method"].str.removeprefix("gnn__")
    original = summary.loc[summary["method"].str.startswith("input__")].copy()
    original["pipeline"] = original["method"].str.removeprefix("input__")
    original = original.rename(
        columns={
            "mean_precision_at_k": "input_mean_precision_at_k",
            "macro_target_precision_at_k": "input_macro_target_precision_at_k",
        }
    )[
        [
            "pipeline",
            "endpoint",
            "k",
            "input_mean_precision_at_k",
            "input_macro_target_precision_at_k",
        ]
    ]
    merged = gnn.merge(
        original, on=["pipeline", "endpoint", "k"], validate="one_to_one"
    )
    merged["gnn_minus_input_precision_at_k"] = (
        merged["mean_precision_at_k"] - merged["input_mean_precision_at_k"]
    )
    merged["gnn_minus_input_macro_target_precision_at_k"] = (
        merged["macro_target_precision_at_k"]
        - merged["input_macro_target_precision_at_k"]
    )
    return merged


def wide_precision_table(
    summary: pd.DataFrame,
    *,
    method_prefix: str,
    pipelines: list[str],
    endpoints: list[str],
    k_values: list[int],
) -> pd.DataFrame:
    selected = summary.loc[
        summary["method"].str.startswith(method_prefix),
        ["method", "endpoint", "k", "mean_precision_at_k"],
    ].copy()
    selected["method"] = selected["method"].str.removeprefix(method_prefix)
    wide = selected.pivot(
        index="method", columns=["endpoint", "k"], values="mean_precision_at_k"
    )
    expected = pd.MultiIndex.from_product(
        [endpoints, k_values], names=["endpoint", "k"]
    )
    expected_rows = pd.MultiIndex.from_product(
        [pipelines, endpoints, k_values], names=["method", "endpoint", "k"]
    )
    observed_rows = pd.MultiIndex.from_frame(selected[["method", "endpoint", "k"]])
    if not expected_rows.isin(observed_rows).all():
        raise ValueError(f"Incomplete {method_prefix!r} endpoint/K results.")
    wide = wide.reindex(index=pipelines, columns=expected)
    wide.columns = [f"{endpoint}_P@{k}" for endpoint, k in wide.columns]
    return wide.rename_axis("pipeline").reset_index()


def write_evaluation_outputs(
    output_dir: Path,
    summaries: list[pd.DataFrame],
    per_queries: list[pd.DataFrame],
    neighbors: list[pd.DataFrame],
    assignments: pd.DataFrame,
    *,
    pipelines: list[str],
    endpoints: list[str],
    k_values: list[int],
) -> pd.DataFrame:
    summary = pd.concat(summaries, ignore_index=True)
    pd.concat(per_queries, ignore_index=True).to_csv(
        output_dir / "retrieval_per_query.csv", index=False
    )
    pd.concat(neighbors, ignore_index=True).to_csv(
        output_dir / "retrieval_neighbors.csv", index=False
    )
    summary.to_csv(output_dir / "retrieval_summary.csv", index=False)
    gnn_incremental(summary).to_csv(
        output_dir / "retrieval_gnn_incremental.csv", index=False
    )
    assignments.to_csv(output_dir / "cluster_assignments.csv", index=False)

    for prefix, filename in (
        ("input__", "retrieval_before_gnn_wide.csv"),
        ("gnn__", "retrieval_after_gnn_wide.csv"),
    ):
        wide_precision_table(
            summary,
            method_prefix=prefix,
            pipelines=pipelines,
            endpoints=endpoints,
            k_values=k_values,
        ).to_csv(output_dir / filename, index=False, float_format="%.4f")
    if 5 in k_values:
        summary.loc[summary["k"].eq(5)].pivot(
            index="method", columns="endpoint", values="mean_precision_at_k"
        ).to_csv(output_dir / "retrieval_p5_wide.csv")
    return summary


def main() -> None:
    args = parse_args()
    if args.print_required_columns:
        print(json.dumps({"graph_feature_columns": list(FEATURE_COLUMNS)}, indent=2))
        return
    require_runtime_args(args)

    k_values = sorted(set(args.k))
    if not k_values or min(k_values) < 1:
        raise ValueError("Every K must be positive.")
    if args.bootstrap_samples < 0:
        raise ValueError("--bootstrap-samples cannot be negative.")
    if len(set(args.pipelines)) != len(args.pipelines):
        raise ValueError("--pipelines contains duplicates.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_external_cohort(args, max_k=max(k_values))
    safe_graph_features = cohort.graph_features.copy()
    safe_graph_features.insert(0, "episode_id", cohort.manifest["episode_id"])
    safe_graph_features.to_csv(args.output_dir / "episode_graph_features.csv", index=False)
    cohort.manifest.to_csv(args.output_dir / "episode_metadata.csv", index=False)
    input_audit = {
        "cohort": cohort.audit,
        "graph_feature_count": len(FEATURE_COLUMNS),
        "graph_features": list(FEATURE_COLUMNS),
        "endpoint_label_codes": cohort.endpoint_mappings,
        "metadata_sha256": sha256_file(args.metadata_path),
        "graph_features_sha256": sha256_file(args.graph_features_path),
    }
    atomic_json(args.output_dir / "data_audit.json", input_audit)
    print(json.dumps(input_audit, indent=2), flush=True)
    if args.audit_only:
        return

    validate_model_matrix(args)
    training_graph_features = load_training_graph_features(
        args.training_graph_features_path
    )
    reference_dir = model_directory(args, args.pipelines[0])
    reference_config = json.loads((reference_dir / "run_config.json").read_text())
    graph_config = reference_config.get("graph_construction", {})
    if graph_config.get("feature_preset") != "stage_radiology":
        raise ValueError("Frozen models must use the stage_radiology feature preset.")
    if tuple(graph_config.get("feature_columns", ())) != FEATURE_COLUMNS:
        raise ValueError("Frozen model graph-feature order differs from this script.")

    graph, graph_stats = build_frozen_inference_graph(
        training_graph_features,
        cohort.graph_features.to_numpy(dtype=np.float64),
        threshold=float(graph_config["threshold"]),
    )
    topology_check = verify_training_topology(
        graph, reference_dir / "graph.bin", len(training_graph_features)
    )
    dgl.save_graphs(str(args.output_dir / "inference_graph.bin"), [graph])

    device = resolve_device(args.device)
    summaries: list[pd.DataFrame] = []
    per_queries: list[pd.DataFrame] = []
    neighbors: list[pd.DataFrame] = []
    assignments = cohort.manifest[["episode_id", "patient_group_id"]].copy()
    model_checks: dict[str, Any] = {}

    for pipeline in args.pipelines:
        print(f"Frozen GNN inference: {pipeline}", flush=True)
        directory = model_directory(args, pipeline)
        config = json.loads((directory / "run_config.json").read_text())
        current_graph_config = config.get("graph_construction", {})
        if current_graph_config.get("feature_preset") != "stage_radiology":
            raise ValueError(f"{pipeline} does not use stage_radiology features.")
        if tuple(current_graph_config.get("feature_columns", ())) != FEATURE_COLUMNS:
            raise ValueError(f"Graph-feature order mismatch for {pipeline}.")
        if float(current_graph_config.get("threshold", -1)) != float(
            graph_config["threshold"]
        ):
            raise ValueError(f"Graph threshold mismatch for {pipeline}.")
        if current_graph_config.get("full_cohort_stats", {}).get(
            "topology_sha256"
        ) != graph_config.get("full_cohort_stats", {}).get("topology_sha256"):
            raise ValueError(f"Internal graph topology mismatch for {pipeline}.")

        external_all = np.load(representation_path(args, pipeline), mmap_mode="r")
        if external_all.ndim != 2 or external_all.shape[0] != cohort.audit["n_metadata_rows"]:
            raise ValueError(
                f"{pipeline} representation shape {external_all.shape} is incompatible "
                "with the full metadata row count."
            )
        external_input = np.asarray(
            external_all[cohort.embedding_rows], dtype=np.float32
        )
        training_input = np.load(directory / "embeddings_org.npy").astype(np.float32)
        if len(training_input) != len(training_graph_features):
            raise ValueError(f"Training feature/embedding row mismatch for {pipeline}.")
        if external_input.shape[1] != training_input.shape[1]:
            raise ValueError(f"Embedding dimension mismatch for {pipeline}.")
        if not np.isfinite(external_input).all():
            raise ValueError(f"External embeddings contain non-finite values: {pipeline}.")

        combined_input = np.concatenate([training_input, external_input])
        model = load_trained_model(directory / "model.pth", device)
        inferred, probabilities = infer_gnn_outputs(
            model, graph, combined_input, device
        )
        n_training = len(training_input)
        saved_training = np.load(directory / "embeddings_gnn.npy")
        max_training_difference = float(
            np.max(np.abs(inferred[:n_training] - saved_training))
        )
        if max_training_difference > 1e-5:
            raise ValueError(
                f"Frozen inference changed saved training embeddings for {pipeline}: "
                f"{max_training_difference}"
            )

        external_gnn = normalized(inferred[n_training:])
        output_path = (
            args.output_dir / "representations" / pipeline / "episode_gnn.emb.npy"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, external_gnn)
        assignments[f"{pipeline}_cluster"] = probabilities[n_training:].argmax(axis=1)
        assignments[f"{pipeline}_confidence"] = probabilities[n_training:].max(axis=1)
        model_checks[pipeline] = {
            "model_sha256": sha256_file(directory / "model.pth"),
            "max_abs_recomputed_vs_saved_training_embedding": (
                max_training_difference
            ),
        }

        for name, values in (
            (f"input__{pipeline}", external_input),
            (f"gnn__{pipeline}", external_gnn),
        ):
            summary, per_query, neighbor = evaluate_representation(
                name,
                values,
                cohort.manifest,
                endpoints=args.endpoints,
                k_values=k_values,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            summaries.append(summary)
            per_queries.append(per_query)
            neighbors.append(neighbor)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = write_evaluation_outputs(
        args.output_dir,
        summaries,
        per_queries,
        neighbors,
        assignments,
        pipelines=args.pipelines,
        endpoints=args.endpoints,
        k_values=k_values,
    )
    report = {
        "status": "complete",
        "analysis_role": "frozen external stage+radiology GNN evaluation",
        "n_pipelines": len(args.pipelines),
        "n_representations_evaluated": int(summary["method"].nunique()),
        "graph_feature_preset": "stage_radiology",
        "graph_inference": (
            "Frozen internally trained GraphSAGE-DMoN model; training-source "
            "edges retained, external-source edges removed."
        ),
        "graph_stats": graph_stats,
        "training_topology_check": topology_check,
        "endpoint_warning": (
            "An endpoint also used to construct graph edges is graph-informed and "
            "must not be interpreted as label-blind external performance."
        ),
        "training_graph_features_sha256": sha256_file(
            args.training_graph_features_path
        ),
        "model_checks": model_checks,
    }
    atomic_json(args.output_dir / "run_config.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
