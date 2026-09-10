#!/usr/bin/env python3
"""Evaluate retrieval on the 100 patients reserved by train_test_split.py.

The trained patients are the retrieval gallery and reserved test patients are
queries. Test patients are never added to the gallery. The script evaluates:

* ``GNN Embeddings``: test embeddings inferred by the trained one-layer
  GraphSAGE encoder. Test nodes receive messages only from training nodes.
* ``Original <source> Embeddings``: the input patient embeddings, as a
  source-specific baseline (for example, FILM or MedEmbed).

For every requested attribute, the report includes Precision@K
(retrieved-neighbor agreement), following ``VisualAppImon.py``. That app calls
the fraction of matching top-5 neighbors "R@5", but the implemented quantity
is Precision@5. Here the same calculation is applied at K=1, 5, and 10 to
Stage, Morphology, Laterality, HER2, Recurrence, and pCR by default.

Two overall rows are reported: an equal-weight mean of the available
attribute-level precisions, and VisualApp's patient-wise mean across valid
attributes followed by a mean across test patients. Attributes whose test
values are entirely missing remain visible with zero valid queries and NaN
precision.

``GNN_Phenotype`` is a derived evaluation attribute: both gallery and test
phenotypes are assigned by the trained DMoN head. Its Precision@K therefore
measures phenotype-neighborhood consistency, not agreement with an independent
ground-truth phenotype label.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("DGLBACKEND", "pytorch")

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# These names must exist in this script's __main__ namespace because model.pth
# may have been written by executing train_gnn.py directly and reference
# __main__.DMoNModel and __main__.DMoN when unpickled.
from .train import DMoN, DMoNModel, compute_cluster_quality  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = (
    PROJECT_ROOT
    / "zhemin"
    / "experiments"
    / "04_phenotype_gnn"
    / "direct_transformer"
    / "dmon_3"
)
DEFAULT_ATTRIBUTES = [
    "Stage_Label",
    "Morphology",
    "Laterality",
    "HER2_Status",
    "Recurrence",
    "pCR_Label",
]
# ``none`` is a valid pCR outcome (no complete response), not missing data.
UNKNOWN_VALUES = {"", "unknown", "nan", "9 (unknown)"}
DERIVED_PHENOTYPE_ATTRIBUTE = "GNN_Phenotype"


def is_missing(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in UNKNOWN_VALUES


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def embedding_source_name(
    run_config: dict[str, Any], override: str | None = None
) -> str:
    if override:
        return override
    source = (
        f"{run_config.get('new_emb_dir', '')}/"
        f"{run_config.get('new_emb_file', '')}"
    ).lower()
    if "medembed" in source:
        return "MedEmbed"
    if "film_supcon" in source or "patient_filmed" in source:
        return "FILM"
    return Path(run_config.get("new_emb_file", "source")).stem


def normalize_patient_ids(series: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any() or np.any(
        numeric.to_numpy(dtype=np.float64) % 1 != 0
    ):
        raise ValueError(f"{column_name} must contain integer patient IDs.")
    return numeric.astype("int64")


def prepare_evaluation_data(
    model_dir: Path,
    run_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Load and strictly align the graph, split, features, and clinical rows."""
    base_model_dir = Path(run_config["base_model_dir"])
    base_clinical_path = base_model_dir / run_config["base_clinical_file"]
    split_path = Path(run_config["split_manifest"])
    new_emb_dir = Path(run_config["new_emb_dir"])
    new_emb_path = new_emb_dir / run_config["new_emb_file"]
    new_meta_path = new_emb_dir / run_config["new_emb_meta_file"]
    new_patient_id_col = (
        run_config.get("new_patient_id_col") or args.new_patient_id_col
    )
    new_label_col = run_config.get("new_label_col") or args.new_label_col
    graph_construction = run_config.get("graph_construction")
    if graph_construction:
        topology_graph_path = model_dir / graph_construction[
            "full_cohort_graph_file"
        ]
        topology_patient_path = model_dir / graph_construction[
            "full_cohort_patient_file"
        ]
    else:
        topology_graph_path = base_model_dir / "graph.bin"
        topology_patient_path = base_clinical_path

    required_paths = [
        base_clinical_path,
        topology_graph_path,
        topology_patient_path,
        split_path,
        new_emb_path,
        new_meta_path,
        model_dir / "model.pth",
        model_dir / "patient_data_expanded_v2.csv",
        model_dir / "embeddings_org.npy",
        model_dir / "embeddings_gnn.npy",
    ]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing required evaluation files: {missing_paths}")

    graphs, _ = dgl.load_graphs(str(topology_graph_path))
    topology_graph = graphs[0]
    base_clinical = pd.read_csv(base_clinical_path)
    topology_patients = (
        pd.read_csv(topology_patient_path)
        if topology_patient_path != base_clinical_path
        else base_clinical
    )
    split_df = pd.read_csv(split_path)
    new_meta = pd.read_csv(new_meta_path)
    new_embeddings = np.load(new_emb_path, mmap_mode="r")
    train_clinical = pd.read_csv(model_dir / "patient_data_expanded_v2.csv")
    train_original = np.load(model_dir / "embeddings_org.npy")
    train_gnn = np.load(model_dir / "embeddings_gnn.npy")

    patient_id_col = run_config.get("split_patient_id_col", args.patient_id_col)
    split_col = run_config.get("split_col", "split")
    for frame_name, frame, column in [
        ("base clinical", base_clinical, args.patient_id_col),
        ("split manifest", split_df, patient_id_col),
        ("trained clinical", train_clinical, args.patient_id_col),
        ("new embedding metadata", new_meta, new_patient_id_col),
        ("topology patient map", topology_patients, args.patient_id_col),
    ]:
        if column not in frame.columns:
            raise KeyError(f"{frame_name} is missing patient ID column {column!r}.")
        frame[column] = normalize_patient_ids(frame[column], column)

    if split_col not in split_df.columns:
        raise KeyError(f"Split manifest is missing split column {split_col!r}.")
    split_df[split_col] = split_df[split_col].astype(str).str.strip().str.lower()
    if set(split_df[split_col]) != {"train", "test"}:
        raise ValueError("Split manifest must contain exactly train and test rows.")
    if split_df[patient_id_col].duplicated().any():
        raise ValueError("Split manifest contains duplicate patient IDs.")
    if base_clinical[args.patient_id_col].duplicated().any():
        raise ValueError("Baseline clinical data contains duplicate patient IDs.")
    if topology_patients[args.patient_id_col].duplicated().any():
        raise ValueError("Topology patient map contains duplicate patient IDs.")
    if len(topology_patients) != topology_graph.num_nodes():
        raise ValueError(
            "Topology patient-map rows do not match the graph node count."
        )
    if train_clinical[args.patient_id_col].duplicated().any():
        raise ValueError("Trained clinical data contains duplicate patient IDs.")
    if len(new_meta) != len(new_embeddings):
        raise ValueError(
            f"Embedding metadata has {len(new_meta)} rows but embeddings have "
            f"{len(new_embeddings)} rows."
        )

    new_meta = new_meta.copy()
    new_meta["_embedding_row_idx"] = np.arange(len(new_meta), dtype=np.int64)
    cohort_filter_applied = bool(
        run_config.get("cohort_filter_applied", True)
    )
    if cohort_filter_applied:
        if new_label_col not in new_meta.columns:
            raise KeyError(
                f"Embedding metadata is missing cohort-label column "
                f"{new_label_col!r}."
            )
        cohort_meta = new_meta.loc[
            new_meta[new_label_col].astype(str)
            == str(run_config["cancer_label"])
        ].drop_duplicates(new_patient_id_col, keep="last")
    else:
        cohort_meta = new_meta.drop_duplicates(
            new_patient_id_col, keep="last"
        )
    embedding_row_by_id = dict(
        zip(cohort_meta[new_patient_id_col], cohort_meta["_embedding_row_idx"])
    )

    train_ids = split_df.loc[
        split_df[split_col] == "train", patient_id_col
    ].tolist()
    test_ids_set = set(
        split_df.loc[split_df[split_col] == "test", patient_id_col].tolist()
    )
    saved_train_ids = train_clinical[args.patient_id_col].tolist()
    if set(saved_train_ids) != set(train_ids):
        raise ValueError(
            "Saved training artifacts do not contain exactly the manifest train patients."
        )
    if set(saved_train_ids).intersection(test_ids_set):
        raise RuntimeError("Test leakage: a reserved patient is in the retrieval gallery.")

    # Use saved training order so gallery clinical rows and saved embeddings
    # remain exactly aligned. Test order follows the saved topology row order.
    topology_row_by_id = pd.Series(
        np.arange(len(topology_patients), dtype=np.int64),
        index=topology_patients[args.patient_id_col],
    ).to_dict()
    missing_topology_ids = set(saved_train_ids).union(test_ids_set).difference(
        topology_row_by_id
    )
    if missing_topology_ids:
        raise ValueError(
            f"{len(missing_topology_ids)} split patients are absent from the "
            "saved topology patient map."
        )
    test_ids = sorted(test_ids_set, key=topology_row_by_id.__getitem__)
    ordered_ids = saved_train_ids + test_ids

    missing_embedding_ids = set(ordered_ids).difference(embedding_row_by_id)
    if missing_embedding_ids:
        raise ValueError(
            f"{len(missing_embedding_ids)} split patients have no new embedding row."
        )

    embedding_rows = np.asarray(
        [embedding_row_by_id[patient_id] for patient_id in ordered_ids],
        dtype=np.int64,
    )
    ordered_original = np.asarray(new_embeddings[embedding_rows], dtype=np.float32)
    if train_original.shape != ordered_original[: len(saved_train_ids)].shape:
        raise ValueError("Saved original training embeddings have an unexpected shape.")
    max_original_difference = float(
        np.max(np.abs(train_original - ordered_original[: len(saved_train_ids)]))
    )
    if max_original_difference > args.alignment_tolerance:
        raise ValueError(
            "Saved training features do not align with the source embeddings; "
            f"maximum absolute difference is {max_original_difference:.6g}."
        )
    if len(train_gnn) != len(saved_train_ids):
        raise ValueError("Saved GNN embeddings are not aligned with training patients.")
    if (
        DERIVED_PHENOTYPE_ATTRIBUTE in args.attributes
        and "Cluster_ID" not in train_clinical.columns
    ):
        raise KeyError(
            "Saved training metadata is missing the newly learned Cluster_ID."
        )

    clinical_by_id = base_clinical.set_index(args.patient_id_col, drop=False)
    ordered_clinical = clinical_by_id.loc[ordered_ids].reset_index(drop=True).copy()
    if "pathologic_complete_response.status" in ordered_clinical.columns:
        ordered_clinical["pCR_Label"] = ordered_clinical[
            "pathologic_complete_response.status"
        ].fillna("Unknown")
    elif "pCR_Label" not in ordered_clinical.columns:
        ordered_clinical["pCR_Label"] = "Unknown"

    if "Baseline_Phenotype" in args.attributes:
        if "phenotype" not in split_df.columns:
            raise KeyError("Split manifest is missing the balanced phenotype column.")
        phenotype_by_id = split_df.set_index(patient_id_col)["phenotype"]
        ordered_clinical["Baseline_Phenotype"] = (
            ordered_clinical[args.patient_id_col].map(phenotype_by_id)
        )

    for attribute in args.attributes:
        if attribute == DERIVED_PHENOTYPE_ATTRIBUTE:
            continue
        if attribute not in ordered_clinical.columns:
            raise KeyError(f"Evaluation attribute {attribute!r} is unavailable.")

    return {
        "base_graph": topology_graph,
        "base_clinical": topology_patients,
        "ordered_ids": ordered_ids,
        "train_ids": saved_train_ids,
        "test_ids": test_ids,
        "ordered_original": ordered_original,
        "ordered_clinical": ordered_clinical,
        "train_original": np.asarray(train_original, dtype=np.float32),
        "train_gnn": np.asarray(train_gnn, dtype=np.float32),
        "saved_train_phenotypes": (
            pd.to_numeric(train_clinical["Cluster_ID"], errors="raise").to_numpy(
                dtype=np.int64
            )
            if "Cluster_ID" in train_clinical.columns
            else None
        ),
        "max_original_difference": max_original_difference,
    }


def build_inference_graph(
    base_graph: dgl.DGLGraph,
    base_clinical: pd.DataFrame,
    ordered_ids: list[int],
    n_train: int,
    patient_id_col: str,
) -> tuple[dgl.DGLGraph, dict[str, int]]:
    """Keep train→train and train→test edges; remove every test-source edge."""
    local_row_by_id = {patient_id: row for row, patient_id in enumerate(ordered_ids)}
    base_to_local = np.full(base_graph.num_nodes(), -1, dtype=np.int64)
    for base_row, patient_id in enumerate(base_clinical[patient_id_col]):
        local_row = local_row_by_id.get(int(patient_id))
        if local_row is not None:
            base_to_local[base_row] = local_row

    source, destination = base_graph.edges()
    source_local = base_to_local[source.cpu().numpy()]
    destination_local = base_to_local[destination.cpu().numpy()]
    in_overlap = (source_local >= 0) & (destination_local >= 0)
    source_is_train = source_local < n_train
    keep = in_overlap & source_is_train

    kept_source = source_local[keep]
    kept_destination = destination_local[keep]
    graph = dgl.graph(
        (
            torch.from_numpy(kept_source),
            torch.from_numpy(kept_destination),
        ),
        num_nodes=len(ordered_ids),
    )
    if "weights" in base_graph.edata:
        kept_edge_rows = torch.from_numpy(
            np.flatnonzero(keep).astype(np.int64, copy=False)
        )
        graph.edata["weights"] = base_graph.edata["weights"][
            kept_edge_rows
        ].clone()
    train_edge_count = int(np.sum(keep & (destination_local < n_train)))
    test_incoming_edge_count = int(np.sum(keep & (destination_local >= n_train)))
    stats = {
        "n_inference_nodes": int(len(ordered_ids)),
        "n_train_nodes": int(n_train),
        "n_test_nodes": int(len(ordered_ids) - n_train),
        "n_train_to_train_edges": train_edge_count,
        "n_train_to_test_edges": test_incoming_edge_count,
        "n_test_source_edges": 0,
        "edge_weights_preserved": int("weights" in graph.edata),
    }
    return graph, stats


def load_trained_model(path: Path, device: torch.device) -> DMoNModel:
    """Load the full-object checkpoint written by train_gnn.py."""
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    if not isinstance(model, DMoNModel):
        raise TypeError(f"Unexpected model type in {path}: {type(model)}")
    return model.to(device).eval()


def infer_gnn_outputs(
    model: DMoNModel,
    graph: dgl.DGLGraph,
    features: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    graph = graph.to(device)
    features_t = torch.from_numpy(features).to(device)
    graph.ndata["feat"] = features_t
    with torch.no_grad():
        embeddings = F.relu(model.encoder(graph, graph.ndata["feat"]))
        assignments = F.softmax(model.dmon.assignment_mat(embeddings), dim=1)
    return (
        embeddings.cpu().numpy().astype(np.float32),
        assignments.cpu().numpy().astype(np.float32),
    )


def cosine_matrix(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    query_norm = queries / np.clip(
        np.linalg.norm(queries, axis=1, keepdims=True), 1e-12, None
    )
    gallery_norm = gallery / np.clip(
        np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12, None
    )
    return query_norm @ gallery_norm.T


def rank_neighbors(
    queries: np.ndarray,
    gallery: np.ndarray,
    query_ids: list[int],
    gallery_ids: list[int],
    feature_source: str,
    max_k: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    # Match VisualAppImon.py: cosine similarity followed by per-query min-max
    # normalization. The normalization is monotonic and therefore does not
    # ordinarily change ranks, but reproducing it makes the implementations
    # directly comparable. A stable full sort also matches pandas.nlargest's
    # first-occurrence behavior for exact ties.
    cosine_similarities = cosine_matrix(queries, gallery)
    similarity_min = cosine_similarities.min(axis=1, keepdims=True)
    similarity_span = (
        cosine_similarities.max(axis=1, keepdims=True) - similarity_min
    )
    visualapp_similarities = np.divide(
        cosine_similarities - similarity_min,
        similarity_span,
        out=np.zeros_like(cosine_similarities),
        where=similarity_span > 0,
    )
    top_indices = np.argsort(
        -visualapp_similarities,
        axis=1,
        kind="stable",
    )[:, :max_k]

    records = []
    for query_row, query_id in enumerate(query_ids):
        for rank, gallery_row in enumerate(top_indices[query_row], start=1):
            records.append(
                {
                    "feature_source": feature_source,
                    "query_patient_id": int(query_id),
                    "rank": rank,
                    "neighbor_patient_id": int(gallery_ids[gallery_row]),
                    "cosine_similarity": float(
                        cosine_similarities[query_row, gallery_row]
                    ),
                    "visualapp_similarity": float(
                        visualapp_similarities[query_row, gallery_row]
                    ),
                }
            )
    return top_indices, pd.DataFrame(records)


def evaluate_rankings(
    top_indices: np.ndarray,
    train_clinical: pd.DataFrame,
    test_clinical: pd.DataFrame,
    feature_source: str,
    patient_id_col: str,
    attributes: list[str],
    k_values: list[int],
) -> pd.DataFrame:
    """Return one Precision@K record per query and clinical attribute."""
    records = []
    for query_row, query in test_clinical.iterrows():
        query_id = int(query[patient_id_col])
        for attribute in attributes:
            target_value = query[attribute]
            if is_missing(target_value):
                continue

            gallery_values = train_clinical[attribute]
            relevant_mask = (gallery_values == target_value).to_numpy()
            for k in k_values:
                retrieved_rows = top_indices[query_row, :k]
                matches = int(relevant_mask[retrieved_rows].sum())
                records.append(
                    {
                        "feature_source": feature_source,
                        "query_patient_id": query_id,
                        "attribute": attribute,
                        "target_value": str(target_value),
                        "k": int(k),
                        "matches": matches,
                        "precision_at_k": matches / k,
                    }
                )
    return pd.DataFrame(records)


def summarize_metrics(
    per_query: pd.DataFrame,
    *,
    feature_sources: list[str],
    attributes: list[str],
    k_values: list[int],
) -> pd.DataFrame:
    if per_query.empty:
        return pd.DataFrame()
    observed = (
        per_query.groupby(["feature_source", "attribute", "k"], sort=True)
        .agg(
            n_queries=("query_patient_id", "nunique"),
            mean_precision_at_k=("precision_at_k", "mean"),
            std_precision_at_k=("precision_at_k", "std"),
        )
        .reset_index()
    )
    # Keep unavailable attributes visible instead of silently omitting them.
    # This is important for the current cohort, whose test Laterality values
    # are all missing/unknown.
    requested_grid = pd.MultiIndex.from_product(
        [feature_sources, attributes, k_values],
        names=["feature_source", "attribute", "k"],
    ).to_frame(index=False)
    summary = requested_grid.merge(
        observed,
        on=["feature_source", "attribute", "k"],
        how="left",
        validate="one_to_one",
    )
    summary["n_queries"] = summary["n_queries"].fillna(0).astype(int)
    summary["aggregation"] = "per-attribute query mean"
    summary["n_attributes_averaged"] = 1.0

    # User-requested overall: first obtain each attribute's test-query mean,
    # then give every attribute with at least one valid query equal weight.
    # In this dataset that is five attributes because Laterality has no valid
    # test values.
    available = summary.loc[summary["n_queries"] > 0]
    attribute_macro = (
        available.groupby(["feature_source", "k"], sort=True)
        .agg(
            n_queries=("n_queries", "max"),
            mean_precision_at_k=("mean_precision_at_k", "mean"),
            std_precision_at_k=("mean_precision_at_k", "std"),
            n_attributes_averaged=("attribute", "nunique"),
        )
        .reset_index()
    )
    attribute_macro.insert(
        1, "attribute", "OVERALL (available-attribute macro)"
    )
    attribute_macro["aggregation"] = (
        "equal mean of available attribute-level precisions"
    )

    # VisualAppImon first averages all valid attribute scores for the selected
    # patient. Reproduce that patient-level macro averaging before averaging
    # across the 100 reserved queries, so every query has equal weight even
    # when different attributes are missing.
    patient_average = (
        per_query.groupby(
            ["feature_source", "query_patient_id", "k"], sort=True
        )
        .agg(
            precision_at_k=("precision_at_k", "mean"),
            n_valid_attributes=("attribute", "nunique"),
        )
        .reset_index()
    )
    overall = (
        patient_average.groupby(["feature_source", "k"], sort=True)
        .agg(
            n_queries=("query_patient_id", "nunique"),
            mean_precision_at_k=("precision_at_k", "mean"),
            std_precision_at_k=("precision_at_k", "std"),
            n_attributes_averaged=("n_valid_attributes", "mean"),
        )
        .reset_index()
    )
    overall.insert(1, "attribute", "OVERALL (VisualApp patient macro)")
    overall["aggregation"] = (
        "per-query valid-attribute mean, then query mean"
    )
    return pd.concat(
        [summary, attribute_macro, overall],
        ignore_index=True,
    )


def summarize_metrics_by_attribute_value(
    per_query: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize retrieval separately for each phenotype/attribute value."""
    if per_query.empty:
        return pd.DataFrame()
    return (
        per_query.groupby(
            ["feature_source", "attribute", "target_value", "k"],
            sort=True,
        )
        .agg(
            n_queries=("query_patient_id", "nunique"),
            mean_precision_at_k=("precision_at_k", "mean"),
            std_precision_at_k=("precision_at_k", "std"),
        )
        .reset_index()
    )


def patient_average_metrics(per_query: pd.DataFrame) -> pd.DataFrame:
    """Return VisualApp-style mean Precision@K across valid attributes/query."""
    if per_query.empty:
        return pd.DataFrame()
    return (
        per_query.groupby(
            ["feature_source", "query_patient_id", "k"], sort=True
        )
        .agg(
            n_valid_attributes=("attribute", "nunique"),
            mean_precision_at_k=("precision_at_k", "mean"),
        )
        .reset_index()
    )


def summarize_cluster_composition(
    clinical: pd.DataFrame,
    *,
    attributes: list[str],
    n_train: int,
) -> pd.DataFrame:
    """Describe DMoN clusters using notebook-style clinical composition."""
    frames = (
        ("train", clinical.iloc[:n_train]),
        ("test", clinical.iloc[n_train:]),
        ("all", clinical),
    )
    records: list[dict[str, object]] = []
    composition_attributes = [
        attribute
        for attribute in attributes
        if attribute != DERIVED_PHENOTYPE_ATTRIBUTE
    ]
    for cohort_name, cohort in frames:
        for cluster_id, cluster in cohort.groupby(
            DERIVED_PHENOTYPE_ATTRIBUTE, sort=True
        ):
            cluster_size = len(cluster)
            for attribute in composition_attributes:
                values = cluster[attribute].map(
                    lambda value: "<missing>" if is_missing(value) else str(value)
                )
                for target_value, count in values.value_counts(
                    dropna=False, sort=False
                ).items():
                    records.append(
                        {
                            "cohort": cohort_name,
                            "cluster_id": int(cluster_id),
                            "cluster_size": cluster_size,
                            "attribute": attribute,
                            "target_value": target_value,
                            "n_patients": int(count),
                            "fraction_within_cluster": count / cluster_size,
                        }
                    )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--source-name",
        default=None,
        help="Human-readable embedding method name used in output labels.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <model-dir>/eval_reserved100.",
    )
    parser.add_argument("--patient-id-col", default="PATIENT_CLINIC_NUMBER")
    parser.add_argument("--new-patient-id-col", default="patient_id_raw")
    parser.add_argument("--new-label-col", default="cancer_label")
    parser.add_argument(
        "--attributes", nargs="+", default=DEFAULT_ATTRIBUTES
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="One or more retrieval cutoffs, for example --k 1 5 10.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--alignment-tolerance",
        type=float,
        default=1e-4,
        help="Maximum absolute difference allowed in embedding alignment checks.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing evaluation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.k = sorted(set(args.k))
    if not args.k or min(args.k) < 1:
        raise ValueError("Every --k value must be at least 1.")

    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else model_dir / "eval_reserved100"
    )
    run_config = load_json(model_dir / "run_config.json")
    data = prepare_evaluation_data(model_dir, run_config, args)

    n_train = len(data["train_ids"])
    n_test = len(data["test_ids"])
    if max(args.k) > n_train:
        raise ValueError(f"Maximum K={max(args.k)} exceeds gallery size {n_train}.")
    expected_test = int(run_config.get("n_test_patients_excluded", n_test))
    if n_test != expected_test:
        raise ValueError(
            f"Found {n_test} test patients but training recorded {expected_test}."
        )

    inference_graph, graph_stats = build_inference_graph(
        base_graph=data["base_graph"],
        base_clinical=data["base_clinical"],
        ordered_ids=data["ordered_ids"],
        n_train=n_train,
        patient_id_col=args.patient_id_col,
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )
    model = load_trained_model(model_dir / "model.pth", device)
    inferred_all, phenotype_probabilities = infer_gnn_outputs(
        model, inference_graph, data["ordered_original"], device
    )
    inferred_train = inferred_all[:n_train]
    inferred_test = inferred_all[n_train:]
    phenotype_ids = np.argmax(phenotype_probabilities, axis=1).astype(np.int64)

    if inferred_train.shape != data["train_gnn"].shape:
        raise ValueError("Recomputed training GNN embeddings have an unexpected shape.")
    max_train_gnn_difference = float(
        np.max(np.abs(inferred_train - data["train_gnn"]))
    )
    if max_train_gnn_difference > args.alignment_tolerance:
        raise ValueError(
            "Recomputed training embeddings do not match saved embeddings; "
            f"maximum absolute difference is {max_train_gnn_difference:.6g}."
        )

    saved_train_phenotypes = data["saved_train_phenotypes"]
    if saved_train_phenotypes is None:
        raise KeyError("Saved training metadata has no learned phenotype IDs.")
    train_phenotype_mismatches = int(
        np.sum(phenotype_ids[:n_train] != saved_train_phenotypes)
    )
    if train_phenotype_mismatches:
        raise ValueError(
            f"Recomputed DMoN assignments disagree with {train_phenotype_mismatches} "
            "saved training phenotype IDs."
        )

    ordered_clinical = data["ordered_clinical"].copy()
    ordered_clinical[DERIVED_PHENOTYPE_ATTRIBUTE] = phenotype_ids
    ordered_clinical["GNN_Phenotype_Confidence"] = phenotype_probabilities.max(
        axis=1
    )
    probability_columns = []
    for cluster_idx in range(phenotype_probabilities.shape[1]):
        column = f"GNN_Phenotype_Probability_{cluster_idx}"
        ordered_clinical[column] = phenotype_probabilities[:, cluster_idx]
        probability_columns.append(column)

    train_clinical = ordered_clinical.iloc[:n_train].reset_index(drop=True)
    test_clinical = ordered_clinical.iloc[n_train:].reset_index(drop=True)
    attribute_valid_query_counts = {
        attribute: int(
            (~test_clinical[attribute].map(is_missing)).sum()
        )
        for attribute in args.attributes
    }
    source_name = embedding_source_name(run_config, args.source_name)
    sources = [
        (
            f"GNN on {source_name}",
            inferred_test,
            data["train_gnn"],
        ),
        (
            f"Original {source_name}",
            data["ordered_original"][n_train:],
            data["train_original"],
        ),
    ]

    neighbor_frames = []
    metric_frames = []
    for label, query_embeddings, gallery_embeddings in sources:
        top_indices, neighbors = rank_neighbors(
            queries=query_embeddings,
            gallery=gallery_embeddings,
            query_ids=data["test_ids"],
            gallery_ids=data["train_ids"],
            feature_source=label,
            max_k=max(args.k),
        )
        neighbor_frames.append(neighbors)
        metric_frames.append(
            evaluate_rankings(
                top_indices=top_indices,
                train_clinical=train_clinical,
                test_clinical=test_clinical,
                feature_source=label,
                patient_id_col=args.patient_id_col,
                attributes=args.attributes,
                k_values=args.k,
            )
        )

    neighbors_df = pd.concat(neighbor_frames, ignore_index=True)
    per_query_df = pd.concat(metric_frames, ignore_index=True)
    per_query_average_df = patient_average_metrics(per_query_df)
    summary_df = summarize_metrics(
        per_query_df,
        feature_sources=[source[0] for source in sources],
        attributes=args.attributes,
        k_values=args.k,
    )
    by_attribute_value_df = summarize_metrics_by_attribute_value(per_query_df)
    cluster_quality = {
        "interpretation": (
            "Notebook-style internal cluster quality using DMoN argmax labels; "
            "these are descriptive metrics, not held-out label accuracy."
        ),
        "train": {
            "input_embeddings": run_config.get(
                "cluster_quality_input_embeddings"
            ),
            "gnn_embeddings": run_config.get(
                "cluster_quality_gnn_embeddings"
            ),
        },
        "test": {
            "input_embeddings": compute_cluster_quality(
                data["ordered_original"][n_train:], phenotype_ids[n_train:]
            ),
            "gnn_embeddings": compute_cluster_quality(
                inferred_test, phenotype_ids[n_train:]
            ),
        },
        "all": {
            "input_embeddings": compute_cluster_quality(
                data["ordered_original"], phenotype_ids
            ),
            "gnn_embeddings": compute_cluster_quality(
                inferred_all, phenotype_ids
            ),
        },
    }
    cluster_composition_df = summarize_cluster_composition(
        ordered_clinical,
        attributes=args.attributes,
        n_train=n_train,
    )

    output_paths = {
        "summary": output_dir / "retrieval_summary.csv",
        "by_attribute_value": (
            output_dir / "retrieval_by_attribute_value.csv"
        ),
        "per_query": output_dir / "retrieval_per_query.csv",
        "per_query_average": output_dir / "retrieval_per_query_average.csv",
        "neighbors": output_dir / "retrieval_neighbors.csv",
        "test_gnn_embeddings": output_dir / "test_embeddings_gnn.npy",
        "test_original_embeddings": output_dir / "test_embeddings_org.npy",
        "test_metadata": output_dir / "test_patient_metadata.csv",
        "cluster_quality": output_dir / "cluster_quality.json",
        "cluster_composition": output_dir / "cluster_composition.csv",
        "report": output_dir / "eval_config.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Evaluation output already exists: {existing}. Pass --overwrite to replace it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_paths["summary"], index=False)
    by_attribute_value_df.to_csv(
        output_paths["by_attribute_value"], index=False
    )
    per_query_df.to_csv(output_paths["per_query"], index=False)
    per_query_average_df.to_csv(output_paths["per_query_average"], index=False)
    neighbors_df.to_csv(output_paths["neighbors"], index=False)
    np.save(output_paths["test_gnn_embeddings"], inferred_test)
    np.save(
        output_paths["test_original_embeddings"],
        data["ordered_original"][n_train:],
    )
    metadata_columns = [args.patient_id_col] + args.attributes
    if DERIVED_PHENOTYPE_ATTRIBUTE in args.attributes:
        metadata_columns += ["GNN_Phenotype_Confidence"] + probability_columns
    test_clinical[metadata_columns].to_csv(
        output_paths["test_metadata"], index=False
    )
    output_paths["cluster_quality"].write_text(
        json.dumps(cluster_quality, indent=2) + "\n", encoding="utf-8"
    )
    cluster_composition_df.to_csv(
        output_paths["cluster_composition"], index=False
    )

    train_phenotype_counts = {
        str(cluster_id): int(count)
        for cluster_id, count in zip(
            *np.unique(phenotype_ids[:n_train], return_counts=True)
        )
    }
    test_phenotype_counts = {
        str(cluster_id): int(count)
        for cluster_id, count in zip(
            *np.unique(phenotype_ids[n_train:], return_counts=True)
        )
    }

    report = {
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "split_manifest": run_config["split_manifest"],
        "n_gallery_train_patients": n_train,
        "n_reserved_test_queries": n_test,
        "test_patients_in_gallery": 0,
        "attributes": args.attributes,
        "attribute_valid_query_counts": attribute_valid_query_counts,
        "k_values": args.k,
        "feature_sources": [source[0] for source in sources],
        "device": str(device),
        "inference_policy": (
            "one-layer GraphSAGE on train-to-train and train-to-test edges only; "
            "all test-source edges, including test-to-test edges, are removed"
        ),
        "graph_stats": graph_stats,
        "alignment_checks": {
            "max_abs_saved_vs_source_train_features": data[
                "max_original_difference"
            ],
            "max_abs_recomputed_vs_saved_train_gnn_embeddings": (
                max_train_gnn_difference
            ),
            "recomputed_vs_saved_train_phenotype_mismatches": (
                train_phenotype_mismatches
            ),
        },
        "phenotype_assignment": {
            "attribute": DERIVED_PHENOTYPE_ATTRIBUTE,
            "method": "argmax of the trained DMoN soft assignment head",
            "n_clusters": int(phenotype_probabilities.shape[1]),
            "train_counts": train_phenotype_counts,
            "test_counts": test_phenotype_counts,
            "interpretation": (
                "Precision@K on GNN_Phenotype measures consistency with the "
                "model-derived phenotype partition, not supervised accuracy."
            ),
        },
        "notebook_style_cluster_quality": cluster_quality,
        "metric_definitions": {
            "ranking": (
                "top-K by cosine similarity after per-query min-max "
                "normalization, with stable first-occurrence tie handling, "
                "matching VisualAppImon.py"
            ),
            "precision_at_k": (
                "matching retrieved gallery patients / K, identical to the "
                "quantity labeled R@5 in VisualAppImon.py"
            ),
            "overall": (
                "for each query, mean Precision@K across attributes whose "
                "query value is not missing/unknown; then mean across queries"
            ),
            "available_attribute_macro": (
                "equal-weight mean of the separately calculated attribute "
                "Precision@K values, excluding attributes with zero valid "
                "test-query values"
            ),
        },
        "known_limitations": [
            (
                "The graph topology is constructed from clinical/radiology "
                "similarity, including stage-related inputs; metrics for graph "
                "construction attributes are not label-blind."
            ),
            (
                "GNN_Phenotype is assigned by this same trained model, so its "
                "retrieval metric is phenotype-neighborhood consistency rather "
                "than validation against an external phenotype ground truth."
            ),
        ],
        "files": {name: path.name for name, path in output_paths.items()},
    }
    output_paths["report"].write_text(json.dumps(report, indent=2) + "\n")

    print(f"Gallery patients: {n_train}")
    print(f"Reserved test queries: {n_test}")
    print("Test patients in gallery: 0")
    print(f"Evaluation outputs: {output_dir}")
    display_columns = [
        "feature_source",
        "attribute",
        "k",
        "n_queries",
        "mean_precision_at_k",
    ]
    print(summary_df[display_columns].to_string(index=False))


if __name__ == "__main__":
    main()
