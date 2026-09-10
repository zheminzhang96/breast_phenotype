#!/usr/bin/env python3
"""Discover new DMoN phenotypes from transformer patient embeddings.

The model architecture follows ``GNN_exploration.ipynb``: a SAGEConv encoder
feeding a DMoN soft-clustering head, trained with modularity + collapse loss.
Its patient graph is rebuilt from the configured clinical/radiology indicators;
the saved graph in ``models/train-07`` is not loaded or reused.

Node features may be any of the aligned 768-dimensional Stage 02 or Stage 03
patient embeddings registered by ``run_matrix.py``. Legacy
``Cluster_ID`` values and phenotype labels are ignored completely. The desired
number of new clusters must be specified explicitly.

Only patients marked ``train`` are included in the induced training subgraph.
The 100 fixed test patients and all incident edges are explicitly excluded and
checked for leakage before training starts.

Example:
    /media/Datacenter_storage/ProstateCancer/envs/cancerGNN/bin/python
        -m zhemin.pipeline.stage_04_gnn.train --n-clusters 3 --device cuda:0

    /media/Datacenter_storage/ProstateCancer/envs/cancerGNN/bin/python
        -m zhemin.pipeline.stage_04_gnn.train --n-clusters 3 --validate-split-only

This writes the self-contained run to
``zhemin/experiments/04_phenotype_gnn/direct_transformer/dmon_<n-clusters>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("DGLBACKEND", "pytorch")

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_MODEL_DIR = PROJECT_ROOT / "models" / "train-07"
DEFAULT_NEW_EMB_DIR = (
    PROJECT_ROOT
    / "zhemin"
    / "experiments"
    / "02_patient_embeddings"
    / "transformer"
)
DEFAULT_NEW_EMB_FILE = "patient.emb.npy"
DEFAULT_NEW_EMB_META_FILE = "patient.meta.csv"
DEFAULT_SPLIT_MANIFEST = (
    PROJECT_ROOT
    / "zhemin"
    / "xiaoyang"
    / "splits"
    / "train_test_split_patients.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "zhemin"
    / "experiments"
    / "04_phenotype_gnn"
    / "direct_transformer"
)

STAGE_FEATURE_COLUMNS = (
    "Merged_Mixed_Stage_Broad_Other",
    "Merged_Mixed_Stage_Broad_Stage_0",
    "Merged_Mixed_Stage_Broad_Stage_1",
    "Merged_Mixed_Stage_Broad_Stage_2",
    "Merged_Mixed_Stage_Broad_Stage_3",
    "Merged_Mixed_Stage_Broad_Stage_4",
    "Merged_Mixed_Stage_Broad_Unknown",
)
KI67_FEATURE_COLUMNS = (
    "Ki67_Category_High",
    "Ki67_Category_Intermediate",
    "Ki67_Category_Low",
)
RADIOLOGY_FEATURE_COLUMNS = (
    "DENSITY_HETEROGENEOUSLY_DENSE",
    "DENSITY_EXTREMELY_DENSE",
    "DENSITY_FIBROGLANDULAR",
    "DENSITY_ENTIRELY_FAT",
    "MASS_GENERIC",
    "MASS_CIRCUMSCRIBED",
    "MASS_DENSITY",
    "MASS_FAT-CONTAINING",
    "MASS_INDISTINCT",
    "MASS_IRREGULAR",
    "MASS_MICROLOBULATED",
    "MASS_OBSCURED",
    "MASS_OVAL",
    "MASS_ROUND",
    "MASS_SPICULATED",
    "CALCIFICATION_BENIGN",
    "CALCIFICATION_DISTRIBUTION",
    "CALCIFICATION_SUSPICIOUS",
    "ASYMMETRY_ASYMMETRY",
    "ASYMMETRY_DEVELOPING",
    "ASYMMETRY_FOCAL",
    "ASYMMETRY_GLOBAL",
    "ASYMMETRY_QUESTIONED",
    "FEATURES_ARCHITECTURAL_DISTORTION",
    "FEATURES_INTRAMAMMARY_LYMPH_NODE",
    "FEATURES_AXILLARY_ADENOPATHY",
    "FEATURES_SKIN_THICKENING",
    "FEATURES_NIPPLE_RETRACTION",
)
EDGE_FEATURE_PRESETS = {
    "stage_ki67_radiology": (
        STAGE_FEATURE_COLUMNS + KI67_FEATURE_COLUMNS + RADIOLOGY_FEATURE_COLUMNS
    ),
    "stage_radiology": STAGE_FEATURE_COLUMNS + RADIOLOGY_FEATURE_COLUMNS,
    "radiology_only": RADIOLOGY_FEATURE_COLUMNS,
}
# Backward-compatible name for existing imports and the notebook-faithful default.
EDGE_FEATURE_COLUMNS = EDGE_FEATURE_PRESETS["stage_ki67_radiology"]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DMoN(nn.Module):
    """Deep Modularity Network soft-clustering head (unchanged from GNN_exploration.ipynb)."""

    def __init__(self, in_dim: int, n_clusters: int, collapse_reg: float):
        super().__init__()
        self.n_clusters = n_clusters
        self.collapse_reg = collapse_reg
        self.assignment_mat = nn.Linear(in_dim, n_clusters)
        nn.init.orthogonal_(self.assignment_mat.weight)

    def forward(
        self, g: dgl.DGLGraph, h: torch.Tensor, adj: torch.Tensor | None = None
    ):
        assignments = F.softmax(self.assignment_mat(h), dim=1)

        # This is exactly trace(S.T @ A @ S) / |E| from the original dense
        # implementation, evaluated directly over graph edges. Avoiding the
        # dense N x N adjacency makes the 11-source comparison practical on CPU.
        source, destination = g.edges()
        edge_agreement = (
            assignments[source] * assignments[destination]
        ).sum(dim=1)
        modularity_loss = -edge_agreement.sum() / g.num_edges()

        cluster_sizes = assignments.sum(dim=0)
        collapse_loss = torch.norm(cluster_sizes) * (self.n_clusters ** 0.5) / h.shape[0] - 1

        total_loss = modularity_loss + self.collapse_reg * collapse_loss
        return assignments, total_loss, modularity_loss, collapse_loss


class DMoNModel(nn.Module):
    """SAGEConv encoder + DMoN head (unchanged from GNN_exploration.ipynb)."""

    def __init__(self, in_dim: int, hidden_dim: int, n_clusters: int, collapse_reg: float = 0.1):
        super().__init__()
        self.encoder = dglnn.SAGEConv(in_dim, hidden_dim, "pool")
        self.dmon = DMoN(hidden_dim, n_clusters, collapse_reg)

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor, adj: torch.Tensor):
        h = F.relu(self.encoder(g, g.ndata["feat"]))
        assignments, dmon_loss, modularity_loss, collapse_loss = self.dmon(g, h, adj)
        return assignments, dmon_loss, h, modularity_loss, collapse_loss


def dense_binary_adjacency(g: dgl.DGLGraph) -> np.ndarray:
    u, v = g.edges()
    num_nodes = g.num_nodes()
    adj = torch.sparse_coo_tensor(torch.stack([u, v]), torch.ones_like(u), (num_nodes, num_nodes))
    adj_dense = adj.to_dense()
    adj_dense[adj_dense > 0] = 1
    return adj_dense.cpu().numpy()


def compute_cluster_quality(embeddings: np.ndarray, cluster_labels: np.ndarray) -> dict:
    n_observed_clusters = int(np.unique(cluster_labels).size)
    if n_observed_clusters < 2:
        return {
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "calinski_harabasz_index": None,
            "n_observed_clusters": n_observed_clusters,
            "quality_status": (
                "unavailable: cluster assignments collapsed to one observed cluster"
            ),
        }
    return {
        "silhouette_score": round(float(silhouette_score(embeddings, cluster_labels)), 4),
        "davies_bouldin_index": round(float(davies_bouldin_score(embeddings, cluster_labels)), 4),
        "calinski_harabasz_index": round(float(calinski_harabasz_score(embeddings, cluster_labels)), 2),
        "n_observed_clusters": n_observed_clusters,
        "quality_status": "available",
    }


def build_clinical_similarity_graph(
    clinical: pd.DataFrame,
    *,
    threshold: float,
    feature_columns: tuple[str, ...] = EDGE_FEATURE_COLUMNS,
) -> tuple[dgl.DGLGraph, dict[str, float | int | str]]:
    """Build the rounded-cosine graph used by GNN_exploration.ipynb."""
    if not 0 <= threshold <= 1:
        raise ValueError("edge similarity threshold must be between zero and one.")
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Edge feature columns must be non-empty and unique.")
    missing_columns = sorted(set(feature_columns).difference(clinical.columns))
    if missing_columns:
        raise KeyError(
            f"Clinical metadata is missing edge feature columns: {missing_columns}"
        )

    numeric = clinical.loc[:, feature_columns].apply(
        pd.to_numeric,
        errors="raise",
    )
    edge_features = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(edge_features).all():
        raise ValueError("Clinical edge features contain non-finite values.")
    if not np.isin(edge_features, (0.0, 1.0)).all():
        raise ValueError("Clinical edge features must be binary one-hot indicators.")
    if np.any(np.linalg.norm(edge_features, axis=1) == 0):
        raise ValueError("At least one patient has an all-zero edge feature vector.")

    similarities = np.round(
        cosine_similarity(edge_features),
        decimals=2,
    )
    source, destination = np.nonzero(similarities >= threshold)
    edge_weights = similarities[source, destination].astype(np.float32)
    graph = dgl.graph(
        (
            torch.from_numpy(source.astype(np.int64, copy=False)),
            torch.from_numpy(destination.astype(np.int64, copy=False)),
        ),
        num_nodes=len(clinical),
    )
    graph.edata["weights"] = torch.from_numpy(edge_weights[:, None])
    if graph.num_nodes() != len(clinical):
        raise RuntimeError("Constructed graph node count does not match clinical rows.")
    if graph.num_edges() == 0:
        raise ValueError("Clinical similarity threshold produced an empty graph.")
    n_self_loops = int((source == destination).sum())
    if n_self_loops != len(clinical):
        raise ValueError(
            "The notebook graph must contain exactly one self-loop per patient."
        )

    topology_hasher = hashlib.sha256()
    topology_hasher.update(edge_features.astype(np.uint8, copy=False).tobytes())
    topology_hasher.update(source.astype(np.int64, copy=False).tobytes())
    topology_hasher.update(destination.astype(np.int64, copy=False).tobytes())
    topology_hasher.update(edge_weights.tobytes())

    stats: dict[str, float | int | str] = {
        "n_nodes": int(graph.num_nodes()),
        "n_directed_edges": int(graph.num_edges()),
        "n_self_loops": n_self_loops,
        "threshold": float(threshold),
        "similarity_rounding_decimals": 2,
        "n_edge_features": len(feature_columns),
        "edge_weight_min": float(edge_weights.min()),
        "edge_weight_max": float(edge_weights.max()),
        "edge_weight_mean": float(edge_weights.mean()),
        "topology_sha256": topology_hasher.hexdigest(),
        "edge_density": float(
            graph.num_edges() / (graph.num_nodes() * graph.num_nodes())
        ),
    }
    return graph, stats


def load_new_embeddings(
    new_emb_dir: Path,
    emb_file: str,
    meta_file: str,
    patient_id_col: str,
    label_col: str | None = None,
    cancer_label: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load row-aligned embeddings, optionally filtering a multi-cancer table."""
    emb = np.load(new_emb_dir / emb_file, mmap_mode="r")
    meta = pd.read_csv(new_emb_dir / meta_file)
    if emb.ndim != 2:
        raise ValueError(f"{emb_file} must be a two-dimensional embedding array.")
    if len(emb) != len(meta):
        raise ValueError(
            f"Row mismatch: {emb_file} has {len(emb)} rows but {meta_file} has {len(meta)} rows."
        )
    if not np.isfinite(emb).all():
        raise ValueError(f"{emb_file} contains non-finite values.")
    if patient_id_col not in meta.columns:
        raise KeyError(
            f"Embedding metadata is missing patient ID column {patient_id_col!r}."
        )

    meta = meta.copy()
    meta["_row_idx"] = np.arange(len(meta))
    if "embedding_row_idx" in meta.columns:
        declared_rows = pd.to_numeric(
            meta["embedding_row_idx"], errors="raise"
        ).to_numpy(dtype=np.int64)
        expected_rows = np.arange(len(meta), dtype=np.int64)
        if not np.array_equal(declared_rows, expected_rows):
            raise ValueError(
                "Embedding metadata embedding_row_idx is not aligned with the "
                "NumPy embedding row order."
            )

    if label_col is not None:
        if label_col not in meta.columns:
            raise KeyError(
                f"Embedding metadata is missing cohort-label column {label_col!r}."
            )
        if cancer_label is None:
            raise ValueError("--cancer-label is required when --new-label-col is set.")
        cohort_values = meta[label_col].astype(str).str.strip().str.casefold()
        meta = meta.loc[cohort_values.eq(cancer_label.strip().casefold())].copy()
        if meta.empty:
            raise ValueError(
                f"No embedding rows matched {label_col}={cancer_label!r}."
            )

    numeric_ids = pd.to_numeric(meta[patient_id_col], errors="raise")
    if numeric_ids.isna().any() or np.any(
        numeric_ids.to_numpy(dtype=np.float64) % 1 != 0
    ):
        raise ValueError(
            f"Embedding metadata column {patient_id_col!r} must contain "
            "integer patient IDs."
        )
    meta[patient_id_col] = numeric_ids.astype("int64")
    n_before = len(meta)
    # Match the original GNN_exploration.ipynb MedEmbed alignment behavior.
    meta = meta.drop_duplicates(subset=[patient_id_col], keep="last")
    if len(meta) != n_before:
        print(f"Dropped {n_before - len(meta)} duplicate patient ids in new embedding metadata.")
    return emb, meta


def load_train_test_split(
    manifest_path: Path,
    patient_id_col: str,
    split_col: str,
) -> tuple[set[int], set[int], pd.DataFrame]:
    """Load a two-way patient split and return disjoint train/test ID sets."""
    split_df = pd.read_csv(
        manifest_path,
        usecols=[patient_id_col, split_col],
    )
    required = {patient_id_col, split_col}
    missing = sorted(required.difference(split_df.columns))
    if missing:
        raise KeyError(f"Split manifest is missing column(s): {missing}")

    split_df = split_df.copy()
    numeric_ids = pd.to_numeric(split_df[patient_id_col], errors="raise")
    if numeric_ids.isna().any() or np.any(
        numeric_ids.to_numpy(dtype=np.float64) % 1 != 0
    ):
        raise ValueError(
            f"Split manifest column {patient_id_col!r} must contain integer patient IDs."
        )
    split_df[patient_id_col] = numeric_ids.astype("int64")
    split_df[split_col] = split_df[split_col].astype(str).str.strip().str.lower()

    if split_df[patient_id_col].duplicated().any():
        duplicate_count = int(split_df[patient_id_col].duplicated().sum())
        raise ValueError(
            f"Split manifest contains {duplicate_count} duplicate patient IDs."
        )

    allowed_splits = {"train", "test"}
    observed_splits = set(split_df[split_col])
    unexpected = observed_splits.difference(allowed_splits)
    if unexpected:
        raise ValueError(
            f"Split manifest must contain only train/test rows; found {sorted(unexpected)}."
        )
    missing_splits = allowed_splits.difference(observed_splits)
    if missing_splits:
        raise ValueError(
            f"Split manifest is missing required split(s): {sorted(missing_splits)}."
        )

    train_ids = set(
        split_df.loc[split_df[split_col] == "train", patient_id_col].tolist()
    )
    test_ids = set(
        split_df.loc[split_df[split_col] == "test", patient_id_col].tolist()
    )
    leaked_ids = train_ids.intersection(test_ids)
    if leaked_ids:
        raise ValueError(
            f"Split leakage: {len(leaked_ids)} patient IDs occur in both train and test."
        )
    if not train_ids or not test_ids:
        raise ValueError("Both train and test patient sets must be non-empty.")

    return train_ids, test_ids, split_df


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    if args.n_clusters < 2:
        raise ValueError("--n-clusters must be at least 2.")

    base_model_dir = Path(args.base_model_dir).expanduser().resolve()
    new_emb_dir = Path(args.new_emb_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path(args.output_root).expanduser().resolve() / f"dmon_{args.n_clusters}"
    )
    split_manifest = Path(args.split_manifest).expanduser().resolve()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        (args.device if args.device != "auto" else "cpu")
    )

    edge_feature_columns = EDGE_FEATURE_PRESETS[args.edge_feature_set]
    # --- Load clinical indicators used to construct a new graph. ---
    base_clinical = pd.read_csv(base_model_dir / args.base_clinical_file)
    base_clinical[args.base_patient_id_col] = base_clinical[args.base_patient_id_col].astype("int64")
    if base_clinical[args.base_patient_id_col].duplicated().any():
        duplicate_count = int(
            base_clinical[args.base_patient_id_col].duplicated().sum()
        )
        raise ValueError(
            f"Baseline clinical data contains {duplicate_count} duplicate patient IDs."
        )
    # The legacy Cluster_ID is a result of the previous GNN and must never
    # become an input, reference target, or evaluation label for this run.
    base_clinical = base_clinical.drop(columns=["Cluster_ID"], errors="ignore")

    # --- Load the explicit two-way split. Only train IDs may enter the graph. ---
    train_ids, test_ids, split_df = load_train_test_split(
        manifest_path=split_manifest,
        patient_id_col=args.split_patient_id_col,
        split_col=args.split_col,
    )
    # --- Load new node embeddings for the requested cancer cohort ---
    new_emb, new_meta = load_new_embeddings(
        new_emb_dir=new_emb_dir,
        emb_file=args.new_emb_file,
        meta_file=args.new_emb_meta_file,
        patient_id_col=args.new_patient_id_col,
        label_col=args.new_label_col,
        cancer_label=args.cancer_label,
    )
    pid_to_row = dict(zip(new_meta[args.new_patient_id_col], new_meta["_row_idx"]))

    # --- Validate that every manifest patient is available in the current
    # baseline/new-embedding overlap. The embedding source may cover additional
    # patients; those are deliberately ignored so different embedding sources
    # use the exact same train/test population. ---
    base_ids = set(base_clinical[args.base_patient_id_col].tolist())
    embedding_ids = set(pid_to_row)
    source_overlap_ids = base_ids.intersection(embedding_ids)
    manifest_ids = train_ids.union(test_ids)

    missing_from_overlap = manifest_ids.difference(source_overlap_ids)
    if missing_from_overlap:
        raise ValueError(
            f"{len(missing_from_overlap)} split-manifest patients are unavailable "
            "in the current baseline/new-embedding overlap. Use an embedding source "
            "that covers the complete fixed split."
        )
    extra_source_patients_ignored = source_overlap_ids.difference(manifest_ids)

    # Build a fresh full-cohort graph from clinical/radiology indicators. The
    # row order is explicitly recorded for future train/test inference.
    cohort_clinical = base_clinical.loc[
        base_clinical[args.base_patient_id_col].isin(manifest_ids)
    ].reset_index(drop=True)
    if set(cohort_clinical[args.base_patient_id_col]) != manifest_ids:
        raise RuntimeError("Full-cohort clinical rows do not match the split manifest.")
    split_by_id = split_df.set_index(args.split_patient_id_col)[args.split_col]
    cohort_clinical[args.split_col] = cohort_clinical[
        args.base_patient_id_col
    ].map(split_by_id)
    if cohort_clinical[args.split_col].isna().any():
        raise RuntimeError("A cohort patient is missing its train/test assignment.")

    g_cohort, cohort_graph_stats = build_clinical_similarity_graph(
        cohort_clinical,
        threshold=args.edge_similarity_threshold,
        feature_columns=edge_feature_columns,
    )
    train_local_idx = np.flatnonzero(
        cohort_clinical[args.split_col].eq("train").to_numpy()
    )
    kept_ids = set(
        cohort_clinical.loc[
            train_local_idx, args.base_patient_id_col
        ].astype("int64")
    )
    leaked_test_ids = kept_ids.intersection(test_ids)
    if leaked_test_ids:
        raise RuntimeError(
            f"Test leakage detected: {len(leaked_test_ids)} test patients entered "
            "the training mask."
        )
    if kept_ids != train_ids:
        raise RuntimeError(
            f"Training mask selected {len(kept_ids)} patients but the manifest "
            f"contains {len(train_ids)} train patients."
        )

    print(
        f"Clinical source patients: {len(base_clinical)}; available embedding patients: "
        f"{len(new_meta)}; source overlap: {len(source_overlap_ids)}"
    )
    print(
        f"Split manifest: {len(train_ids)} train / {len(test_ids)} test; "
        f"training graph keeps {len(train_local_idx)} nodes, excludes all test patients, "
        f"and ignores {len(extra_source_patients_ignored)} extra source-covered patients."
    )
    print("Full-cohort clinical similarity graph:", cohort_graph_stats)
    print(
        "Legacy Cluster_ID/phenotype labels: ignored; "
        f"discovering {args.n_clusters} new clusters."
    )
    print(f"Output directory: {output_dir}")
    if len(train_local_idx) < args.n_clusters * 2:
        raise ValueError("Too few overlapping patients to train the requested number of clusters.")

    g_sub = dgl.node_subgraph(g_cohort, train_local_idx)
    sub_clinical = cohort_clinical.iloc[train_local_idx].reset_index(drop=True)

    new_rows = sub_clinical[args.base_patient_id_col].map(pid_to_row).to_numpy()
    new_feat = new_emb[new_rows].astype(np.float32)
    g_sub.ndata["feat"] = torch.from_numpy(new_feat)

    print(f"Training subgraph: {g_sub.num_nodes()} nodes, {g_sub.num_edges()} edges, dim={new_feat.shape[1]}")
    if args.validate_split_only:
        print("Split validation passed; exiting before model training.")
        return

    expected_outputs = [
        output_dir / "model.pth",
        output_dir / "graph.bin",
        output_dir / "cohort_graph.bin",
        output_dir / "cohort_graph_patients.csv",
        output_dir / "patient_data_expanded_v2.csv",
        output_dir / "embeddings_org.npy",
        output_dir / "embeddings_gnn.npy",
        output_dir / "run_config.json",
    ]
    existing_outputs = [path for path in expected_outputs if path.exists()]
    if existing_outputs and not args.overwrite_output:
        raise FileExistsError(
            f"Refusing to overwrite an existing GNN run in {output_dir}: "
            f"{existing_outputs}. Use a different --output-dir or pass "
            "--overwrite-output intentionally."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    g_sub = g_sub.to(device)
    g_sub.ndata["feat"] = g_sub.ndata["feat"].to(device)

    embedding_dim = new_feat.shape[1]

    print("adjacency edge count:", g_sub.num_edges())
    adj_t = None

    model = DMoNModel(
        in_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        n_clusters=args.n_clusters,
        collapse_reg=args.collapse_reg,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    counter = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(args.epochs):
        model.train()
        assignments, dmon_loss, _, modularity_loss, collapse_loss = model(g_sub, g_sub.ndata["feat"], adj_t)
        loss = dmon_loss
        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            counter = 0
        else:
            counter += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": epoch,
                "loss": current_loss,
                "modularity_loss": modularity_loss.item(),
                "collapse_loss": collapse_loss.item(),
            }
        )
        if counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (best loss: {best_loss:.4f} @ epoch {best_epoch})")
            break
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {current_loss:.4f}")

    if best_model_state is None:
        raise RuntimeError("No model checkpoint was selected during training.")
    model.load_state_dict(best_model_state)
    print(f"Restored best model from epoch {best_epoch}.")
    model.eval()
    with torch.no_grad():
        assignments, _, z, _, _ = model(g_sub, g_sub.ndata["feat"], adj_t)
    cluster_labels = torch.argmax(assignments, dim=1).cpu().numpy()

    sub_clinical["Cluster_ID"] = cluster_labels
    z_np = z.cpu().numpy()

    gnn_quality = compute_cluster_quality(z_np, cluster_labels)
    input_quality = compute_cluster_quality(new_feat, cluster_labels)
    print("Cluster quality (GNN embeddings):", gnn_quality)
    print("Cluster quality (input embeddings):", input_quality)

    # --- Save artifacts, mirroring models/train-07's layout ---
    torch.save(model, output_dir / "model.pth")
    dgl.save_graphs(str(output_dir / "graph.bin"), [g_sub.to("cpu")])
    dgl.save_graphs(str(output_dir / "cohort_graph.bin"), [g_cohort])
    cohort_clinical[
        [args.base_patient_id_col, args.split_col]
    ].to_csv(output_dir / "cohort_graph_patients.csv", index=False)
    sub_clinical.to_csv(output_dir / "patient_data_expanded_v2.csv", index=False)
    np.save(output_dir / "embeddings_org.npy", new_feat)
    np.save(output_dir / "embeddings_gnn.npy", z_np)

    run_config = {
        "base_model_dir": str(base_model_dir),
        "base_clinical_file": args.base_clinical_file,
        "new_emb_dir": str(new_emb_dir),
        "new_emb_file": args.new_emb_file,
        "new_emb_meta_file": args.new_emb_meta_file,
        "new_label_col": args.new_label_col,
        "new_patient_id_col": args.new_patient_id_col,
        "cancer_label": args.cancer_label,
        "cohort_filter_applied": args.new_label_col is not None,
        "split_manifest": str(split_manifest),
        "split_patient_id_col": args.split_patient_id_col,
        "split_col": args.split_col,
        "legacy_cluster_id_policy": "ignored_and_dropped",
        "legacy_split_phenotype_policy": "not_loaded",
        "base_graph_topology_reused": False,
        "method_reference": "GNN_exploration.ipynb cells 14-18",
        "architecture": {
            "encoder": "one-layer DGL SAGEConv",
            "aggregator": "pool",
            "activation": "ReLU",
            "output": "GraphSAGE hidden representation z",
            "edge_weights_stored": True,
            "edge_weights_used_by_encoder": False,
            "dmon_objective": (
                "notebook simplified edge-agreement modularity plus collapse loss"
            ),
        },
        "graph_construction": {
            "method": "rounded_cosine_similarity_threshold",
            "threshold": args.edge_similarity_threshold,
            "rounding_decimals": 2,
            "binary_edges": True,
            "edge_weights_used_by_model": False,
            "feature_preset": args.edge_feature_set,
            "feature_columns": list(edge_feature_columns),
            "full_cohort_graph_file": "cohort_graph.bin",
            "full_cohort_patient_file": "cohort_graph_patients.csv",
            "training_graph_file": "graph.bin",
            "full_cohort_stats": cohort_graph_stats,
            "training_graph_stats": {
                "n_nodes": int(g_sub.num_nodes()),
                "n_directed_edges": int(g_sub.num_edges()),
            },
        },
        "output_dir": str(output_dir),
        "run_name": output_dir.name,
        "n_baseline_nodes": int(len(base_clinical)),
        "n_new_emb_patients": int(len(new_meta)),
        "n_source_overlap_patients": int(len(source_overlap_ids)),
        "n_split_population_patients": int(len(manifest_ids)),
        "n_extra_source_patients_ignored": int(len(extra_source_patients_ignored)),
        "n_eligible_overlap_patients": int(len(manifest_ids)),
        "n_train_patients": int(len(train_ids)),
        "n_test_patients_excluded": int(len(test_ids)),
        "n_overlap_patients_trained": int(len(train_local_idx)),
        "verified_test_leakage_count": int(len(leaked_test_ids)),
        "embedding_dim": int(embedding_dim),
        "hyperparameters": {
            "hidden_dim": args.hidden_dim,
            "n_clusters": args.n_clusters,
            "collapse_reg": args.collapse_reg,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
        },
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "saved_model_policy": "minimum_unsupervised_training_loss",
        "epochs_trained": len(history),
        "cluster_quality_gnn_embeddings": gnn_quality,
        "cluster_quality_input_embeddings": input_quality,
        "cluster_sizes": {
            str(k): int(v) for k, v in zip(*np.unique(cluster_labels, return_counts=True))
        },
        "history": history,
    }
    with (output_dir / "run_config.json").open("w") as handle:
        json.dump(run_config, handle, indent=2)

    print(f"Saved GNN training artifacts -> {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-model-dir",
        default=str(DEFAULT_BASE_MODEL_DIR),
        help=(
            "Directory containing the clinical/radiology indicator table used "
            "to build a fresh graph. Its saved graph, cluster IDs, and model "
            "weights are not reused."
        ),
    )
    parser.add_argument("--base-clinical-file", default="patient_data_expanded_v2.csv")
    parser.add_argument("--base-patient-id-col", default="PATIENT_CLINIC_NUMBER")

    parser.add_argument(
        "--new-emb-dir", default=str(DEFAULT_NEW_EMB_DIR)
    )
    parser.add_argument("--new-emb-file", default=DEFAULT_NEW_EMB_FILE)
    parser.add_argument("--new-emb-meta-file", default=DEFAULT_NEW_EMB_META_FILE)
    parser.add_argument(
        "--new-label-col",
        default=None,
        help=(
            "Optional cohort-label column for a multi-cancer embedding table. "
            "Leave unset for the already breast-filtered transformer export."
        ),
    )
    parser.add_argument("--new-patient-id-col", default="PATIENT_CLINIC_NUMBER")
    parser.add_argument(
        "--cancer-label",
        default=None,
        help="Cohort value selected from --new-label-col when that option is set.",
    )
    parser.add_argument(
        "--split-manifest",
        default=str(DEFAULT_SPLIT_MANIFEST),
        help=(
            "CSV produced by zhemin/train_test_split.py. The current eligible "
            "overlap must be represented exactly, and only rows marked train "
            "are admitted to the GNN training graph."
        ),
    )
    parser.add_argument("--split-patient-id-col", default="PATIENT_CLINIC_NUMBER")
    parser.add_argument("--split-col", default="split")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=(
            "Parent directory for automatically named dmon_<n-clusters> run folders."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Optional exact run directory override. By default the run is written "
            "to --output-root/dmon_<n-clusters>."
        ),
    )

    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--n-clusters",
        type=int,
        required=True,
        help=(
            "Number of new phenotypes to discover. This must be chosen "
            "explicitly and is not inferred from any legacy Cluster_ID."
        ),
    )
    parser.add_argument("--collapse-reg", type=float, default=1.0)
    parser.add_argument(
        "--edge-feature-set",
        choices=sorted(EDGE_FEATURE_PRESETS),
        default="stage_ki67_radiology",
        help=(
            "Clinical indicator preset used to construct graph edges. "
            "stage_radiology excludes Ki-67 consistently from every node."
        ),
    )
    parser.add_argument(
        "--edge-similarity-threshold",
        type=float,
        default=0.5,
        help=(
            "Create an edge when the rounded cosine similarity over the "
            "configured stage/Ki-67/radiology indicators is at least this value."
        ),
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--validate-split-only",
        action="store_true",
        help="Build and validate the training-only subgraph, then exit before training.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Intentionally replace existing artifacts in the resolved run directory.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
