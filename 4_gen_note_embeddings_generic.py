#!/usr/bin/env python3
"""

Model input per row is:  summary + "\\n\\n" + ParaphraseDemographics_0.30

Outputs:
  {out_dir}/{name}.note.emb.npy   - float32 array (n_notes, embedding_dim)
  {out_dir}/{name}.note.meta.csv  - key columns aligned row-by-row with .npy
  {out_dir}/{name}.note.run.json  - run config/stats

Examples:
  python gen_note_embeddings_generic.py --input-csv step2_ehr_textulization/CN_EHR_myeloma_min2_paraphrase.csv --name myeloma

  python gen_note_embeddings_generic.py \\
      --input-csv /path/to/notes.csv --name melanoma \\
      --summary-col summary --paraphrase-col ParaphraseDemographics_0.30

To adapt this to another corpus, the usual changes are --summary-col,
--paraphrase-col, and META_COLS below.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "3")

import torch
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "abhinand/MedEmbed-base-v0.1"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
OUT_DIR = ROOT / "phase1" / "emb_data" / "note_embeddings" / "medembed"

SUMMARY_COL = "summary"
PARA_COL = "ParaphraseDemographics_0.30"

# Copied into the metadata CSV. Columns the input does not have are skipped,
# so this list can cover several corpora (breast/prostate use Reference_Dx_Date,
# myeloma uses dxdate).
META_COLS = [
    "__note_rowid",
    "PATIENT_CLINIC_NUMBER",
    "CLINICAL_DOCUMENT_ORIGINAL_DTM",
    "Reference_Dx_Date",
    "dxdate",
    "dt_days",
]

# ── Script ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True, help="Paraphrase CSV to embed.")
    parser.add_argument(
        "--name",
        required=True,
        help="Output filename prefix, also written to the cancer_type column.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--summary-col", default=SUMMARY_COL)
    parser.add_argument("--paraphrase-col", default=PARA_COL)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    return parser.parse_args()


def build_texts(df: pd.DataFrame, summary_col: str, para_col: str) -> list[str]:
    s = df[summary_col].fillna("").astype(str).str.strip()
    p = df[para_col].fillna("").astype(str).str.strip()
    # Replace literal "nan" strings that can appear after CSV round-trips
    s = s.replace({"nan": "", "NaN": "", "None": ""})
    p = p.replace({"nan": "", "NaN": "", "None": ""})
    return (s + "\n\n" + p).str.strip().tolist()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Reading: {args.input_csv}")

    t0 = time.time()
    df = pd.read_csv(args.input_csv, low_memory=False)
    print(f"  Rows loaded: {len(df):,}")

    for col in (args.summary_col, args.paraphrase_col):
        if col not in df.columns:
            raise SystemExit(f"Input CSV has no column {col!r}. Use --summary-col / --paraphrase-col.")

    texts = build_texts(df, args.summary_col, args.paraphrase_col)
    if not texts:
        raise SystemExit("The input CSV contains no rows to embed.")

    print(f"\nLoading {args.model_name} ...")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = SentenceTransformer(
        args.model_name,
        device=device,
        model_kwargs={"torch_dtype": dtype, "attn_implementation": "eager"},
    )
    model.max_seq_length = args.max_length

    # Encode each distinct text once, then put the vectors back in row order.
    unique_texts = list(dict.fromkeys(texts))
    print(
        f"  Encoding {len(unique_texts):,} unique texts from {len(texts):,} rows "
        f"(batch_size={args.batch_size}, max_length={args.max_length}) ..."
    )
    unique_emb = model.encode(
        unique_texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    emb_by_text = dict(zip(unique_texts, unique_emb))
    emb = np.stack([emb_by_text[text] for text in texts]).astype(np.float32)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min  |  shape={emb.shape}")

    emb_path = args.out_dir / f"{args.name}.note.emb.npy"
    meta_path = args.out_dir / f"{args.name}.note.meta.csv"
    run_path = args.out_dir / f"{args.name}.note.run.json"

    # Write to a temp file first so an interrupted run leaves no partial output.
    emb_tmp_path = emb_path.with_suffix(".tmp.npy")
    np.save(emb_tmp_path, emb)
    os.replace(emb_tmp_path, emb_path)

    meta_cols = [c for c in META_COLS if c in df.columns]
    meta_df = df[meta_cols].copy()
    meta_df["cancer_type"] = args.name
    meta_df["emb_row_idx"] = np.arange(len(df))
    meta_tmp_path = meta_path.with_suffix(".tmp.csv")
    meta_df.to_csv(meta_tmp_path, index=False)
    os.replace(meta_tmp_path, meta_path)

    run_info = {
        "cancer": args.name,
        "csv": str(args.input_csv),
        "model": args.model_name,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "n_rows": len(df),
        "n_unique_texts": len(unique_texts),
        "embedding_dim": int(emb.shape[1]),
        "emb_shape": list(emb.shape),
        "emb_path": str(emb_path),
        "meta_path": str(meta_path),
        "elapsed_sec": round(elapsed, 1),
        "device": device,
    }
    run_tmp_path = run_path.with_suffix(".tmp.json")
    with run_tmp_path.open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    os.replace(run_tmp_path, run_path)

    print(f"  Saved: {emb_path}")
    print(f"  Saved: {meta_path}")
    print(f"  Saved: {run_path}")


if __name__ == "__main__":
    main()
