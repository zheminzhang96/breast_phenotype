# Note Embeddings — Setup and Usage

How to download the embedding model and run `gen_note_embeddings_generic.py` on
your own CSV of clinical notes.

The script embeds one note per row using
[`abhinand/MedEmbed-base-v0.1`](https://huggingface.co/abhinand/MedEmbed-base-v0.1)
and writes three aligned files per run.

---

## 1. Requirements

Python 3.10 or newer, and roughly 1 GB of free disk for the model cache.
A GPU is optional — the script runs on CPU, just slower.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install "torch>=2.1" sentence-transformers transformers "numpy<2" pandas
```

Tested with the versions below. Others will very likely work; these are what the
published results were produced with.

| Package | Version |
| --- | --- |
| Python | 3.10.19 |
| torch | 2.9.1+cu128 |
| sentence-transformers | 5.1.2 |
| transformers | 4.57.6 |
| huggingface_hub | 0.36.0 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |

`numpy<2` is pinned because some prebuilt torch wheels are still compiled
against numpy 1.x.

For GPU support, install the torch build matching your CUDA version from
<https://pytorch.org/get-started/locally/> instead of the plain `torch` wheel.

---

## 2. Download the model

The model is about 440 MB. You do **not** have to download it manually — the
first run fetches it automatically and caches it in `~/.cache/huggingface`.
Every later run loads from that cache with no network access.

To fetch it ahead of time (recommended, so a long embedding job does not fail
partway on a network hiccup):

```bash
hf download abhinand/MedEmbed-base-v0.1
```

On older `huggingface_hub` versions the command is
`huggingface-cli download abhinand/MedEmbed-base-v0.1`. From Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("abhinand/MedEmbed-base-v0.1")
```

### Downloading to a specific folder

Useful if you want the weights inside the project, or need to copy them to a
machine with no internet:

```bash
hf download abhinand/MedEmbed-base-v0.1 --local-dir ./models/MedEmbed-base-v0.1
```

Then point the script at that folder instead of the hub name:

```bash
python gen_note_embeddings_generic.py \
    --input-csv notes.csv --name mycohort \
    --model-name ./models/MedEmbed-base-v0.1
```

---

## 3. Choose your GPU — read this before your first run

The script sets a default GPU index at the top of the file:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "3")
```

`"3"` is specific to the 4-GPU machine this was written on. **On a machine with
fewer than four GPUs, that index does not exist, CUDA silently becomes
unavailable, and the script falls back to CPU** — it still produces correct
output, just far slower, and the only hint is `Using device: cpu` in the log.

Set the variable explicitly for your machine:

```bash
# single GPU, or "use the first one"
CUDA_VISIBLE_DEVICES=0 python gen_note_embeddings_generic.py ...

# force CPU
CUDA_VISIBLE_DEVICES="" python gen_note_embeddings_generic.py ...
```

Check what you have with `nvidia-smi`. Whatever index you pick becomes
`cuda:0` inside the process, which is why the log always says `cuda:0`.

If you are adapting this script for your own site, changing that `"3"` to `"0"`
is a sensible first edit.

---

## 4. Prepare your input CSV

**Required** — two text columns, joined as `summary + "\n\n" + paraphrase` to
form the model input:

| Column | Purpose |
| --- | --- |
| `summary` | LLM-generated note summary |
| `ParaphraseDemographics_0.30` | Demographic paraphrase of the note |

If your columns are named differently, pass `--summary-col` and
`--paraphrase-col`. Empty or missing values are treated as empty strings, so a
row with only one of the two still embeds fine.

**Optional** — copied into the metadata CSV when present, silently skipped when
absent: `__note_rowid`, `PATIENT_CLINIC_NUMBER`,
`CLINICAL_DOCUMENT_ORIGINAL_DTM`, `Reference_Dx_Date`, `dxdate`, `dt_days`.

To keep different columns, edit the `META_COLS` list near the top of the script.
Downstream code in this repo expects `PATIENT_CLINIC_NUMBER` and `dt_days`, so
keep those if you plan to use it.

---

## 5. Run it

```bash

CUDA_VISIBLE_DEVICES=0 python gen_note_embeddings_generic.py \
    --input-csv csv_for_emory_breast_cancer_with_summary+paraphrase.csv \
    --name breast
```

Output goes to `phase1/emb_data/note_embeddings/medembed/` unless you pass
`--out-dir`.

| Option | Default | Notes |
| --- | --- | --- |
| `--input-csv` | *required* | CSV to embed |
| `--name` | *required* | Output filename prefix; also written to the `cancer_type` column |
| `--out-dir` | `phase1/emb_data/note_embeddings/medembed` | Created if missing |
| `--summary-col` | `summary` | |
| `--paraphrase-col` | `ParaphraseDemographics_0.30` | |
| `--model-name` | `abhinand/MedEmbed-base-v0.1` | Hub id or local folder |
| `--batch-size` | `128` | Lower it if you hit out-of-memory |
| `--max-length` | `512` | Tokens per note; text beyond this is truncated |

`BATCH_SIZE` and `MAX_LENGTH` also work as environment variables.

### Expected output

Three files, all row-aligned with each other:

```text
{out_dir}/{name}.note.emb.npy    float32 (n_notes, 768), L2-normalized
{out_dir}/{name}.note.meta.csv   metadata + cancer_type + emb_row_idx
{out_dir}/{name}.note.run.json   model, parameters, shapes, timing
```

Row `i` of the `.npy` corresponds to row `i` of the `.meta.csv` and to row `i`
of your input CSV. Identical note texts are encoded once and share a vector, so
`n_unique_texts` in the run JSON is usually lower than `n_rows` — this is a
speed optimization and does not change any output vector.

A run looks like this:

```text
Using device: cuda:0
Reading: step2_ehr_textulization/CN_EHR_myeloma_min2_paraphrase.csv
  Rows loaded: 9,083

Loading abhinand/MedEmbed-base-v0.1 ...
  Encoding 8,269 unique texts from 9,083 rows (batch_size=128, max_length=512) ...
  Done in 0.1 min  |  shape=(9083, 768)
```

---

## 6. Check the result

```python
import json
import numpy as np
import pandas as pd

emb = np.load("../medembed/breast.note.emb.npy")
meta = pd.read_csv("../medembed/breast.note.meta.csv")
run = json.load(open("../medembed/breast.note.run.json"))

assert emb.shape[0] == len(meta), "embeddings and metadata are misaligned"
assert emb.shape[1] == 768, f"unexpected dimension {emb.shape[1]}"
assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3), "not normalized"
print(f"{emb.shape[0]:,} notes x {emb.shape[1]} dims, model {run['model']}")
```

Because vectors are L2-normalized, cosine similarity is just a dot product:

```python
sims = emb @ emb[0]
```


