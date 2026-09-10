# External patient-embedding evaluation

run_all_external.py is a frozen two-stage pipeline. It never trains, tunes, or
selects a model on external data.

~~~text
note embeddings
      |
      v
Stage 1: mean + time-decay + optional temporal Transformer
      |
      v
patient-level embeddings
      |
      v
Stage 2 (optional): frozen ICD-supervised DML adapters
      |
      v
adapted patient-level embeddings
~~~

## Important ICD detail

The DML adapters were trained internally using ICD comorbidity similarity as
supervision. The saved adapter architecture takes only a 768-dimensional
patient embedding at inference time.

The external institution does not provide ICD codes to this script. Enabling
--run-dml means "apply weights learned with ICD supervision," not "add
external ICD features." A model that consumes external ICD codes would be a
different model and evaluation protocol.

## Input files

Input identifiers must already be deidentified. The two CSV files use fixed,
canonical column names to keep the handoff simple.

### 1. Note embeddings

notes.emb.npy is a floating-point NumPy array:

~~~text
shape: (N_notes, 768)
row i: embedding for row i of notes.meta.csv
~~~

The note embedding model, text fields, preprocessing, and truncation should
match model development. Use --input-protocol-status matched only after this
has been verified.

### 2. Note metadata

notes.meta.csv has exactly one row per note embedding:

~~~csv
episode_id,patient_id,days_from_diagnosis
episode_001,patient_001,-30
episode_001,patient_001,-2
episode_001,patient_001,0
episode_001,patient_001,14
~~~

- episode_id: deidentified diagnosis-episode identifier.
- patient_id: deidentified patient identifier.
- days_from_diagnosis: negative before diagnosis, zero on diagnosis day, and
  positive after diagnosis.

The CSV row order must exactly match the NumPy array row order.

### 3. Episode metadata

episodes.csv has exactly one row per diagnosis episode:

~~~csv
episode_id,patient_id,eligible,stage,histology,grade,er,pr,her2,recurrence_any
episode_001,patient_001,true,II,IDC,2,positive,positive,negative,0
episode_002,patient_002,true,I,ILC,1,positive,negative,negative,1
~~~

- episode_id and patient_id must match the note metadata.
- eligible is optional. Missing means all episodes are institution-eligible.
- Evaluation columns are optional and are named with --endpoints.
- Endpoint values may be numeric or categorical.
- Missing labels may be empty or one of the configured missing-label values.

The script additionally requires at least --min-pre-notes notes with a
negative diagnosis-day offset. The default is two.

Print the machine-readable contract with:

~~~bash
python -m zhemin.pipeline.seer_external.run_all_external --describe-formats
~~~

## Weights to share

Stage 1 pooling needs no weights. To include the Transformer, share one
checkpoint and pass it with --transformer-checkpoint. Its
--transformer-name must match the DML base directory; the default is
transformer_combined.

For optional Stage 2, share each desired adapter and its configuration:

~~~text
dml_weights/
  mean/
    triplet/best.pt
    triplet/run_config.json
    supcon/best.pt
    supcon/run_config.json
    fastap/best.pt
    fastap/run_config.json
  time_decay/
    <objective>/best.pt
    <objective>/run_config.json
  transformer_combined/
    <objective>/best.pt
    <objective>/run_config.json
~~~

Only directories for requested --dml-objectives are required.

## Stage 1 only

This writes mean, time-decay, and Transformer patient embeddings:

~~~bash
python -m zhemin.pipeline.seer_external.run_all_external \
  --note-emb-path /external/notes.emb.npy \
  --note-meta-path /external/notes.meta.csv \
  --episode-meta-path /external/episodes.csv \
  --note-run-path /external/notes.run.json \
  --transformer-checkpoint /weights/transformer_combined.best.pt \
  --transformer-name transformer_combined \
  --endpoints stage histology grade er pr her2 recurrence_any \
  --input-protocol-status matched \
  --device cuda:0 \
  --output-dir /external/results
~~~

Omit --transformer-checkpoint to produce only mean and time-decay
representations. Omit --endpoints to generate embeddings without running
within-institution retrieval.

## Stage 1 plus optional DML

Add --run-dml and the DML weight root:

~~~bash
python -m zhemin.pipeline.seer_external.run_all_external \
  --note-emb-path /external/notes.emb.npy \
  --note-meta-path /external/notes.meta.csv \
  --episode-meta-path /external/episodes.csv \
  --transformer-checkpoint /weights/transformer_combined.best.pt \
  --transformer-name transformer_combined \
  --run-dml \
  --dml-root /weights/dml_weights \
  --dml-objectives triplet supcon fastap \
  --endpoints stage histology grade er pr her2 recurrence_any \
  --input-protocol-status matched \
  --device cuda:0 \
  --output-dir /external/results
~~~

Run input validation first by adding --audit-only.

## Output files

episode_metadata.csv defines the row order for every patient embedding:

~~~csv
episode_id,evaluation_episode_id,patient_group_id,n_notes,n_pre_notes,n_dx_notes,n_post_notes,stage
episode_001,1,1,4,2,1,1,II
~~~

The source patient_id is removed. The deidentified source episode_id is
retained so the institution can join separately prepared graph features.

Stage 1 arrays:

~~~text
representations/mean/episode.emb.npy
representations/time_decay/episode.emb.npy
representations/transformer_combined/episode.emb.npy
~~~

Optional Stage 2 arrays follow this pattern:

~~~text
representations/mean_icd_triplet/episode.emb.npy
representations/mean_icd_supcon/episode.emb.npy
representations/mean_icd_fastap/episode.emb.npy
~~~

Every array has shape (N_included_episodes, 768) and the exact row order of
episode_metadata.csv.

Other outputs:

~~~text
cohort_flow.csv
note_selection.csv
transformer_stage_predictions.csv
data_audit.json
run_config.json
retrieval_summary.csv                 # only with --endpoints
retrieval_per_query.csv               # only with --endpoints
retrieval_neighbors.csv               # only with --endpoints
retrieval_dml_incremental.csv         # only with --run-dml and --endpoints
~~~

The result directory can be passed directly as --representation-root to
run_gnn_external.py. Its graph-feature CSV should use the same deidentified
episode_id values.
