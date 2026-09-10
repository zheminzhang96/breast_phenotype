# External institution GNN evaluation

`run_gnn_external.py` applies the frozen stage+radiology GNN models to an
external cohort. It does not train, tune, or select a model on external data,
and it has no SEER-specific file paths or parsers.

Use `run_all_external.py` first when episode representations still need to be
created from note embeddings. Its output directory is the representation root
expected here; see `EXTERNAL_STAGE02_STAGE03_HANDOFF.md` for that upstream data
contract.

When using that output, pass --patient-id-column patient_group_id and list the
representations actually produced with --pipelines. A Stage 1 run with one
Transformer normally produces mean, time_decay, and transformer_combined; an
optional DML run adds names such as mean_icd_supcon. The graph-feature CSV
should use the same deidentified episode_id values as the upstream
episode_metadata.csv.

## Input contract

The external institution supplies three aligned inputs:

1. **Episode metadata CSV**

   One row per episode. It must contain an episode ID, a patient ID, and every
   endpoint requested with `--endpoints`. IDs may be strings or integers.
   Endpoint labels may be numeric or categorical. Every representation array
   must use this exact row order.

2. **Episode graph-feature CSV**

   One row per episode, with an episode ID and the 35 binary features expected
   by the frozen model: 7 broad-stage indicators and 28 radiology indicators.
   Exactly one broad-stage indicator must be active per episode. Print the
   canonical names with:

   ```bash
   python -m zhemin.pipeline.seer_external.run_gnn_external \
     --print-required-columns
   ```

   If local column names differ, pass `--feature-map feature_map.json`. The
   JSON keys are canonical names and values are local CSV column names. The map
   may be partial; omitted names are assumed to already be canonical.

3. **Representation arrays**

   By default, each selected pipeline is read from:

   ```text
   <representation-root>/representations/<pipeline>/episode.emb.npy
   ```

   Each file must be a finite `N x D` NumPy array, where `N` equals the full
   metadata row count. Use `--embedding-template` for another layout.

## Frozen artifact bundle

The sending institution supplies the frozen model bundle under:

```text
<model-root>/<pipeline>/<model-subdir>/
  model.pth
  run_config.json
  graph.bin
  embeddings_org.npy
  embeddings_gnn.npy
```

It must also supply a deidentified `N_train x 35` `.npy` or `.csv` file via
`--training-graph-features-path`. Rows must be in the exact internal node order
used by `graph.bin`. The external script verifies that these features reproduce
the saved training graph before inference. Do not distribute the source
clinical table merely to provide these 35 columns.

Full-object PyTorch checkpoints require the same project code and compatible
PyTorch/DGL versions used to create them. The model bundle and deidentified
training arrays should still undergo the institution's disclosure review.

## Run

Start with validation only:

```bash
python -m zhemin.pipeline.seer_external.run_gnn_external \
  --model-root /path/to/frozen_models \
  --training-graph-features-path /path/to/training_graph_features.npy \
  --representation-root /path/to/external_representations \
  --metadata-path /path/to/episodes.csv \
  --graph-features-path /path/to/episode_graph_features.csv \
  --episode-id-column encounter_id \
  --patient-id-column local_patient_id \
  --endpoints stage histology grade er pr her2 recurrence_any \
  --audit-only
```

Then remove `--audit-only`, select a device, and choose one or more pipelines:

```bash
python -m zhemin.pipeline.seer_external.run_gnn_external \
  --model-root /path/to/frozen_models \
  --training-graph-features-path /path/to/training_graph_features.npy \
  --representation-root /path/to/external_representations \
  --metadata-path /path/to/episodes.csv \
  --graph-features-path /path/to/episode_graph_features.csv \
  --episode-id-column encounter_id \
  --patient-id-column local_patient_id \
  --endpoints stage histology grade er pr her2 recurrence_any \
  --pipelines mean mean_icd_supcon \
  --device cuda:0 \
  --output-dir external_gnn_results
```

Use `--eligibility-column` to honor a pre-specified cohort flag. Add
`--require-radiology-report` if the protocol excludes episodes without a linked
radiology report; this requires the graph CSV's `n_radiology_reports` column,
or a different count column selected with `--radiology-count-column`.

## Outputs and interpretation

The main files are `retrieval_summary.csv`, the before/after wide tables,
`retrieval_gnn_incremental.csv`, and `cluster_assignments.csv`. Retrieval is
leave-one-patient-out, and confidence intervals use patient-level clustered
bootstrap sampling.

Source episode and patient IDs are never written. The output uses run-local
integer IDs, and `data_audit.json` stores endpoint label-code mappings.

An endpoint represented in the graph features is not label-blind. In
particular, stage retrieval is graph-informed when broad stage is used to build
edges and must be reported as a sensitivity analysis, not independent external
prediction performance.
