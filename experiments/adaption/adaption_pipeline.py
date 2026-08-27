"""
adaption_pipeline.py -- reproducible record of the Adaption run (2026-08-27).

What this documents, step by step (run with `adaption>=0.9.0`):
  1. Build a training CSV from the moat corpus (v3 train split, clean rows,
     prompt + templated completion + domain context).
  2. Upload it to Adaption via presigned URL.
  3. Request a FREE credit estimate before any paid run.
  4. Run Adaptive Data (column mapping: prompt/completion/context=domain).
  5. Download the adapted dataset (the `enhanced_completion` column carries
     the model-generated, domain-grounded answers -- this is the durable
     asset: corpus/moat_brick4_adapted.csv).
  6. Re-upload the enhanced pairs as a RAW (training-ready) dataset.
  7. Launch AutoScientist (Qwen3.5-0.8B, LoRA, instruction format,
     no synthetic augmentation, max 3 iterations) and download the best
     checkpoint.

Canonical run values (2026-08-27):
  v3 corpus dataset : 1c2854e6-4af0-4017-a4d6-4e5d60030638 (1,768 rows ingested)
  adapted rows      : 1,117 (platform deduped/filtered during evaluation)
  evaluation grade  : D -> B (4.0 -> 7.5, +87.5%) on the source dataset
  enhanced dataset  : 88c24bed-0adf-48d0-b780-94dbf9e63fc0 (raw mode, 1,117 rows)
  AutoScientist run : e2cdb7be-9cf8-4463-b438-4ec4036f5d29
  model             : Qwen/Qwen3.5-0.8B, LoRA r16/alpha32 (recommended hyperparams)
  credit cost       : 18 credits (Adaptive Data, 1,673-row corpus)
  external imports  : 5G-NR QA (26,926 rows), Agriculture QA (22,615),
                      Personal Finance Africa (201) -- 12 credits total at 500/200-row caps

Notes / gotchas learned:
  - `datasets.create` with processing_mode="raw" + inline column_mapping makes a
    training-ready dataset; plain create leaves it in awaiting_column_selection.
  - AutoScientist.create with an explicit column_mapping failed with
    "columns not in this dataset" -- omitting it lets the platform infer from
    the raw dataset's configured mapping.
  - The presigned PUT needs generous httpx timeouts on slow connections
    (600s total; default 5s write timeout fails on ~5MB CSVs).
  - estimate=True returns estimated_credits_consumed WITHOUT starting the run.
"""
