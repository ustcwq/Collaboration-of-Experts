# Bench-CoE ModelScope Resource Preparation

This repository provides a ModelScope-only, auditable resource pipeline for the dynamic expert experiments. It does not run benchmark comparisons.

## Environment

```bash
export BENCHCOE_ASSET_ROOT=/absolute/path/to/benchcoe_assets
export MODELSCOPE_CACHE=/absolute/path/to/modelscope_cache
export BENCHCOE_DATA_ROOT=/absolute/path/to/repository/data
export BENCHCOE_PROFILE=smoke
export MODELSCOPE_DOMAIN=modelscope.cn
# export MODELSCOPE_API_TOKEN=...  # only for resources requiring authentication
```

Do not point either path at `/`, `$HOME`, or `~`. The downloader requires 1.25 times the known snapshot size in free space and blocks unknown-size downloads unless the operator explicitly accepts that risk.

## Workflow

```bash
python tools/modelscope_assets.py resolve --profile smoke
python tools/modelscope_assets.py estimate --profile smoke
python tools/modelscope_assets.py download --profile smoke
python tools/modelscope_assets.py verify --profile smoke

python tools/preprocess_benchmarks.py \
  --dataset mmlu_pro --split validation --smoke

python tools/modelscope_assets.py lock --profile smoke
python tools/modelscope_assets.py report --profile smoke
python tools/verify_assets.py
```

`resolve` uses only the installed ModelScope SDK. Search-only resources remain unresolved until an exact candidate is verified; no Hugging Face or other fallback is attempted. Downloads use stable incomplete directories, file locks, retries, a ready marker, and atomic publication. Failed snapshots move to `quarantine/`.

## Outputs

The asset root contains separate model, processed dataset, image, lock, manifest, log, temporary, and quarantine directories. Set `BENCHCOE_DATA_ROOT` to place downloaded raw datasets in the repository's `data/` directory without duplicating them under the asset root. The immutable contract files are:

- `manifests/asset_lock.json` and `manifests/asset_lock.sha256`
- `manifests/protocol_lock.yaml` and `manifests/protocol_lock.sha256`
- `manifests/model_registry.resolved.json`
- `manifests/dataset_registry.resolved.json`
- `manifests/resource_status.csv`
- `manifests/disk_report.json`
- `manifests/licenses_report.md`
- `manifests/overlap_report.jsonl`
- `manifests/preparation_report.md`

Each processed dataset contains `samples.jsonl`, `dataset_manifest.json`, `qa_report.json`, `id_map.jsonl`, `overlap_report.jsonl`, and `README.generated.md` using `benchcoe_unified_v1`.

## Safety Contract

- Source labels are available only for source capability, Improve5 local reliability, Improve6 repair edges, and source-only validation.
- Target labels are final-scoring only.
- A benchmark split cannot be both source and target.
- Qwen3 uses `enable_thinking=False`; prompt variants and seeds are not independent experts.
- Lock mismatch is a hard failure. The verifier never repairs, redownloads, deletes, or silently ignores mismatches.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic resources and do not require downloading full models.
