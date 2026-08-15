# Subject-Bert-Bench-CoE

This folder contains the reproducible scripts for the GAOKAO-to-MMLU-Pro
Subject-Bert-Bench-CoE pipeline.

## 1. GAOKAO subject leaderboard

```bash
python -m bench_coe.build_gaokao_leaderboard \
  --data-dir GAOKAO-Bench-2010-2022/Data \
  --models-dir models \
  --output-dir outputs/bench_coe/gaokao \
  --local-only
```

Main outputs:

- `outputs/bench_coe/gaokao/local_accuracy_by_subject.csv`
- `outputs/bench_coe/gaokao/local_accuracy_by_subject.png`
- `outputs/bench_coe/gaokao/local_expert_subject_mapping.json`

## 2. BERT router training

The default router is now a subject classifier: `num_labels` equals the number
of GAOKAO subjects. The expert selection is controlled by
`route_label_manifest.json` through `subject_to_model`, so changing experts does
not require retraining BERT.

```bash
CUDA_VISIBLE_DEVICES=0 python -m bench_coe.train_bert_router \
  --mapping-json outputs/bench_coe/gaokao/local_expert_subject_mapping.json \
  --objective-dir GAOKAO-Bench-2010-2022/Data/Objective_Questions \
  --bert-model models/bert-base-uncased \
  --output-dir outputs/bench_coe/router/bert-base-gaokao-subject \
  --label-mode subject \
  --epochs 5 \
  --batch-size 64 \
  --max-length 256 \
  --fp16 \
  --num-workers 0
```

Main outputs:

- `outputs/bench_coe/router/bert-base-gaokao-subject/gaokao_router_samples.jsonl`
- `outputs/bench_coe/router/bert-base-gaokao-subject/route_label_manifest.json`
- `outputs/bench_coe/router/bert-base-gaokao-subject/model`
- `outputs/bench_coe/router/bert-base-gaokao-subject/train_metrics.json`

To reproduce the older direct expert-label router, add `--label-mode expert`.

## 3. MMLU-Pro Subject-Bert-Bench-CoE evaluation

Use `FactoryS` for vLLM inference in this workspace; it has the installed vLLM
runtime and can see GPUs 0-3 when `CUDA_VISIBLE_DEVICES` is set.

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 /home/sm5/anaconda3/envs/FactoryS/bin/python \
  -m bench_coe.evaluate_subject_bert_bench_coe \
  --router-dir outputs/bench_coe/router/bert-base-gaokao-subject/model \
  --route-label-manifest outputs/bench_coe/router/bert-base-gaokao-subject/route_label_manifest.json \
  --models-dir models \
  --mmlu-data-dir MMLU-Pro/data \
  --output-dir outputs/bench_coe/mmlu_pro_subject_bert_bench_coe_smoke \
  --splits validation test \
  --max-examples-per-split 2 \
  --gpu-devices 0 \
  --tensor-parallel-size 1 \
  --max-new-tokens 16 \
  --max-model-len 2048 \
  --batch-size 2 \
  --router-device cpu \
  --resume
```

Full validation + test:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/sm5/anaconda3/envs/FactoryS/bin/python \
  -m bench_coe.evaluate_subject_bert_bench_coe \
  --router-dir outputs/bench_coe/router/bert-base-gaokao-subject/model \
  --route-label-manifest outputs/bench_coe/router/bert-base-gaokao-subject/route_label_manifest.json \
  --models-dir models \
  --mmlu-data-dir MMLU-Pro/data \
  --output-dir outputs/bench_coe/mmlu_pro_subject_bert_bench_coe \
  --splits validation test \
  --gpu-devices 0,1,2,3 \
  --expert-gpus 0,1,2,3 \
  --parallel-experts \
  --tensor-parallel-size 1 \
  --max-new-tokens 1024 \
  --max-model-len 4096 \
  --batch-size 256 \
  --router-batch-size 512 \
  --router-device cuda \
  --resume
```

Eight-GPU version:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /home/sm5/anaconda3/envs/FactoryS/bin/python \
  -m bench_coe.evaluate_subject_bert_bench_coe \
  --router-dir outputs/bench_coe/router/bert-base-gaokao-subject/model \
  --route-label-manifest outputs/bench_coe/router/bert-base-gaokao-subject/route_label_manifest.json \
  --models-dir models \
  --mmlu-data-dir MMLU-Pro/data \
  --output-dir outputs/bench_coe/mmlu_pro_subject_bert_bench_coe \
  --splits validation test \
  --gpu-devices 0,1,2,3,4,5,6,7 \
  --expert-gpus 0,1,2,3,4,5,6,7 \
  --parallel-experts \
  --tensor-parallel-size 1 \
  --max-new-tokens 1024 \
  --max-model-len 4096 \
  --batch-size 256 \
  --router-batch-size 512 \
  --router-device cuda \
  --resume
```

The parallel evaluator assigns at most one vLLM worker to each GPU at a time.
If there are fewer routed experts than GPUs, it splits the largest expert's
examples over multiple single-GPU replicas. If there are more experts than GPUs,
it runs multiple waves.

## 4. Reversed pipeline: MMLU-Pro prior -> GAOKAO

Build the MMLU-Pro category leaderboard and category-to-expert mapping:

```bash
python -m bench_coe.build_mmlu_leaderboard \
  --evaluation-summary-root outputs/bench_coe/mmlu_pro_validation_single_models \
  --expert-pool-config bench_coe/configs/expert_pools.json \
  --expert-pool language_7b_9b_specialists \
  --output-dir outputs/bench_coe/mmlu_validation_7b_9b \
  --prefix validation_7b_9b
```

This configuration compares only the 7B-9B language models and excludes
`Qwen3-8B`, `Qwen3.5-9B`, and `DeepSeek-R1-0528-Qwen3-8B` from specialist
selection. The mapping is built from the MMLU-Pro validation evaluations rather
than the held-out test summaries. Exact per-subject ties are resolved using the
higher overall validation accuracy, followed by the stable model index.

Train a BERT router whose labels are the 14 MMLU-Pro categories:

Prepare and inspect the leakage-free validation labels first:

```bash
python -m bench_coe.train_mmlu_router \
  --mapping-json outputs/bench_coe/mmlu_validation_7b_9b/validation_7b_9b_expert_category_mapping.json \
  --mmlu-data-dir MMLU-Pro/data \
  --splits validation \
  --output-dir outputs/bench_coe/router/mmlu_validation_7b_9b_subject_labels \
  --prepare-only
```

Then train the classifier with the same mapping and source split:

```bash
CUDA_VISIBLE_DEVICES=0 python -m bench_coe.train_mmlu_router \
  --mapping-json outputs/bench_coe/mmlu_validation_7b_9b/validation_7b_9b_expert_category_mapping.json \
  --mmlu-data-dir MMLU-Pro/data \
  --splits validation \
  --bert-model models/bert-base-uncased \
  --output-dir outputs/bench_coe/router/bert-base-mmlu-validation-7b-9b-category \
  --epochs 3 \
  --batch-size 64 \
  --max-length 256 \
  --fp16 \
  --num-workers 0
```

Evaluate on GAOKAO with the existing per-model GAOKAO result files. This is an
offline evaluator: it routes each GAOKAO question, selects the mapped expert's
stored answer, and scores it without rerunning vLLM generation.

```bash
CUDA_VISIBLE_DEVICES=0 python -m bench_coe.evaluate_mmlu_router_on_gaokao \
  --router-dir outputs/bench_coe/router/bert-base-mmlu-validation-7b-9b-category/model \
  --route-label-manifest outputs/bench_coe/router/bert-base-mmlu-validation-7b-9b-category/route_label_manifest.json \
  --gaokao-data-dir GAOKAO-Bench-2010-2022/Data \
  --benchmark gaokao2010 \
  --output-dir outputs/bench_coe/mmlu_router_on_gaokao2010 \
  --router-device cuda \
  --router-batch-size 512 \
  --reextract-empty
```
