from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

from bench_coe.evaluate_tinyllava_subject_router import load_expert_rows, load_ids
from bench_coe.gaokao_utils import write_json


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Evaluate a cached-feature direct expert query router.")
    parser.add_argument("--feature-cache",type=Path,required=True); parser.add_argument("--classifier",type=Path,required=True)
    parser.add_argument("--route-label-manifest",type=Path,required=True); parser.add_argument("--predictions-root",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--include-ids-json",type=Path); parser.add_argument("--query-labels",type=Path)
    parser.add_argument("--model-prefix",default="")
    return parser.parse_args()


def read_json(path:Path)->Any:
    with path.open('r',encoding='utf-8') as f:return json.load(f)


def read_jsonl(path:Path)->list[dict[str,Any]]:
    with path.open('r',encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]


def main()->None:
    args=parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True); manifest=read_json(args.route_label_manifest); route_models=list(manifest['model_names'])
    eligible_route_models=[model for model in route_models if str(model).startswith(args.model_prefix)] if args.model_prefix else route_models
    requested_models=[str(model)[len(args.model_prefix):] for model in eligible_route_models]
    models=[model for model in requested_models if (args.predictions_root/model/'predictions.jsonl').exists()]
    missing_models=[model for model in requested_models if model not in models]
    include_ids=load_ids(args.include_ids_json); rows,experts=load_expert_rows(args.predictions_root,models,include_ids); ids=[str(r['id']) for r in rows]
    cache=torch.load(args.feature_cache,map_location='cpu',weights_only=False); lookup={str(v):i for i,v in enumerate(cache['ids'])}; features=cache['features'][[lookup[i] for i in ids]]
    ckpt=torch.load(args.classifier,map_location='cpu',weights_only=False); model=nn.Sequential(nn.LayerNorm(int(ckpt['feature_dim'])),nn.Dropout(.15),nn.Linear(int(ckpt['feature_dim']),int(ckpt['num_labels']))); model.load_state_dict(ckpt['state_dict']); model.eval()
    with torch.inference_mode(): labels=model(features).argmax(-1).tolist()
    targets={str(r['id']):int(r['label']) for r in read_jsonl(args.query_labels)} if args.query_labels else {}
    output=[]; single=Counter(); oracle=0; label_correct=label_total=0
    for row,label in zip(rows,labels):
        sample_id=str(row['id']); routed_key=str(manifest['route_label_to_model'][str(label)]); route_valid=(not args.model_prefix) or routed_key.startswith(args.model_prefix); routed_model=routed_key[len(args.model_prefix):] if route_valid else None; route_available=route_valid and routed_model in experts; routed=experts[routed_model][sample_id] if route_available else {}
        correctness={m:bool(experts[m][sample_id].get('is_correct',False)) for m in models}
        for m,v in correctness.items():single[m]+=int(v)
        oracle+=int(any(correctness.values()))
        if sample_id in targets: label_total+=1; label_correct+=int(label==targets[sample_id])
        output.append({'id':sample_id,'route_label':label,'routed_key':routed_key,'route_valid_for_modality':route_valid,'route_expert_available':route_available,'routed_model':routed_model,'is_correct':bool(routed.get('is_correct',False)),'prediction':routed.get('prediction'),'answer':routed.get('answer')})
    total=len(output); single_acc={m:single[m]/total for m in models}; best=max(models,key=lambda m:(single_acc[m],-models.index(m)))
    summary={'count':total,'routed_accuracy':sum(int(r['is_correct']) for r in output)/total,'query_label_accuracy':label_correct/label_total if label_total else None,'query_label_count':label_total,'cross_modality_route_count':sum(int(not r['route_valid_for_modality']) for r in output),'unavailable_expert_route_count':sum(int(r['route_valid_for_modality'] and not r['route_expert_available']) for r in output),'available_models':models,'missing_models':missing_models,'oracle_any_expert_accuracy':oracle/total,'single_model_accuracies':single_acc,'best_single_model':best,'best_single_accuracy':single_acc[best],'routed_model_counts':dict(Counter(str(r['routed_model']) for r in output))}
    write_json(args.output_dir/'predictions.json',output); write_json(args.output_dir/'summary.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
