from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bench_coe.gaokao_utils import write_json
from bench_coe.train_bert_router import stratified_split


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--backend',choices=['bert','feature'],required=True); p.add_argument('--samples-jsonl',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--split-manifest',type=Path); p.add_argument('--evaluate-all',action='store_true'); p.add_argument('--available-models',default=''); p.add_argument('--bert-model',type=Path); p.add_argument('--feature-cache',type=Path); p.add_argument('--classifier',type=Path); p.add_argument('--validation-fraction',type=float,default=.15); p.add_argument('--seed',type=int,default=42); p.add_argument('--batch-size',type=int,default=128); return p.parse_args()


def read_json(path:Path)->Any:
    with path.open() as f:return json.load(f)


def read_jsonl(path:Path):
    with path.open() as f:return [json.loads(x) for x in f if x.strip()]


def predict(args,rows):
    if args.backend=='feature':
        cache=torch.load(args.feature_cache,map_location='cpu',weights_only=False); lookup={str(v):i for i,v in enumerate(cache['ids'])}; x=cache['features'][[lookup[r['id']] for r in rows]]; ck=torch.load(args.classifier,map_location='cpu',weights_only=False); m=nn.Sequential(nn.LayerNorm(int(ck['feature_dim'])),nn.Dropout(.15),nn.Linear(int(ck['feature_dim']),int(ck['num_labels']))); m.load_state_dict(ck['state_dict']); m.eval();
        with torch.inference_mode():return m(x).argmax(-1).tolist()
    tokenizer=AutoTokenizer.from_pretrained(str(args.bert_model),local_files_only=True); model=AutoModelForSequenceClassification.from_pretrained(str(args.bert_model),local_files_only=True).cuda().eval(); out=[]
    for start in range(0,len(rows),args.batch_size):
        enc=tokenizer([r['text'] for r in rows[start:start+args.batch_size]],truncation=True,padding=True,max_length=256,return_tensors='pt').to('cuda')
        with torch.inference_mode():out.extend(model(**enc).logits.argmax(-1).tolist())
    return out


def main():
    args=parse_args(); samples=read_jsonl(args.samples_jsonl); manifest=read_json(args.manifest)
    if args.evaluate_all: rows=samples
    elif args.split_manifest: ids=set(read_json(args.split_manifest)['validation_ids']); rows=[r for r in samples if r['id'] in ids]
    else: _,rows=stratified_split(samples,args.validation_fraction,args.seed)
    labels=predict(args,rows); totals=Counter(); correct=Counter(); target=Counter(); target_totals=Counter(); cross=Counter(); unavailable=Counter(); routed_keys=Counter(); available={value.strip() for value in args.available_models.split(',') if value.strip()}; single_correct=Counter(); oracle_any=0
    for row,label in zip(rows,labels):
        modality=row['modality']; key=manifest['route_label_to_model'][str(label)]; correct_models=set(row.get('correct_models',[])); totals[modality]+=1; has_target=int(row.get('label',-1))>=0; target_totals[modality]+=int(has_target); target[modality]+=int(has_target and label==row['label']); expected=modality+'::'; valid=key.startswith(expected); cross[modality]+=int(not valid); raw=key[len(expected):] if valid else ''; is_available=(not available) or raw in available; unavailable[modality]+=int(valid and not is_available); correct[modality]+=int(valid and is_available and raw in correct_models); routed_keys[key]+=1; oracle_any+=int(bool(correct_models));
        for model_name in available: single_correct[model_name]+=int(model_name in correct_models)
    single_accuracies={model_name:single_correct[model_name]/len(rows) for model_name in sorted(available)}; best_single=max(single_accuracies,key=single_accuracies.get) if single_accuracies else None
    summary={'count':len(rows),'target_label_accuracy':sum(target.values())/sum(target_totals.values()) if sum(target_totals.values()) else None,'target_label_count':sum(target_totals.values()),'routed_accuracy':sum(correct.values())/len(rows),'cross_modality_route_count':sum(cross.values()),'unavailable_expert_route_count':sum(unavailable.values()),'oracle_any_expert_accuracy':oracle_any/len(rows),'single_model_accuracies':single_accuracies,'best_single_model':best_single,'best_single_accuracy':single_accuracies.get(best_single) if best_single else None,'by_modality':{m:{'count':totals[m],'target_label_accuracy':target[m]/target_totals[m] if target_totals[m] else None,'target_label_count':target_totals[m],'routed_accuracy':correct[m]/totals[m],'cross_modality_route_count':cross[m],'unavailable_expert_route_count':unavailable[m]} for m in totals},'routed_key_counts':dict(routed_keys)}; args.output_dir.mkdir(parents=True,exist_ok=True); write_json(args.output_dir/'summary.json',summary); print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
