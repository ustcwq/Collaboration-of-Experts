from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bench_coe.evaluate_tinyllava_subject_router import format_text
from bench_coe.gaokao_utils import write_json


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser(description="Combine language and multimodal direct-expert query labels.")
    parser.add_argument("--language-samples",type=Path,required=True); parser.add_argument("--multimodal-labels",type=Path,required=True)
    parser.add_argument("--multimodal-reference",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    return parser.parse_args()


def read_jsonl(path:Path)->list[dict[str,Any]]:
    with path.open('r',encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]


def main()->None:
    args=parse_args(); language=read_jsonl(args.language_samples); mm_labels=read_jsonl(args.multimodal_labels); mm_ref={str(r['id']):r for r in read_jsonl(args.multimodal_reference)}
    expert_keys=sorted({f"language::{r['target_model']}" for r in language}|{f"multimodal::{r['target_model']}" for r in mm_labels}); key_to_label={key:i for i,key in enumerate(expert_keys)}
    rows=[]
    for row in language:
        key=f"language::{row['target_model']}"; rows.append({'id':f"language::{row['id']}",'modality':'language','text':'[LANGUAGE]\n'+row['text'],'target_model':key,'correct_models':row.get('correct_models',[]),'label':key_to_label[key]})
    for label_row in mm_labels:
        ref=mm_ref[str(label_row['id'])]; key=f"multimodal::{label_row['target_model']}"; rows.append({'id':f"multimodal::{label_row['id']}",'modality':'multimodal','source_id':str(label_row['id']),'image_path':ref['image_path'],'text':'[MULTIMODAL]\n'+format_text(ref),'target_model':key,'correct_models':label_row.get('correct_models',[]),'label':key_to_label[key]})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with (args.output_dir/'query_router_samples.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_json(args.output_dir/'route_label_manifest.json',{'label_mode':'unified_modality_expert','num_route_labels':len(expert_keys),'model_names':expert_keys,'model_to_route_label':key_to_label,'route_label_to_model':{str(v):k for k,v in key_to_label.items()},'language_source':'MMLU-Pro test-source labels','multimodal_source':'MMMU validation internal train split'})
    write_json(args.output_dir/'statistics.json',{'total':len(rows),'language':len(language),'multimodal':len(mm_labels),'num_labels':len(expert_keys)})
    print(json.dumps({'total':len(rows),'language':len(language),'multimodal':len(mm_labels),'num_labels':len(expert_keys)},ensure_ascii=False))


if __name__=='__main__':main()
