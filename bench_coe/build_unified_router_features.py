from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description='Build unified language+multimodal router features.'); p.add_argument('--backend',choices=['tinyllava','qwen3vl'],required=True); p.add_argument('--model-path',type=Path,required=True); p.add_argument('--samples-jsonl',type=Path,required=True); p.add_argument('--multimodal-feature-cache',type=Path); p.add_argument('--reuse-feature-cache',type=Path); p.add_argument('--output',type=Path,required=True); p.add_argument('--batch-size',type=int,default=32); return p.parse_args()


def read_jsonl(path:Path)->list[dict[str,Any]]:
    with path.open('r',encoding='utf-8') as f:return [json.loads(line) for line in f if line.strip()]


def mean_hidden(hidden:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    weights=mask.unsqueeze(-1).to(hidden.dtype); return (hidden*weights).sum(1)/weights.sum(1).clamp_min(1)


def load_reused_features(path:Path|None)->dict[str,torch.Tensor]:
    if path is None:return {}
    cache=torch.load(path,map_location='cpu',weights_only=False)
    return {str(sample_id):cache['features'][index] for index,sample_id in enumerate(cache['ids'])}


@torch.inference_mode()
def tiny(args:argparse.Namespace,rows:list[dict[str,Any]])->torch.Tensor:
    tokenizer=AutoTokenizer.from_pretrained(str(args.model_path),trust_remote_code=True,use_fast=False,local_files_only=True); tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(str(args.model_path),trust_remote_code=True,dtype=torch.float16,low_cpu_mem_usage=True,local_files_only=True,attn_implementation='eager').cuda().eval()
    language=[r for r in rows if r['modality']=='language']; language_features=[]
    for start in tqdm(range(0,len(language),args.batch_size),desc='TinyLLaVA language features'):
        batch=language[start:start+args.batch_size]; encoded=tokenizer([r['text'] for r in batch],truncation=True,padding=True,max_length=256,return_tensors='pt').to('cuda')
        out=model.language_model.model(**encoded,use_cache=False,return_dict=True); text=mean_hidden(out.last_hidden_state,encoded['attention_mask']).float().cpu(); language_features.append(torch.cat([torch.zeros_like(text),text],-1))
    lang=torch.cat(language_features); lang_lookup={r['id']:lang[i] for i,r in enumerate(language)}
    mm=torch.load(args.multimodal_feature_cache,map_location='cpu',weights_only=False); mm_lookup={str(v):mm['features'][i] for i,v in enumerate(mm['ids'])}
    return torch.stack([lang_lookup[r['id']] if r['modality']=='language' else mm_lookup[str(r['source_id'])] for r in rows])


@torch.inference_mode()
def qwen(args:argparse.Namespace,rows:list[dict[str,Any]])->torch.Tensor:
    result=load_reused_features(args.reuse_feature_cache); missing=[row for row in rows if row['id'] not in result]
    if not missing:return torch.stack([result[row['id']] for row in rows])
    processor=AutoProcessor.from_pretrained(str(args.model_path),local_files_only=True,trust_remote_code=True); tokenizer=processor.tokenizer; tokenizer.pad_token=tokenizer.pad_token or tokenizer.eos_token
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(args.model_path),local_files_only=True,trust_remote_code=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    language=[r for r in missing if r['modality']=='language']
    for start in tqdm(range(0,len(language),args.batch_size),desc='Qwen3-VL language features'):
        batch=language[start:start+args.batch_size]; encoded=tokenizer([r['text'] for r in batch],truncation=True,padding=True,max_length=256,return_tensors='pt').to('cuda'); out=model.model(**encoded,return_dict=True); pooled=mean_hidden(out.last_hidden_state,encoded['attention_mask']).float().cpu()
        for row,feature in zip(batch,pooled):result[row['id']]=feature
    for row in tqdm([r for r in missing if r['modality']=='multimodal'],desc='Qwen3-VL multimodal features'):
        content=[{'type':'image','image':str(Path(row['image_path']).resolve())},{'type':'text','text':row['text']}]; inputs=processor.apply_chat_template([{'role':'user','content':content}],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda'); out=model.model(**inputs,return_dict=True); result[row['id']]=mean_hidden(out.last_hidden_state,inputs['attention_mask']).float().cpu()[0]
    return torch.stack([result[r['id']] for r in rows])


def main()->None:
    args=parse_args(); rows=read_jsonl(args.samples_jsonl); args.output.parent.mkdir(parents=True,exist_ok=True); features=tiny(args,rows) if args.backend=='tinyllava' else qwen(args,rows); torch.save({'ids':[r['id'] for r in rows],'features':features},args.output); print(features.shape)


if __name__=='__main__':main()
