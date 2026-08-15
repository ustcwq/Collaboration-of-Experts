---
license: apache-2.0
pipeline_tag: image-text-to-text
---

**<center><span style="font-size:2em;">TinyLLaVA</span></center>**

[![arXiv](https://img.shields.io/badge/Arxiv-2402.14289-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2402.14289)[![Github](https://img.shields.io/badge/Github-Github-blue.svg)](https://github.com/TinyLLaVA/TinyLLaVA_Factory)[![Demo](https://img.shields.io/badge/Demo-Demo-red.svg)](http://8843843nmph5.vicp.fun/#/)
TinyLLaVA has released a family of small-scale Large Multimodel Models(LMMs), ranging from 1.4B to 3.1B. Our best model, TinyLLaVA-Phi-2-SigLIP-3.1B, achieves better overall performance against existing 7B models such as LLaVA-1.5 and Qwen-VL.

Here, we introduce TinyLLaVA-Phi-2-SigLIP-3.1B, which is trained by the [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) codebase. For LLM and vision tower, we choose [Phi-2](microsoft/phi-2) and [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384), respectively. The dataset used for training this model is the [ShareGPT4V](https://github.com/InternLM/InternLM-XComposer/blob/main/projects/ShareGPT4V/docs/Data.md) dataset.

### Usage
Execute the following test code:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_path = 'tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B'
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
model.cuda()
config = model.config
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False, model_max_length = config.tokenizer_model_max_length,padding_side = config.tokenizer_padding_side)
prompt="What are these?"
image_url="http://images.cocodataset.org/test-stuff2017/000000000001.jpg"
output_text, genertaion_time = model.chat(prompt=prompt, image=image_url, tokenizer=tokenizer)

print('model output:', output_text)
print('runing time:', genertaion_time)
```
### Result
|                          model_name                          | vqav2 | gqa   | sqa | textvqa  | MM-VET | POPE | MME  | MMMU |
| :----------------------------------------------------------: | ----- | ------- | ----- | ----- | ------- | ----- | ------ | ------ |
| [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) |  78.5  | 62.0    |  66.8  |  58.2 | 30.5  | 85.9 | 1510.7   | - |
| [bczhou/TinyLLaVA-3.1B](https://huggingface.co/bczhou/TinyLLaVA-3.1B) (our legacy model) | 79.9  | 62.0    | 69.1  | 59.1 | 32.0  | 86.4  | 1464.9   | - |
| [tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B](https://huggingface.co/tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B) | 78.4 | 61.6   | 64.4 | 53.6 | 26.9 | 86.4  | 1339.0 | 31.7 |
| [tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B](https://huggingface.co/tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B) | 80.1 | 62.1   | 73.0 | 60.3 | 37.5 | 87.2  | 1466.4 | 38.4 |

P.S. [TinyLLaVA Factory](https://github.com/TinyLLaVA/TinyLLaVA_Factory) is an open-source modular codebase for small-scale LMMs with a focus on simplicity of code implementations, extensibility of new features, and reproducibility of training results. This code repository provides standard training&evaluating pipelines, flexible data preprocessing&model configurations, and easily extensible architectures. Users can customize their own LMMs with minimal coding effort and less coding mistake.

TinyLLaVA Factory integrates a suite of cutting-edge models and methods. 
  - LLM currently supports OpenELM, TinyLlama, StableLM, Qwen, Gemma, and Phi. 
  - Vision tower currently supports CLIP, SigLIP, Dino, and combination of CLIP and Dino.
  - Connector currently supports MLP, Qformer, and Resampler.
