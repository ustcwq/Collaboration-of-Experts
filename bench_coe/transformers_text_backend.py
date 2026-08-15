from __future__ import annotations

from dataclasses import dataclass
import os
from types import SimpleNamespace
from typing import Any


@dataclass
class TransformersSamplingParams:
    max_tokens: int
    temperature: float = 0.0
    stop: tuple[str, ...] = ()


class TransformersTextLLM:
    def __init__(
        self,
        model_path: str,
        gpu_id: str | None,
        max_model_len: int,
        attn_implementation: str = "eager",
    ) -> None:
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        self.max_model_len = max_model_len
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "fix_mistral_regex": True,
        }
        self.processor = AutoProcessor.from_pretrained(model_path, **tokenizer_kwargs)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.batch_size = max(1, int(os.environ.get("BENCH_COE_TRANSFORMERS_BATCH_SIZE", "8")))

        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "dtype": torch.bfloat16,
            "attn_implementation": attn_implementation,
            "low_cpu_mem_usage": True,
        }
        errors: list[str] = []
        self.model = None
        for model_class in (AutoModelForImageTextToText, AutoModelForCausalLM):
            try:
                self.model = model_class.from_pretrained(model_path, **load_kwargs).cuda().eval()
                break
            except Exception as exc:
                errors.append(f"{model_class.__name__}: {type(exc).__name__}: {exc}")
        if self.model is None:
            raise RuntimeError("Unable to load Transformers text model: " + " | ".join(errors))

    def get_tokenizer(self) -> Any:
        return self.tokenizer

    def generate(self, prompts: list[str], sampling: TransformersSamplingParams) -> list[Any]:
        import torch

        results: list[Any] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[start : start + self.batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max(1, self.max_model_len - sampling.max_tokens),
            )
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            input_length = int(inputs["input_ids"].shape[1])
            stop_kwargs = {}
            if sampling.stop:
                stop_kwargs = {"stop_strings": list(sampling.stop), "tokenizer": self.tokenizer}
            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    do_sample=sampling.temperature > 0,
                    temperature=max(sampling.temperature, 1e-5),
                    max_new_tokens=sampling.max_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **stop_kwargs,
                )
            texts = self.tokenizer.batch_decode(generated[:, input_length:], skip_special_tokens=True)
            results.extend(SimpleNamespace(outputs=[SimpleNamespace(text=text)]) for text in texts)
        return results
