from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import manifest_sha256, sha256_file
from .features import records_by_question
from .schema import ObservableQueryBatch


class MiniLMResponseEncoder:
    """Deterministic, locally cached response encoder used by answer-aware baselines."""

    def __init__(
        self,
        model_id: str,
        device: Any,
        *,
        batch_size: int = 256,
        max_length: int = 128,
    ) -> None:
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        snapshot = Path(snapshot_download(model_id, local_files_only=True))
        self.model_id = model_id
        self.snapshot = snapshot
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        if self.batch_size < 1 or self.max_length < 8:
            raise ValueError("Response-encoder batch size and max length must be positive")
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.model = AutoModel.from_pretrained(snapshot, local_files_only=True).to(device).eval()
        self.dimension = int(self.model.config.hidden_size)
        self._cache: dict[str, np.ndarray] = {}
        bound_files = [
            path
            for path in snapshot.rglob("*")
            if path.is_file()
            and path.name
            in {
                "config.json",
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
            }
        ]
        self.snapshot_manifest_sha256 = manifest_sha256(
            {str(path.relative_to(snapshot)): sha256_file(path) for path in sorted(bound_files)}
        )

    @staticmethod
    def _bounded_text(text: str) -> str:
        clean = " ".join(str(text or "").split())
        if len(clean) <= 1024:
            return clean
        return f"{clean[:512]} [TRUNCATED_MIDDLE] {clean[-512:]}"

    def _encode_missing(self, texts_by_hash: dict[str, str]) -> None:
        if not texts_by_hash:
            return
        import torch

        keys = sorted(texts_by_hash)
        for start in range(0, len(keys), self.batch_size):
            batch_keys = keys[start : start + self.batch_size]
            encoded = self.tokenizer(
                [texts_by_hash[key] for key in batch_keys],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            values = pooled.detach().cpu().numpy().astype(np.float32)
            for key, value in zip(batch_keys, values, strict=True):
                self._cache[key] = value

    def encode_batch(self, batch: ObservableQueryBatch) -> np.ndarray:
        grouped = records_by_question(batch)
        text_hashes: list[list[str | None]] = []
        missing: dict[str, str] = {}
        for question_id in batch.question_ids:
            by_expert = {record.expert_id: record for record in grouped[question_id]}
            row: list[str | None] = []
            for expert in batch.pool.expert_ids:
                record = by_expert[expert]
                if not record.valid_output:
                    row.append(None)
                    continue
                text = self._bounded_text(record.raw_output or record.normalized_answer or "")
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                row.append(digest)
                if digest not in self._cache:
                    missing[digest] = text
            text_hashes.append(row)
        self._encode_missing(missing)
        result = np.zeros(
            (len(batch.question_ids), len(batch.pool.expert_ids), self.dimension),
            dtype=np.float32,
        )
        for row_index, row in enumerate(text_hashes):
            for col, digest in enumerate(row):
                if digest is not None:
                    result[row_index, col] = self._cache[digest]
        return result

    def diagnostics(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "dimension": self.dimension,
            "pooling": "attention-mask mean pooling followed by L2 normalization",
            "max_length": self.max_length,
            "text_window": "first 512 and last 512 normalized characters",
            "cached_unique_responses": len(self._cache),
        }
