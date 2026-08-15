from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bench_coe.assets.paths import AssetPaths
from bench_coe.assets.validation import verify_lock, write_lock_with_hash


class LockingTests(unittest.TestCase):
    def test_external_dataset_root_has_portable_path(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            paths = AssetPaths(base / "assets", base / "cache", base / "repo/data")
            self.assertEqual(paths.relative_to_root(base / "repo/data/arc"), "data/arc")

    def test_tampered_lock_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "asset_lock.json"
            digest = root / "asset_lock.sha256"
            write_lock_with_hash(lock, {"resources": []}, digest)
            self.assertEqual(verify_lock(lock, digest)["status"], "valid")
            lock.write_text('{"resources":[{"tampered":true}]}')
            self.assertEqual(verify_lock(lock, digest)["error"], "lock_hash_mismatch")

    def test_local_model_weights_load_offline(self):
        from transformers import BertConfig, BertModel, BertTokenizer
        from tools.verify_assets import offline_load

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab = root / "vocab.txt"
            vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nready\n", encoding="utf-8")
            BertTokenizer(vocab_file=str(vocab)).save_pretrained(root)
            BertModel(BertConfig(vocab_size=6, hidden_size=8, num_hidden_layers=1, num_attention_heads=1, intermediate_size=16)).save_pretrained(root)
            result = offline_load(root, trust_remote_code=False)
            self.assertEqual(result["status"], "loaded_offline")
            self.assertTrue({"model_weights", "causal_lm_weights"} & set(result["components"]))


if __name__ == "__main__":
    unittest.main()
