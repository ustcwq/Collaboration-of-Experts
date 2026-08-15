from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT / "outputs/bench_coe/scale_transfer_views_20260802"
BENCH_ROOTS = [
    ROOT / "outputs/model_benchmarks/scale_extension_existing_gap_full_20260731",
    ROOT / "outputs/model_benchmarks/scale_extension_full_20260731",
    ROOT / "outputs/model_benchmarks/family_scale_expansion_full_20260731",
    ROOT / "outputs/model_benchmarks/missing_leaderboard_family_scales_full_20260801",
    ROOT / "outputs/model_benchmarks/autonomous_remaining_full_20260802",
]

TEXT_MODELS = [
    "Qwen3-1.7B", "Qwen2.5-3B-Instruct", "Qwen3-4B-Instruct-2507",
    "granite-3.3-2b-instruct", "internlm2_5-1_8b-chat", "gemma-2-2b-it",
    "Llama-3.2-3B-Instruct", "DeepSeek-R1-Distill-Qwen-1.5B",
    "Ministral-3-3B-Instruct-2512",
    "Qwen3-14B", "Qwen2.5-14B-Instruct", "Baichuan2-13B-Chat",
    "DeepSeek-R1-Distill-Qwen-14B", "Mistral-Nemo-Instruct-2407",
]
VISION_MODELS = [
    "GLM-4.1V-9B-Thinking", "InternVL3_5-8B", "Qwen2.5-VL-7B-Instruct",
    "Qwen3-VL-8B-Thinking", "Llama-3.1-Nemotron-Nano-VL-8B-V1",
    "Phi-4-reasoning-vision-15B",
    "InternVL3_5-14B", "gemma-3-12b-it",
]


def usable(path: Path) -> bool:
    return path.is_dir() and any(path.glob("predictions.json*"))


def usable_gaokao_mm(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*_2010-2023_*.json"))


def replace_link(destination: Path, source: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.symlink_to(source.resolve(), target_is_directory=True)


def first_usable(candidates: list[Path]) -> Path | None:
    return next((path for path in candidates if usable(path)), None)


def materialize() -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    validation_root = ROOT / "outputs/model_benchmarks/improve56_scale_sources_20260801/text/mmlu_validation"
    for model in TEXT_MODELS:
        model_sources: dict[str, str] = {}
        validation = validation_root / model / "CoT/validation"
        if validation.is_dir() and any(validation.glob("*.json")):
            destination = VIEW / "text/mmlu_validation" / model / "CoT/validation"
            replace_link(destination, validation)
            model_sources["mmlu_validation"] = str(validation)

        test_candidates = [root / "text/mmlu_pro_test" / model / "CoT/validation" for root in BENCH_ROOTS]
        test_source = next((path for path in test_candidates if path.is_dir() and any(path.glob("*.json"))), None)
        if test_source:
            destination = VIEW / "text/mmlu_test" / model / "CoT/all"
            replace_link(destination, test_source)
            model_sources["mmlu_test"] = str(test_source)

        for benchmark in ("bbh", "gpqa", "mmstar_text_only"):
            candidates = [root / "text/official" / benchmark / model for root in BENCH_ROOTS]
            candidates.append(ROOT / "outputs/model_benchmarks/official_code_local_models" / benchmark / model)
            source = first_usable(candidates)
            if source:
                destination = VIEW / "text" / benchmark / model
                replace_link(destination, source)
                model_sources[benchmark] = str(source)

        gaokao_candidates = [root / "text/gaokao/gaokao_2010_2022" / model for root in BENCH_ROOTS]
        gaokao_source = first_usable(gaokao_candidates)
        if gaokao_source:
            destination = VIEW / "text/gaokao_2010_2022" / model
            replace_link(destination, gaokao_source)
            model_sources["gaokao_2010_2022"] = str(gaokao_source)
        manifest[model] = model_sources

    canonical = ROOT / "outputs/multimodal_babyvision_models"
    paths = {
        "cmmmu": ("vision/cmmmu/cmmmu/val", "cmmmu/val"),
        "mmmu_pro": ("vision/mmmu_pro/mmmu_pro/standard_10_options/test", "mmmu_pro/standard_10_options/test"),
        "mathvista": ("vision/mathvista/mathvista/testmini", "mathvista/testmini"),
    }
    for model in VISION_MODELS:
        model_sources = manifest.setdefault(model, {})
        for benchmark, (nested, canonical_path) in paths.items():
            candidates = [root / nested / model for root in BENCH_ROOTS]
            candidates.append(canonical / canonical_path / model)
            source = first_usable(candidates)
            if source:
                destination = VIEW / "vision" / canonical_path / model
                replace_link(destination, source)
                model_sources[benchmark] = str(source)

        gaokao_mm_candidates = [root / "vision/gaokao_mm" / model for root in BENCH_ROOTS]
        gaokao_mm_source = next((path for path in gaokao_mm_candidates if usable_gaokao_mm(path)), None)
        if gaokao_mm_source:
            destination = VIEW / "vision/gaokao_mm" / model
            replace_link(destination, gaokao_mm_source)
            model_sources["gaokao_mm"] = str(gaokao_mm_source)

    VIEW.mkdir(parents=True, exist_ok=True)
    (VIEW / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    result = materialize()
    print(json.dumps({model: sorted(sources) for model, sources in result.items()}, ensure_ascii=False, indent=2))
