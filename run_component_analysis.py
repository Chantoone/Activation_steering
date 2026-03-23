from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from build_data import build_prompt_records_from_folder
from custom_translation_dataset import CustomTranslationDataset
from translation_perplexity_utils import (
    get_generated_token_perplexity,
    scan_path_patching_by_generated_perplexity,
)
from easy_transformer.EasyTransformer import EasyTransformer


def _ensure_dir(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_easy_transformer(model_name: str, device: Optional[str] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EasyTransformer.from_pretrained(
        model_name,
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        device=device,
    )
    model.eval()

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def _maybe_limit_records(records: List[Dict[str, Any]], max_examples: Optional[int]):
    if max_examples is None:
        return records
    return records[: int(max_examples)]


def _build_dataset(
    folder_path: str,
    prompt_type: str,
    tokenizer,
    max_examples: Optional[int] = None,
):
    records = build_prompt_records_from_folder(
        folder_path=folder_path,
        prompt_type=prompt_type,
        tokenizer=tokenizer,
    )
    records = _maybe_limit_records(records, max_examples=max_examples)
    dataset = CustomTranslationDataset(records, tokenizer=tokenizer)
    return dataset, records


def _dataset_alignment_check(dataset_a, dataset_b, n_check: int = 5):
    if len(dataset_a) != len(dataset_b):
        raise ValueError(
            f"Dataset size mismatch: {len(dataset_a)} vs {len(dataset_b)}"
        )

    n_check = min(n_check, len(dataset_a))
    for i in range(n_check):
        sample_a = dataset_a.preprocess_df_trans[i]
        sample_b = dataset_b.preprocess_df_trans[i]
        if sample_a["src_text"] != sample_b["src_text"]:
            raise ValueError(f"src_text mismatch at index {i}")
        if sample_a["out_token_str"] != sample_b["out_token_str"]:
            raise ValueError(f"out_token_str mismatch at index {i}")


def _top_heads(head_results: torch.Tensor, top_k: int = 10):
    flat_vals, flat_idx = head_results.flatten().topk(min(top_k, head_results.numel()))
    n_heads = head_results.shape[1]
    output = []
    for value, idx in zip(flat_vals.tolist(), flat_idx.tolist()):
        layer = idx // n_heads
        head = idx % n_heads
        output.append(
            {
                "layer": int(layer),
                "head": int(head),
                "delta_pct": float(value),
            }
        )
    return output


def _top_mlps(mlp_results: torch.Tensor, top_k: int = 10):
    flat = mlp_results.flatten()
    flat_vals, flat_idx = flat.topk(min(top_k, flat.numel()))
    output = []
    for value, idx in zip(flat_vals.tolist(), flat_idx.tolist()):
        output.append(
            {
                "layer": int(idx),
                "delta_pct": float(value),
            }
        )
    return output


def _save_results(
    output_dir: Optional[Path],
    prefix: str,
    baseline_ppl: float,
    head_results: torch.Tensor,
    mlp_results: torch.Tensor,
    top_k: int,
):
    if output_dir is None:
        return

    torch.save(head_results, output_dir / f"{prefix}_head_results.pt")
    torch.save(mlp_results, output_dir / f"{prefix}_mlp_results.pt")
    with open(output_dir / f"{prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"baseline_ppl={baseline_ppl}\n")
        f.write(f"head_shape={tuple(head_results.shape)}\n")
        f.write(f"mlp_shape={tuple(mlp_results.shape)}\n")
        f.write(f"top_heads={_top_heads(head_results, top_k=top_k)}\n")
        f.write(f"top_mlps={_top_mlps(mlp_results, top_k=top_k)}\n")


def compare_prompt_styles(
    folder_path: str,
    model_name: str,
    prompt_a: str = "direct",
    prompt_b: str = "multipivot",
    max_examples: Optional[int] = None,
    batch_size: int = 16,
    scan_components: bool = True,
    top_k: int = 10,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
):
    """
    Build two prompt-style datasets from parquet files, compute baseline perplexities,
    and optionally scan important components in both directions.

    Example:
        python run_component_analysis.py compare_prompt_styles ^
            --folder_path test_data ^
            --model_name meta-llama/Llama-2-7b ^
            --prompt_a direct ^
            --prompt_b multipivot
    """
    output_dir_path = _ensure_dir(output_dir)

    model, tokenizer, resolved_device = _load_easy_transformer(
        model_name=model_name,
        device=device,
    )

    dataset_a, records_a = _build_dataset(
        folder_path=folder_path,
        prompt_type=prompt_a,
        tokenizer=tokenizer,
        max_examples=max_examples,
    )
    dataset_b, records_b = _build_dataset(
        folder_path=folder_path,
        prompt_type=prompt_b,
        tokenizer=tokenizer,
        max_examples=max_examples,
    )

    _dataset_alignment_check(dataset_a, dataset_b)

    ppl_a = float(get_generated_token_perplexity(model, dataset_a))
    ppl_b = float(get_generated_token_perplexity(model, dataset_b))

    results: Dict[str, Any] = {
        "model_name": model_name,
        "device": resolved_device,
        "folder_path": folder_path,
        "prompt_a": prompt_a,
        "prompt_b": prompt_b,
        "num_examples": len(dataset_a),
        "ppl_a": ppl_a,
        "ppl_b": ppl_b,
        "records_a_preview": records_a[:1],
        "records_b_preview": records_b[:1],
    }

    if output_dir_path is not None:
        with open(output_dir_path / "baseline_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"model_name={model_name}\n")
            f.write(f"device={resolved_device}\n")
            f.write(f"folder_path={folder_path}\n")
            f.write(f"prompt_a={prompt_a}\n")
            f.write(f"prompt_b={prompt_b}\n")
            f.write(f"num_examples={len(dataset_a)}\n")
            f.write(f"ppl_a={ppl_a}\n")
            f.write(f"ppl_b={ppl_b}\n")

    if not scan_components:
        return results

    receiver_hooks = [(f"blocks.{model.cfg.n_layers - 1}.hook_resid_post", None)]

    baseline_ppl_a, head_results_b_to_a, mlp_results_b_to_a = (
        scan_path_patching_by_generated_perplexity(
            model,
            translation_dataset=dataset_a,
            flipped_translation_dataset=dataset_b,
            receiver_hooks=receiver_hooks,
            batch_size=batch_size,
        )
    )
    _save_results(
        output_dir=output_dir_path,
        prefix=f"{prompt_b}_to_{prompt_a}",
        baseline_ppl=baseline_ppl_a,
        head_results=head_results_b_to_a,
        mlp_results=mlp_results_b_to_a,
        top_k=top_k,
    )

    baseline_ppl_b, head_results_a_to_b, mlp_results_a_to_b = (
        scan_path_patching_by_generated_perplexity(
            model,
            translation_dataset=dataset_b,
            flipped_translation_dataset=dataset_a,
            receiver_hooks=receiver_hooks,
            batch_size=batch_size,
        )
    )
    _save_results(
        output_dir=output_dir_path,
        prefix=f"{prompt_a}_to_{prompt_b}",
        baseline_ppl=baseline_ppl_b,
        head_results=head_results_a_to_b,
        mlp_results=mlp_results_a_to_b,
        top_k=top_k,
    )

    results.update(
        {
            "baseline_ppl_a": float(baseline_ppl_a),
            "baseline_ppl_b": float(baseline_ppl_b),
            "top_heads_b_to_a": _top_heads(head_results_b_to_a, top_k=top_k),
            "top_heads_a_to_b": _top_heads(head_results_a_to_b, top_k=top_k),
            "top_mlps_b_to_a": _top_mlps(mlp_results_b_to_a, top_k=top_k),
            "top_mlps_a_to_b": _top_mlps(mlp_results_a_to_b, top_k=top_k),
        }
    )
    return results


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "compare_prompt_styles": compare_prompt_styles,
        }
    )
