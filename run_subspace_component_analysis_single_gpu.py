from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import gc

import torch

from build_data import build_prompt_records_from_folder
from custom_translation_dataset import CustomTranslationDataset
from easy_transformer.EasyTransformer import EasyTransformer
from easy_transformer.experiments import get_act_hook
from translation_perplexity_utils import (
    get_generated_prediction_positions,
    get_generated_token_perplexity,
    get_generated_token_perplexity_from_logits,
)
from translation_utils import patch_all


Component = Tuple[str, Optional[int]]


def _ensure_dir(path: Optional[str]) -> Optional[Path]:
    if path is None:
        return None
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_single_device(device: Optional[str]) -> str:
    if device is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if "," in device:
        return device.split(",")[0].strip()
    return device


def _resolve_dtype(dtype: str, device: str):
    normalized = dtype.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "auto":
        if device.startswith("cuda"):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def _clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_device_total_memory_gb(device: str) -> float:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return 0.0
    try:
        index = torch.device(device).index
        if index is None:
            index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
    except Exception:
        return 0.0
    return float(props.total_memory) / (1024**3)


def _prefer_gpu_cache(device: str) -> bool:
    return device.startswith("cuda") and _get_device_total_memory_gb(device) >= 40


def _auto_batch_sizes(
    device: str,
    batch_size: int,
    identification_batch_size: int,
) -> Tuple[int, int]:
    effective_batch_size = int(batch_size)
    effective_identification_batch_size = int(identification_batch_size)

    if (
        device.startswith("cuda")
        and effective_batch_size == 1
        and effective_identification_batch_size == 1
    ):
        total_memory_gb = _get_device_total_memory_gb(device)
        if total_memory_gb >= 40:
            return 16, 32
        if total_memory_gb >= 24:
            return 8, 16
        if total_memory_gb >= 16:
            return 4, 8

    return effective_batch_size, effective_identification_batch_size


def _is_oom_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _load_easy_transformer(
    model_name: str,
    device: Optional[str] = None,
    dtype: str = "auto",
):
    resolved_device = _resolve_single_device(device)
    resolved_dtype = _resolve_dtype(dtype, resolved_device)

    model = EasyTransformer.from_pretrained(
        model_name,
        fold_ln=True,
        center_writing_weights=False,
        center_unembed=False,
        device=resolved_device,
    )
    model.eval()

    if resolved_device.startswith("cuda") and resolved_dtype != torch.float32:
        if resolved_dtype == torch.float16:
            model.half()
        elif resolved_dtype == torch.bfloat16:
            model.bfloat16()

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, resolved_device, str(resolved_dtype).replace("torch.", "")


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


def _normalize_source_file(source_file: Optional[str]) -> str:
    if source_file is None:
        return "unknown_source"
    return str(source_file)


def _safe_output_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)


def _all_components(model) -> List[Component]:
    components: List[Component] = []
    for layer in range(model.cfg.n_layers):
        components.append((f"blocks.{layer}.hook_mlp_out", None))
        for head_idx in range(model.cfg.n_heads):
            components.append((f"blocks.{layer}.attn.hook_z", head_idx))
    return components


def _component_to_tensor_maps(model, deltas: Dict[Component, float]):
    head_results = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    mlp_results = torch.zeros((model.cfg.n_layers, 1))

    for (hook_name, head_idx), delta in deltas.items():
        layer = int(hook_name.split(".")[1])
        if head_idx is None:
            mlp_results[layer, 0] = float(delta)
        else:
            head_results[layer, head_idx] = float(delta)

    return head_results, mlp_results


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
    flat_vals, flat_idx = mlp_results.flatten().topk(min(top_k, mlp_results.numel()))
    output = []
    for value, idx in zip(flat_vals.tolist(), flat_idx.tolist()):
        output.append(
            {
                "layer": int(idx),
                "delta_pct": float(value),
            }
        )
    return output


def _format_top_heads(top_heads: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for rank_idx, item in enumerate(top_heads, start=1):
        lines.append(
            f"{rank_idx}. layer={item['layer']} head={item['head']} delta_pct={item['delta_pct']:.6f}"
        )
    return lines


def _format_top_mlps(top_mlps: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for rank_idx, item in enumerate(top_mlps, start=1):
        lines.append(
            f"{rank_idx}. layer={item['layer']} delta_pct={item['delta_pct']:.6f}"
        )
    return lines


def _write_ranked_section(f, title: str, lines: List[str]):
    f.write(f"{title}\n")
    if not lines:
        f.write("none\n")
        return
    for line in lines:
        f.write(f"{line}\n")


def _write_baseline_summary_stub(
    output_path: Path,
    *,
    dtype: str,
    prompt_orig: str,
    prompt_new: str,
    num_examples: int,
    rank: int,
):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("method=subspace_intervened_path_patching\n")
        f.write("metric=generated_span_perplexity\n")
        f.write("implementation=single_gpu_memory_optimized\n")
        f.write("attention_component=hook_z\n")
        f.write(f"dtype={dtype}\n")
        f.write(f"prompt_orig={prompt_orig}\n")
        f.write(f"prompt_new={prompt_new}\n")
        f.write(f"num_examples={num_examples}\n")
        f.write(f"subspace_rank={rank}\n")


def _write_legacy_overall_summary(
    output_path: Path,
    *,
    model_name: str,
    resolved_device: str,
    folder_path: str,
    prompt_orig: str,
    prompt_new: str,
    overall_results: Dict[str, Any],
    per_source_file_results: Dict[str, Any],
):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"model_name={model_name}\n")
        f.write(f"device={resolved_device}\n")
        f.write(f"folder_path={folder_path}\n")
        f.write(f"prompt_a={prompt_new}\n")
        f.write(f"prompt_b={prompt_orig}\n")
        f.write(f"num_examples={overall_results['num_examples']}\n")
        f.write(f"baseline_ppl={overall_results['baseline_ppl']}\n")
        f.write(f"num_source_files={len(per_source_file_results)}\n")


def _write_overall_summary(
    output_path: Path,
    *,
    model_name: str,
    resolved_device: str,
    resolved_dtype: str,
    folder_path: str,
    prompt_orig: str,
    prompt_new: str,
    rank: int,
    overall_results: Dict[str, Any],
    per_source_file_results: Dict[str, Any],
):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("method=subspace_intervened_path_patching\n")
        f.write("metric=generated_span_perplexity\n")
        f.write("implementation=single_gpu_memory_optimized\n")
        f.write("attention_component=hook_z\n")
        f.write(f"model_name={model_name}\n")
        f.write(f"device={resolved_device}\n")
        f.write(f"dtype={resolved_dtype}\n")
        f.write(f"folder_path={folder_path}\n")
        f.write(f"prompt_orig={prompt_orig}\n")
        f.write(f"prompt_new={prompt_new}\n")
        f.write(f"num_examples={overall_results['num_examples']}\n")
        f.write(f"subspace_rank={rank}\n")
        f.write(f"baseline_ppl={overall_results['baseline_ppl']}\n")
        f.write(f"num_source_files={len(per_source_file_results)}\n")
        f.write("\n")

        _write_ranked_section(
            f,
            "important_attention_heads_overall:",
            _format_top_heads(overall_results.get("top_heads", [])),
        )
        f.write("\n")
        _write_ranked_section(
            f,
            "important_mlp_layers_overall:",
            _format_top_mlps(overall_results.get("top_mlps", [])),
        )
        f.write("\n")

        f.write("important_components_by_source_file:\n")
        if not per_source_file_results:
            f.write("none\n")
            return

        for source_file, source_results in sorted(per_source_file_results.items()):
            f.write(f"source_file={source_file}\n")
            f.write(f"num_examples={source_results['num_examples']}\n")
            f.write(f"baseline_ppl={source_results['baseline_ppl']}\n")
            _write_ranked_section(
                f,
                "top_attention_heads:",
                _format_top_heads(source_results.get("top_heads", [])),
            )
            _write_ranked_section(
                f,
                "top_mlp_layers:",
                _format_top_mlps(source_results.get("top_mlps", [])),
            )
            f.write("\n")


def _save_results(
    output_dir: Optional[Path],
    prefix: str,
    baseline_ppl: float,
    head_results: torch.Tensor,
    mlp_results: torch.Tensor,
    rank: int,
    top_k: int,
    dtype: str,
):
    if output_dir is None:
        return

    torch.save(head_results, output_dir / f"{prefix}_head_results.pt")
    torch.save(mlp_results, output_dir / f"{prefix}_mlp_results.pt")
    with open(output_dir / f"{prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write("method=subspace_intervened_path_patching\n")
        f.write("metric=generated_span_perplexity\n")
        f.write("implementation=single_gpu_memory_optimized\n")
        f.write("attention_component=hook_z\n")
        f.write(f"dtype={dtype}\n")
        f.write(f"subspace_rank={rank}\n")
        f.write(f"baseline_ppl={baseline_ppl}\n")
        f.write(f"head_shape={tuple(head_results.shape)}\n")
        f.write(f"mlp_shape={tuple(mlp_results.shape)}\n")
        f.write(f"top_heads={_top_heads(head_results, top_k=top_k)}\n")
        f.write(f"top_mlps={_top_mlps(mlp_results, top_k=top_k)}\n")


def _extract_component_sequence(
    component: Component,
    activations: Dict[str, torch.Tensor],
    batch_idx: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    hook_name, head_idx = component
    act = activations[hook_name]
    if act.dim() == 4:
        if head_idx is None:
            raise ValueError(f"Head index must be provided for {hook_name}")
        return act[batch_idx, positions, head_idx]
    if act.dim() == 3:
        return act[batch_idx, positions]
    raise ValueError(f"Unsupported activation shape {act.shape} for {hook_name}")


def _set_component_sequence(
    component: Component,
    z: torch.Tensor,
    batch_idx: int,
    positions: torch.Tensor,
    patched_value: torch.Tensor,
) -> torch.Tensor:
    _, head_idx = component
    if z.dim() == 4:
        if head_idx is None:
            raise ValueError("Head index must be provided for 4D activations")
        z[batch_idx, positions, head_idx] = patched_value.to(z.device, dtype=z.dtype)
    elif z.dim() == 3:
        z[batch_idx, positions] = patched_value.to(z.device, dtype=z.dtype)
    else:
        raise ValueError(f"Unsupported activation shape {z.shape}")
    return z


def _shared_generated_positions(
    orig_positions: Sequence[torch.Tensor],
    new_positions: Sequence[torch.Tensor],
    orig_seq_len: int,
    new_seq_len: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    shared_positions: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for orig_pos, new_pos in zip(orig_positions, new_positions):
        orig_trimmed = orig_pos[orig_pos < orig_seq_len]
        new_trimmed = new_pos[new_pos < new_seq_len]
        shared_len = min(orig_trimmed.numel(), new_trimmed.numel())
        shared_positions.append((orig_trimmed[:shared_len], new_trimmed[:shared_len]))
    return shared_positions


def _project_patch(
    a_plus: torch.Tensor,
    a_minus: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    compute_dtype = a_plus.dtype
    if a_plus.device.type == "cpu" and a_plus.dtype in {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float32

    a_plus_compute = a_plus.to(dtype=compute_dtype)
    a_minus_compute = a_minus.to(dtype=compute_dtype)
    W_compute = W.to(device=a_plus.device, dtype=compute_dtype)

    W_orth, _ = torch.linalg.qr(W_compute)
    proj = W_orth @ W_orth.T
    identity = torch.eye(proj.shape[0], device=proj.device, dtype=proj.dtype)
    patched = torch.matmul(a_minus_compute, proj.T) + torch.matmul(
        a_plus_compute, (identity - proj).T
    )
    return patched.to(device=a_plus.device, dtype=a_plus.dtype)


def _leading_subspace_from_deltas(
    deltas: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    compute_dtype = deltas.dtype
    if compute_dtype in {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float32

    Mc = deltas.T.to(dtype=compute_dtype)
    ones = torch.ones(Mc.shape[1], device=Mc.device, dtype=Mc.dtype)
    Sc_prime = (Mc @ ones) / Mc.shape[1]
    residual = Mc - Sc_prime.unsqueeze(1) @ ones.unsqueeze(0)
    U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
    r = min(rank, U.shape[1])
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r]
    Mc_prime = Sc_prime.unsqueeze(1) + (U_r * S_r) @ Vh_r
    leading = (Mc_prime @ ones) / Mc_prime.shape[1]
    leading = leading / (leading.norm() + 1e-12)
    return leading.unsqueeze(1)


def _forward_logits(model, dataset):
    model_device = _get_model_device(model)
    toks = dataset.toks.long().to(model_device)
    with torch.no_grad():
        logits = model(toks)
    return logits


def _collect_delta_rows_for_component(
    model,
    dataset_pos,
    dataset_neg,
    component: Component,
    batch_size: int,
) -> torch.Tensor:
    if len(dataset_pos) != len(dataset_neg):
        raise ValueError("Contrastive datasets must align")

    hook_name, _ = component
    delta_rows: List[torch.Tensor] = []
    model_device = _get_model_device(model)
    cache_device = str(model_device) if model_device.type == "cuda" else "cpu"

    for start in range(0, len(dataset_pos), batch_size):
        end = start + batch_size
        pos_batch = dataset_pos[start:end]
        neg_batch = dataset_neg[start:end]

        pos_cache: Dict[str, torch.Tensor] = {}
        model.reset_hooks()
        model.cache_some(
            pos_cache,
            lambda name: name == hook_name,
            device=cache_device,
            suppress_warning=True,
        )
        with torch.no_grad():
            model(pos_batch.toks.long().to(model_device))
        model.reset_hooks()

        neg_cache: Dict[str, torch.Tensor] = {}
        model.cache_some(
            neg_cache,
            lambda name: name == hook_name,
            device=cache_device,
            suppress_warning=True,
        )
        with torch.no_grad():
            model(neg_batch.toks.long().to(model_device))
        model.reset_hooks()

        pos_positions = get_generated_prediction_positions(pos_batch)
        neg_positions = get_generated_prediction_positions(neg_batch)
        shared_positions = _shared_generated_positions(
            pos_positions,
            neg_positions,
            orig_seq_len=pos_cache[hook_name].shape[1],
            new_seq_len=neg_cache[hook_name].shape[1],
        )

        for batch_idx, (pos_idx, neg_idx) in enumerate(shared_positions):
            if pos_idx.numel() == 0:
                continue
            a_pos = _extract_component_sequence(component, pos_cache, batch_idx, pos_idx)
            a_neg = _extract_component_sequence(component, neg_cache, batch_idx, neg_idx)
            delta_rows.append((a_pos - a_neg).float().cpu())

        del pos_cache
        del neg_cache
        _clear_memory()

    if not delta_rows:
        raise ValueError(f"No shared generated positions found for component {component}")

    return torch.cat(delta_rows, dim=0)


def _identify_subspaces_for_hook(
    model,
    dataset_pos,
    dataset_neg,
    hook_name: str,
    rank: int,
    identification_batch_size: int,
) -> Dict[Component, torch.Tensor]:
    if len(dataset_pos) != len(dataset_neg):
        raise ValueError("Contrastive datasets must align")

    model_device = _get_model_device(model)
    cache_device = str(model_device) if _prefer_gpu_cache(str(model_device)) else "cpu"
    grouped_deltas: List[torch.Tensor] = []

    for start in range(0, len(dataset_pos), identification_batch_size):
        end = start + identification_batch_size
        pos_batch = dataset_pos[start:end]
        neg_batch = dataset_neg[start:end]

        pos_cache: Dict[str, torch.Tensor] = {}
        model.reset_hooks()
        model.cache_some(
            pos_cache,
            lambda name: name == hook_name,
            device=cache_device,
            suppress_warning=True,
        )
        with torch.no_grad():
            model(pos_batch.toks.long().to(model_device))
        model.reset_hooks()

        neg_cache: Dict[str, torch.Tensor] = {}
        model.cache_some(
            neg_cache,
            lambda name: name == hook_name,
            device=cache_device,
            suppress_warning=True,
        )
        with torch.no_grad():
            model(neg_batch.toks.long().to(model_device))
        model.reset_hooks()

        pos_positions = get_generated_prediction_positions(pos_batch)
        neg_positions = get_generated_prediction_positions(neg_batch)
        shared_positions = _shared_generated_positions(
            pos_positions,
            neg_positions,
            orig_seq_len=pos_cache[hook_name].shape[1],
            new_seq_len=neg_cache[hook_name].shape[1],
        )

        for batch_idx, (pos_idx, neg_idx) in enumerate(shared_positions):
            if pos_idx.numel() == 0:
                continue
            pos_idx = pos_idx.to(pos_cache[hook_name].device)
            neg_idx = neg_idx.to(neg_cache[hook_name].device)
            grouped_deltas.append(
                (
                    pos_cache[hook_name][batch_idx, pos_idx]
                    - neg_cache[hook_name][batch_idx, neg_idx]
                ).float()
            )

        del pos_cache
        del neg_cache
        _clear_memory()

    if not grouped_deltas:
        raise ValueError(f"No shared generated positions found for hook {hook_name}")

    stacked = torch.cat(grouped_deltas, dim=0)
    result: Dict[Component, torch.Tensor] = {}
    layer = int(hook_name.split(".")[1])

    if stacked.dim() == 2:
        result[(hook_name, None)] = _leading_subspace_from_deltas(stacked, rank=rank)
        return result

    if stacked.dim() != 3:
        raise ValueError(f"Unsupported activation shape for hook {hook_name}: {stacked.shape}")

    for head_idx in range(stacked.shape[1]):
        result[(hook_name, head_idx)] = _leading_subspace_from_deltas(
            stacked[:, head_idx, :],
            rank=rank,
        )
    return result


def task_steering_subspace_identification_generated_single_gpu(
    model,
    dataset_pos,
    dataset_neg,
    component: Component,
    rank: int = 1,
    identification_batch_size: int = 1,
):
    deltas = _collect_delta_rows_for_component(
        model=model,
        dataset_pos=dataset_pos,
        dataset_neg=dataset_neg,
        component=component,
        batch_size=identification_batch_size,
    )
    return _leading_subspace_from_deltas(deltas, rank=rank)


def _build_master_patching_caches(
    model,
    dataset_orig,
    dataset_new,
    freeze_mlps: bool,
):
    model_device = _get_model_device(model)
    cache_device = str(model_device) if _prefer_gpu_cache(str(model_device)) else "cpu"

    sender_hook_names = {
        f"blocks.{layer}.hook_mlp_out"
        for layer in range(model.cfg.n_layers)
    }
    sender_hook_names.update(
        f"blocks.{layer}.attn.hook_z"
        for layer in range(model.cfg.n_layers)
    )

    target_hook_names = set(sender_hook_names)
    for layer in range(model.cfg.n_layers):
        for hook_template in [
            "blocks.{}.attn.hook_q",
            "blocks.{}.attn.hook_k",
            "blocks.{}.attn.hook_v",
        ]:
            target_hook_names.add(hook_template.format(layer))
        if freeze_mlps:
            target_hook_names.add(f"blocks.{layer}.hook_mlp_out")

    sender_cache: Dict[str, torch.Tensor] = {}
    model.reset_hooks()
    model.cache_some(
        sender_cache,
        lambda name: name in sender_hook_names,
        device=cache_device,
        suppress_warning=True,
    )
    with torch.no_grad():
        model(dataset_new.toks.long().to(model_device))
    model.reset_hooks()

    target_cache: Dict[str, torch.Tensor] = {}
    model.cache_some(
        target_cache,
        lambda name: name in target_hook_names,
        device=cache_device,
        suppress_warning=True,
    )
    with torch.no_grad():
        model(dataset_orig.toks.long().to(model_device))
    model.reset_hooks()

    return sender_cache, target_cache


def _subspace_delta_for_component_batch(
    model,
    dataset_orig,
    dataset_new,
    component: Component,
    subspace: torch.Tensor,
    y_orig: torch.Tensor,
    receiver_hooks: List[Component],
    freeze_mlps: bool,
    have_internal_interactions: bool,
    epsilon: float = 1e-8,
    sender_cache: Optional[Dict[str, torch.Tensor]] = None,
    target_cache: Optional[Dict[str, torch.Tensor]] = None,
    orig_positions: Optional[Sequence[torch.Tensor]] = None,
    new_positions: Optional[Sequence[torch.Tensor]] = None,
):
    hook_name, _ = component
    receiver_hook_names = [x[0] for x in receiver_hooks]
    model_device = _get_model_device(model)

    if orig_positions is None:
        orig_positions = get_generated_prediction_positions(dataset_orig)
    if new_positions is None:
        new_positions = get_generated_prediction_positions(dataset_new)

    owns_sender_cache = sender_cache is None
    owns_target_cache = target_cache is None

    if sender_cache is None:
        sender_cache = {}
        cache_device = "cpu"
        model.reset_hooks()
        model.cache_some(
            sender_cache,
            lambda name: name == hook_name,
            device=cache_device,
            suppress_warning=True,
        )
        with torch.no_grad():
            model(dataset_new.toks.long().to(model_device))
        model.reset_hooks()

    if target_cache is None:
        target_hook_names = {hook_name}
        for layer in range(model.cfg.n_layers):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                target_hook_names.add(hook_template.format(layer))
            if freeze_mlps:
                target_hook_names.add(f"blocks.{layer}.hook_mlp_out")

        target_cache = {}
        model.cache_some(
            target_cache,
            lambda name: name in target_hook_names,
            device="cpu",
            suppress_warning=True,
        )
        with torch.no_grad():
            model(dataset_orig.toks.long().to(model_device))
        model.reset_hooks()

    shared_positions = _shared_generated_positions(
        orig_positions,
        new_positions,
        orig_seq_len=target_cache[hook_name].shape[1],
        new_seq_len=sender_cache[hook_name].shape[1],
    )

    def subspace_patch(
        z,
        hook,
        comp=component,
        W=subspace,
        src=sender_cache[hook_name],
        tgt=target_cache[hook_name],
        batch_shared_positions=shared_positions,
    ):
        for batch_idx, (orig_idx, new_idx) in enumerate(batch_shared_positions):
            if orig_idx.numel() == 0:
                continue
            orig_idx_z = orig_idx.to(z.device)
            orig_idx_tgt = orig_idx.to(tgt.device)
            new_idx_src = new_idx.to(src.device)
            a_plus = _extract_component_sequence(comp, {hook_name: tgt}, batch_idx, orig_idx_tgt)
            a_minus = _extract_component_sequence(comp, {hook_name: src}, batch_idx, new_idx_src)
            patched = _project_patch(a_plus, a_minus, W)
            _set_component_sequence(comp, z, batch_idx, orig_idx_z, patched)
        return z

    freezer_hooks = []
    for layer in range(model.cfg.n_layers):
        for h_idx in range(model.cfg.n_heads):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                h_name = hook_template.format(layer)
                if have_internal_interactions and h_name in receiver_hook_names:
                    continue
                freeze_hook = get_act_hook(
                    patch_all,
                    alt_act=target_cache[h_name],
                    idx=h_idx,
                    dim=2,
                    name=h_name,
                )
                freezer_hooks.append((h_name, freeze_hook))

        if freeze_mlps:
            h_name = f"blocks.{layer}.hook_mlp_out"
            freeze_hook = get_act_hook(
                patch_all,
                alt_act=target_cache[h_name],
                idx=None,
                dim=None,
                name=h_name,
            )
            freezer_hooks.append((h_name, freeze_hook))

    receiver_cache: Dict[str, torch.Tensor] = {}
    model.reset_hooks()
    for h_name, hk in freezer_hooks:
        model.add_hook(h_name, hk)
    model.add_hook(hook_name, subspace_patch)
    model.cache_some(
        receiver_cache,
        lambda n: n in receiver_hook_names,
        device="cpu",
        suppress_warning=True,
    )
    with torch.no_grad():
        model(dataset_orig.toks.long().to(model_device))

    model.reset_hooks()
    for h_name, hk in freezer_hooks:
        model.add_hook(h_name, hk)
    model.add_hook(hook_name, subspace_patch)
    for r_name, r_head in receiver_hooks:
        cached_act = receiver_cache[r_name]

        def restore_generated_positions(
            z,
            hook,
            alt_act=cached_act,
            comp=(r_name, r_head),
            batch_positions=orig_positions,
        ):
            for batch_idx, pos in enumerate(batch_positions):
                if pos.numel() == 0:
                    continue
                pos_z = pos[pos < z.shape[1]].to(z.device)
                pos_alt = pos[pos < alt_act.shape[1]].to(alt_act.device)
                shared_len = min(pos_z.numel(), pos_alt.numel())
                if shared_len == 0:
                    continue
                patched_val = _extract_component_sequence(
                    comp,
                    {r_name: alt_act},
                    batch_idx,
                    pos_alt[:shared_len],
                )
                _set_component_sequence(comp, z, batch_idx, pos_z[:shared_len], patched_val)
            return z

        model.add_hook(r_name, restore_generated_positions)

    with torch.no_grad():
        logits_new = model(dataset_orig.toks.long().to(model_device))

    delta = (
        get_generated_token_perplexity_from_logits(logits_new, dataset_orig) - y_orig
    ) / (y_orig + epsilon)
    model.reset_hooks()

    if owns_sender_cache:
        del sender_cache
    if owns_target_cache:
        del target_cache
    del receiver_cache
    del logits_new
    if owns_sender_cache or owns_target_cache:
        _clear_memory()
    return float(delta.mean().detach().cpu())


def _scan_components_with_subspace(
    model,
    dataset_orig,
    dataset_new,
    rank: int,
    batch_size: int,
    identification_batch_size: int,
    receiver_hooks: List[Component],
    freeze_mlps: bool,
    have_internal_interactions: bool,
):
    components = _all_components(model)
    model_device = _get_model_device(model)
    baseline_values = get_generated_token_perplexity(
        model,
        dataset_orig,
        all=True,
        batch_size=batch_size,
    )
    baseline_ppl = float(baseline_values.mean().item())

    subspace_dict: Dict[Component, torch.Tensor] = {}
    hook_names = sorted({component[0] for component in components})
    for hook_name in hook_names:
        hook_subspaces = _identify_subspaces_for_hook(
            model=model,
            dataset_pos=dataset_orig,
            dataset_neg=dataset_new,
            hook_name=hook_name,
            rank=rank,
            identification_batch_size=identification_batch_size,
        )
        for component, subspace in hook_subspaces.items():
            subspace_dict[component] = subspace.to(model_device)
        _clear_memory()

    totals: Dict[Component, float] = {component: 0.0 for component in components}
    counts = 0
    use_master_caches = _prefer_gpu_cache(str(model_device))

    for start in range(0, len(dataset_orig), batch_size):
        end = start + batch_size
        orig_batch = dataset_orig[start:end]
        new_batch = dataset_new[start:end]

        logits_orig = _forward_logits(model, orig_batch)
        y_orig = get_generated_token_perplexity_from_logits(
            logits_orig,
            orig_batch,
        ).detach()
        del logits_orig

        orig_positions = get_generated_prediction_positions(orig_batch)
        new_positions = get_generated_prediction_positions(new_batch)

        sender_cache = None
        target_cache = None
        if use_master_caches:
            try:
                sender_cache, target_cache = _build_master_patching_caches(
                    model=model,
                    dataset_orig=orig_batch,
                    dataset_new=new_batch,
                    freeze_mlps=freeze_mlps,
                )
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                sender_cache = None
                target_cache = None
                use_master_caches = False
                _clear_memory()

        for component in components:
            delta = _subspace_delta_for_component_batch(
                model=model,
                dataset_orig=orig_batch,
                dataset_new=new_batch,
                component=component,
                subspace=subspace_dict[component],
                y_orig=y_orig,
                receiver_hooks=receiver_hooks,
                freeze_mlps=freeze_mlps,
                have_internal_interactions=have_internal_interactions,
                sender_cache=sender_cache,
                target_cache=target_cache,
                orig_positions=orig_positions,
                new_positions=new_positions,
            )
            totals[component] += float(delta) * len(orig_batch)

        if sender_cache is not None:
            del sender_cache
        if target_cache is not None:
            del target_cache
        counts += len(orig_batch)
        del y_orig
        _clear_memory()

    deltas = {
        component: totals[component] / max(counts, 1)
        for component in components
    }

    delta_pct = {
        component: 100.0 * float(delta)
        for component, delta in deltas.items()
    }
    head_results, mlp_results = _component_to_tensor_maps(model, delta_pct)
    return baseline_ppl, head_results, mlp_results


def _analyze_dataset_pair(
    model,
    dataset_orig,
    dataset_new,
    records_orig,
    records_new,
    prompt_orig: str,
    prompt_new: str,
    batch_size: int,
    identification_batch_size: int,
    rank: int,
    top_k: int,
    dtype: str,
    output_dir_path: Optional[Path] = None,
    output_prefix_suffix: str = "",
    freeze_mlps: bool = False,
    have_internal_interactions: bool = False,
):
    _dataset_alignment_check(dataset_orig, dataset_new)

    results: Dict[str, Any] = {
        "num_examples": len(dataset_orig),
        "records_orig_preview": records_orig[:1],
        "records_new_preview": records_new[:1],
        "method": "subspace_intervened_path_patching",
        "metric": "generated_span_perplexity",
        "subspace_rank": rank,
        "implementation": "single_gpu_memory_optimized",
        "attention_component": "hook_z",
        "dtype": dtype,
    }

    summary_prefix = f"subspace_baseline_summary{output_prefix_suffix}"
    if output_dir_path is not None:
        _write_baseline_summary_stub(
            output_dir_path / f"{summary_prefix}.txt",
            dtype=dtype,
            prompt_orig=prompt_orig,
            prompt_new=prompt_new,
            num_examples=len(dataset_orig),
            rank=rank,
        )
        _write_baseline_summary_stub(
            output_dir_path / f"baseline_summary{output_prefix_suffix}.txt",
            dtype=dtype,
            prompt_orig=prompt_orig,
            prompt_new=prompt_new,
            num_examples=len(dataset_orig),
            rank=rank,
        )

    receiver_hooks = [(f"blocks.{model.cfg.n_layers - 1}.hook_resid_post", None)]
    baseline_ppl, head_results, mlp_results = _scan_components_with_subspace(
        model=model,
        dataset_orig=dataset_orig,
        dataset_new=dataset_new,
        rank=rank,
        batch_size=batch_size,
        identification_batch_size=identification_batch_size,
        receiver_hooks=receiver_hooks,
        freeze_mlps=freeze_mlps,
        have_internal_interactions=have_internal_interactions,
    )

    _save_results(
        output_dir=output_dir_path,
        prefix=f"subspace_{prompt_new}_into_{prompt_orig}{output_prefix_suffix}",
        baseline_ppl=baseline_ppl,
        head_results=head_results,
        mlp_results=mlp_results,
        rank=rank,
        top_k=top_k,
        dtype=dtype,
    )

    results.update(
        {
            "baseline_ppl": baseline_ppl,
            "top_heads": _top_heads(head_results, top_k=top_k),
            "top_mlps": _top_mlps(mlp_results, top_k=top_k),
        }
    )
    return results


def _build_source_file_results(
    model,
    tokenizer,
    records_orig,
    records_new,
    prompt_orig: str,
    prompt_new: str,
    batch_size: int,
    identification_batch_size: int,
    rank: int,
    top_k: int,
    dtype: str,
    output_dir_path: Optional[Path] = None,
    freeze_mlps: bool = False,
    have_internal_interactions: bool = False,
):
    grouped_results: Dict[str, Any] = {}
    grouped_pos: Dict[str, List[Dict[str, Any]]] = {}
    grouped_neg: Dict[str, List[Dict[str, Any]]] = {}

    for record in records_orig:
        source_file = _normalize_source_file(record.get("source_file"))
        grouped_pos.setdefault(source_file, []).append(record)

    for record in records_new:
        source_file = _normalize_source_file(record.get("source_file"))
        grouped_neg.setdefault(source_file, []).append(record)

    if set(grouped_pos) != set(grouped_neg):
        raise ValueError(
            f"source_file mismatch between prompt sets: {sorted(grouped_pos)} vs {sorted(grouped_neg)}"
        )

    for source_file in sorted(grouped_pos):
        source_records_orig = grouped_pos[source_file]
        source_records_new = grouped_neg[source_file]
        dataset_orig = CustomTranslationDataset(source_records_orig, tokenizer=tokenizer)
        dataset_new = CustomTranslationDataset(source_records_new, tokenizer=tokenizer)
        source_suffix = f"__{_safe_output_name(Path(source_file).stem)}"

        grouped_results[source_file] = _analyze_dataset_pair(
            model=model,
            dataset_orig=dataset_orig,
            dataset_new=dataset_new,
            records_orig=source_records_orig,
            records_new=source_records_new,
            prompt_orig=prompt_orig,
            prompt_new=prompt_new,
            batch_size=batch_size,
            identification_batch_size=identification_batch_size,
            rank=rank,
            top_k=top_k,
            dtype=dtype,
            output_dir_path=output_dir_path,
            output_prefix_suffix=source_suffix,
            freeze_mlps=freeze_mlps,
            have_internal_interactions=have_internal_interactions,
        )

    return grouped_results


def compare_prompt_styles_with_subspace(
    folder_path: str,
    model_name: str,
    prompt_orig: str = "multipivot",
    prompt_new: str = "direct",
    max_examples: Optional[int] = None,
    batch_size: int = 1,
    identification_batch_size: int = 1,
    rank: int = 1,
    top_k: int = 20,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    dtype: str = "auto",
    freeze_mlps: bool = False,
    have_internal_interactions: bool = False,
):
    """
    Single-GPU subspace component analysis aligned with the reference path-patching
    workflow, while scoring outputs by generated-span perplexity.

    Notes:
    - forces a single device even if a comma-separated device list is passed
    - uses `attn.hook_z` instead of `attn.hook_result`
    - `identification_batch_size` is kept for CLI compatibility, but identification
      now follows the reference implementation directly
    """
    output_dir_path = _ensure_dir(output_dir)

    model, tokenizer, resolved_device, resolved_dtype = _load_easy_transformer(
        model_name=model_name,
        device=device,
        dtype=dtype,
    )
    batch_size, identification_batch_size = _auto_batch_sizes(
        device=resolved_device,
        batch_size=batch_size,
        identification_batch_size=identification_batch_size,
    )

    dataset_orig, records_orig = _build_dataset(
        folder_path=folder_path,
        prompt_type=prompt_orig,
        tokenizer=tokenizer,
        max_examples=max_examples,
    )
    dataset_new, records_new = _build_dataset(
        folder_path=folder_path,
        prompt_type=prompt_new,
        tokenizer=tokenizer,
        max_examples=max_examples,
    )

    results: Dict[str, Any] = {
        "model_name": model_name,
        "device": resolved_device,
        "dtype": resolved_dtype,
        "folder_path": folder_path,
        "prompt_orig": prompt_orig,
        "prompt_new": prompt_new,
        "method": "subspace_intervened_path_patching",
        "metric": "generated_span_perplexity",
        "subspace_rank": rank,
        "implementation": "single_gpu_memory_optimized",
        "attention_component": "hook_z",
        "batch_size": batch_size,
        "identification_batch_size": identification_batch_size,
    }

    overall_results = _analyze_dataset_pair(
        model=model,
        dataset_orig=dataset_orig,
        dataset_new=dataset_new,
        records_orig=records_orig,
        records_new=records_new,
        prompt_orig=prompt_orig,
        prompt_new=prompt_new,
        batch_size=batch_size,
        identification_batch_size=identification_batch_size,
        rank=rank,
        top_k=top_k,
        dtype=resolved_dtype,
        output_dir_path=output_dir_path,
        freeze_mlps=freeze_mlps,
        have_internal_interactions=have_internal_interactions,
    )
    per_source_file_results = _build_source_file_results(
        model=model,
        tokenizer=tokenizer,
        records_orig=records_orig,
        records_new=records_new,
        prompt_orig=prompt_orig,
        prompt_new=prompt_new,
        batch_size=batch_size,
        identification_batch_size=identification_batch_size,
        rank=rank,
        top_k=top_k,
        dtype=resolved_dtype,
        output_dir_path=output_dir_path,
        freeze_mlps=freeze_mlps,
        have_internal_interactions=have_internal_interactions,
    )

    if output_dir_path is not None:
        _write_overall_summary(
            output_path=output_dir_path / "subspace_baseline_summary.txt",
            model_name=model_name,
            resolved_device=resolved_device,
            resolved_dtype=resolved_dtype,
            folder_path=folder_path,
            prompt_orig=prompt_orig,
            prompt_new=prompt_new,
            rank=rank,
            overall_results=overall_results,
            per_source_file_results=per_source_file_results,
        )
        _write_legacy_overall_summary(
            output_path=output_dir_path / "baseline_summary.txt",
            model_name=model_name,
            resolved_device=resolved_device,
            folder_path=folder_path,
            prompt_orig=prompt_orig,
            prompt_new=prompt_new,
            overall_results=overall_results,
            per_source_file_results=per_source_file_results,
        )

    results.update(
        {
            **overall_results,
            "per_source_file": per_source_file_results,
        }
    )
    return results


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "compare_prompt_styles_with_subspace": compare_prompt_styles_with_subspace,
        }
    )
