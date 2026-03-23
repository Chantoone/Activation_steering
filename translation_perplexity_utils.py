from typing import List, Optional, Sequence, Tuple
import warnings

import torch
from tqdm import tqdm

from easy_transformer.experiments import get_act_hook
from easy_transformer.utils import lm_cross_entropy_loss
from translation_utils import handle_all_and_std, patch_all


Component = Tuple[str, Optional[int]]


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_generated_prediction_positions(
    translation_dataset,
) -> List[torch.Tensor]:
    """
    Return the positions whose logits predict the target translation tokens.

    If the target tokens occupy absolute positions [s, ..., e] in the prompt,
    then we need logits from positions [s-1, ..., e-1].
    For single-token targets this reduces to the original `word_idx["end"]`.
    """
    cached = getattr(translation_dataset, "_generated_prediction_positions", None)
    if cached is not None:
        return cached

    positions: List[torch.Tensor] = []
    prepend_bos = getattr(translation_dataset, "prepend_bos", False)

    for i, sample in enumerate(translation_dataset.preprocess_df_trans):
        prompt = sample["prompt"]
        target_text = sample["out_token_str"]
        prefix_text = sample.get("prefix_prompt")
        if prefix_text is None:
            if not prompt.endswith(target_text):
                raise ValueError(
                    f"Prompt at index {i} does not end with its target text, "
                    "and no prefix_prompt was provided."
                )
            prefix_text = prompt[: len(prompt) - len(target_text)]
        elif not isinstance(prefix_text, str):
            raise ValueError(
                f"Sample {i} has invalid prefix_prompt of type {type(prefix_text)}."
            )

        prefix_ids = translation_dataset.tokenizer(
            prefix_text,
            add_special_tokens=prepend_bos,
        )["input_ids"]
        full_len = int(translation_dataset.attn_mask[i].sum().item())
        prefix_len = len(prefix_ids)

        if prefix_len <= 0:
            raise ValueError(
                f"Sample {i} has no prefix tokens before the generated target span."
            )
        if prefix_len >= full_len:
            raise ValueError(
                f"Sample {i} has empty generated target span (prefix_len={prefix_len}, "
                f"full_len={full_len})."
            )

        positions.append(torch.arange(prefix_len - 1, full_len - 1, dtype=torch.long))

    translation_dataset._generated_prediction_positions = positions
    return positions


def get_generated_target_token_mask(translation_dataset, device=None) -> torch.Tensor:
    """Mask of shape [batch, seq-1] selecting losses for all generated target tokens."""
    prediction_positions = get_generated_prediction_positions(translation_dataset)
    if device is None:
        device = translation_dataset.toks.device

    mask = torch.zeros(
        (len(translation_dataset), translation_dataset.toks.shape[1] - 1),
        dtype=torch.bool,
        device=device,
    )
    for i, pos in enumerate(prediction_positions):
        if pos.numel() > 0:
            mask[i, pos.to(device)] = True
    return mask


def get_generated_token_nll(
    model,
    translation_dataset,
    all=False,
    std=False,
):
    """
    Mean next-token negative log-likelihood over the full generated target span.
    """
    device = _get_model_device(model)
    toks = translation_dataset.toks.long().to(device)

    with torch.no_grad():
        logits = model(toks).detach()

    per_token_nll = lm_cross_entropy_loss(logits, toks, return_per_token=True)
    target_mask = get_generated_target_token_mask(translation_dataset, device=per_token_nll.device)
    token_counts = target_mask.sum(dim=1).clamp_min(1)
    per_example_nll = (per_token_nll * target_mask).sum(dim=1) / token_counts
    return handle_all_and_std(per_example_nll, all, std)


def get_generated_token_perplexity(
    model,
    translation_dataset,
    all=False,
    std=False,
):
    """
    Per-example perplexity computed on all generated target tokens.
    """
    device = _get_model_device(model)
    toks = translation_dataset.toks.long().to(device)

    with torch.no_grad():
        logits = model(toks).detach()

    per_token_nll = lm_cross_entropy_loss(logits, toks, return_per_token=True)
    target_mask = get_generated_target_token_mask(translation_dataset, device=per_token_nll.device)
    token_counts = target_mask.sum(dim=1).clamp_min(1)
    per_example_nll = (per_token_nll * target_mask).sum(dim=1) / token_counts
    per_example_ppl = torch.exp(per_example_nll)
    return handle_all_and_std(per_example_ppl, all, std)


def _patch_generated_positions(
    z: torch.Tensor,
    source_act: torch.Tensor,
    batch_orig_positions: Sequence[torch.Tensor],
    batch_new_positions: Sequence[torch.Tensor],
):
    for batch_idx, (orig_pos, new_pos) in enumerate(
        zip(batch_orig_positions, batch_new_positions)
    ):
        if orig_pos.numel() == 0:
            continue
        if orig_pos.numel() != new_pos.numel():
            raise ValueError(
                f"Mismatched generated span lengths in batch item {batch_idx}: "
                f"{orig_pos.numel()} vs {new_pos.numel()}"
            )
        z[batch_idx, orig_pos.to(z.device)] = source_act[
            batch_idx, new_pos.to(source_act.device)
        ].to(z.device)
    return z


def _positions_allclose(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    batch_positions: Sequence[torch.Tensor],
) -> bool:
    for batch_idx, pos in enumerate(batch_positions):
        if pos.numel() == 0:
            continue
        pos = pos.to(lhs.device)
        if not torch.allclose(lhs[batch_idx, pos], rhs[batch_idx, pos]):
            return False
    return True


def batch_path_patching_generated_perplexity(
    model,
    D_new,
    D_orig,
    sender_heads,
    receiver_hooks,
    batch_size=128,
    extra_hooks=[],
    freeze_mlps=False,
    have_internal_interactions=False,
):
    """
    Batched path patching scored by perplexity over the entire generated target span.

    Unlike the original `batch_path_patching`, this patches all prediction positions
    corresponding to target generation, not just the single `end` position.
    """
    model_device = _get_model_device(model)
    all_perplexities = torch.zeros(len(D_orig), device=model_device)

    full_new_positions = get_generated_prediction_positions(D_new)
    full_orig_positions = get_generated_prediction_positions(D_orig)

    for i in tqdm(range(0, len(D_new), batch_size)):
        batch_D_new = D_new[i : i + batch_size]
        batch_D_orig = D_orig[i : i + batch_size]
        batch_new_positions = full_new_positions[i : i + batch_size]
        batch_orig_positions = full_orig_positions[i : i + batch_size]

        def patch_positions(z, source_act, hook):
            return _patch_generated_positions(
                z=z,
                source_act=source_act,
                batch_orig_positions=batch_orig_positions,
                batch_new_positions=batch_new_positions,
            )

        sender_hooks = []
        for layer, head_idx in sender_heads:
            if head_idx is None:
                sender_hooks.append((f"blocks.{layer}.hook_mlp_out", None))
            else:
                sender_hooks.append((f"blocks.{layer}.attn.hook_result", head_idx))

        sender_hook_names = [x[0] for x in sender_hooks]
        receiver_hook_names = [x[0] for x in receiver_hooks]

        sender_cache = {}
        model.reset_hooks()
        for hook in extra_hooks:
            model.add_hook(*hook)
        model.cache_some(
            sender_cache, lambda x: x in sender_hook_names, suppress_warning=True
        )
        with torch.no_grad():
            model(batch_D_new.toks.long().to(model_device))

        target_cache = {}
        model.reset_hooks()
        for hook in extra_hooks:
            model.add_hook(*hook)
        model.cache_all(target_cache, suppress_warning=True)
        with torch.no_grad():
            model(batch_D_orig.toks.long().to(model_device))

        receiver_cache = {}
        model.reset_hooks()
        model.cache_some(
            receiver_cache,
            lambda x: x in receiver_hook_names,
            suppress_warning=True,
            verbose=False,
        )

        for layer in range(model.cfg.n_layers):
            for head_idx in range(model.cfg.n_heads):
                for hook_template in [
                    "blocks.{}.attn.hook_q",
                    "blocks.{}.attn.hook_k",
                    "blocks.{}.attn.hook_v",
                ]:
                    hook_name = hook_template.format(layer)

                    if have_internal_interactions and hook_name in receiver_hook_names:
                        continue

                    hook = get_act_hook(
                        patch_all,
                        alt_act=target_cache[hook_name],
                        idx=head_idx,
                        dim=2 if head_idx is not None else None,
                        name=hook_name,
                    )
                    model.add_hook(hook_name, hook)

            if freeze_mlps:
                hook_name = f"blocks.{layer}.hook_mlp_out"
                hook = get_act_hook(
                    patch_all,
                    alt_act=target_cache[hook_name],
                    idx=None,
                    dim=None,
                    name=hook_name,
                )
                model.add_hook(hook_name, hook)

        for hook in extra_hooks:
            model.add_hook(*hook)

        for hook_name, head_idx in sender_hooks:
            assert not torch.allclose(sender_cache[hook_name], target_cache[hook_name]), (
                hook_name,
                head_idx,
            )
            hook = get_act_hook(
                patch_positions,
                alt_act=sender_cache[hook_name],
                idx=head_idx,
                dim=2 if head_idx is not None else None,
                name=hook_name,
            )
            model.add_hook(hook_name, hook)

        with torch.no_grad():
            model(batch_D_orig.toks.long().to(model_device))

        model.reset_hooks()
        hooks = []
        for hook in extra_hooks:
            hooks.append(hook)

        for hook_name, head_idx in receiver_hooks:
            if _positions_allclose(
                receiver_cache[hook_name],
                target_cache[hook_name],
                batch_orig_positions,
            ):
                warnings.warn(f"Torch all close for {hook_name}")

            hook = get_act_hook(
                patch_positions,
                alt_act=receiver_cache[hook_name],
                idx=head_idx,
                dim=2 if head_idx is not None else None,
                name=hook_name,
            )
            hooks.append((hook_name, hook))

        model.reset_hooks()
        for hook_name, hook in hooks:
            model.add_hook(hook_name, hook)

        cur_perplexity = get_generated_token_perplexity(model, batch_D_orig, all=True)
        all_perplexities[i : i + batch_size] = cur_perplexity.to(model_device)

    return all_perplexities.mean().item()


def scan_path_patching_by_generated_perplexity(
    model,
    translation_dataset,
    flipped_translation_dataset,
    receiver_hooks,
    batch_size=128,
    freeze_mlps=False,
    have_internal_interactions=False,
    normalize=True,
):
    """
    Scan all attention heads and MLPs, measuring their effect via generated-span perplexity.

    Returns:
        baseline_perplexity, head_results, mlp_results

    Interpretation:
        Positive values mean patching from the flipped dataset increases perplexity
        relative to the original task, so the component is more important.
    """
    model.reset_hooks()
    baseline_perplexity = float(
        get_generated_token_perplexity(model, translation_dataset).item()
    )

    head_results = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    mlp_results = torch.zeros((model.cfg.n_layers, 1))

    for source_layer in tqdm(range(model.cfg.n_layers)):
        for source_head_idx in [None] + list(range(model.cfg.n_heads)):
            model.reset_hooks()
            cur_perplexity = batch_path_patching_generated_perplexity(
                model=model,
                D_new=flipped_translation_dataset,
                D_orig=translation_dataset,
                sender_heads=[(source_layer, source_head_idx)],
                receiver_hooks=receiver_hooks,
                batch_size=batch_size,
                extra_hooks=[],
                freeze_mlps=freeze_mlps,
                have_internal_interactions=have_internal_interactions,
            )

            delta = cur_perplexity - baseline_perplexity
            if source_head_idx is None:
                mlp_results[source_layer] = delta
            else:
                head_results[source_layer, source_head_idx] = delta

    if normalize:
        head_results = 100 * head_results / baseline_perplexity
        mlp_results = 100 * mlp_results / baseline_perplexity

    return baseline_perplexity, head_results, mlp_results
