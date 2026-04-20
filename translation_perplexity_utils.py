from typing import Callable, Dict, List, Optional, Sequence, Tuple
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


def _get_generated_per_example_nll(
    model,
    translation_dataset,
    batch_size: int = 4,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    all_nll = []
    model_device = _get_model_device(model)

    for start in range(0, len(translation_dataset), batch_size):
        batch_dataset = translation_dataset[start : start + batch_size]
        toks = batch_dataset.toks.long().to(model_device)

        with torch.no_grad():
            logits = model(toks).detach()

        if toks.device != logits.device:
            toks = toks.to(logits.device)
        per_token_nll = lm_cross_entropy_loss(logits, toks, return_per_token=True)
        target_mask = get_generated_target_token_mask(
            batch_dataset, device=per_token_nll.device
        )
        token_counts = target_mask.sum(dim=1).clamp_min(1)
        per_example_nll = (per_token_nll * target_mask).sum(dim=1) / token_counts
        all_nll.append(per_example_nll.to(model_device))

        del logits
        del toks
        del per_token_nll
        del target_mask
        del token_counts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(all_nll, dim=0)


def get_generated_token_nll(
    model,
    translation_dataset,
    all=False,
    std=False,
    batch_size: int = 4,
):
    """
    Mean next-token negative log-likelihood over the full generated target span.
    """
    per_example_nll = _get_generated_per_example_nll(
        model,
        translation_dataset,
        batch_size=batch_size,
    )
    return handle_all_and_std(per_example_nll, all, std)


def get_generated_token_perplexity(
    model,
    translation_dataset,
    all=False,
    std=False,
    batch_size: int = 4,
):
    """
    Per-example perplexity computed on all generated target tokens.
    """
    per_example_nll = _get_generated_per_example_nll(
        model,
        translation_dataset,
        batch_size=batch_size,
    )
    per_example_ppl = torch.exp(per_example_nll)
    return handle_all_and_std(per_example_ppl, all, std)


def get_generated_token_perplexity_from_logits(
    logits: torch.Tensor,
    translation_dataset,
) -> torch.Tensor:
    """Per-example perplexity over the full generated target span from precomputed logits."""
    toks = translation_dataset.toks.long()
    if toks.device != logits.device:
        toks = toks.to(logits.device)

    per_token_nll = lm_cross_entropy_loss(logits, toks, return_per_token=True)
    target_mask = get_generated_target_token_mask(
        translation_dataset,
        device=per_token_nll.device,
    )
    token_counts = target_mask.sum(dim=1).clamp_min(1)
    per_example_nll = (per_token_nll * target_mask).sum(dim=1) / token_counts
    return torch.exp(per_example_nll)


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
        orig_pos = orig_pos.to(z.device)
        new_pos = new_pos.to(source_act.device)

        # Prompt styles can tokenize to different lengths, so only patch positions
        # that exist in both the destination and source activations.
        orig_pos = orig_pos[orig_pos < z.shape[1]]
        new_pos = new_pos[new_pos < source_act.shape[1]]

        shared_len = min(orig_pos.numel(), new_pos.numel())
        if shared_len == 0:
            continue

        z[batch_idx, orig_pos[:shared_len]] = source_act[
            batch_idx, new_pos[:shared_len]
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


def _project_patch(
    a_plus: torch.Tensor,
    a_minus: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    compute_dtype = a_plus.dtype
    if a_plus.device.type == "cpu" and a_plus.dtype in {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float32
    elif W.dtype != a_plus.dtype:
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


def _extract_component_sequence(
    component: Component,
    activations: Dict[str, torch.Tensor],
    batch_idx: int,
    positions: torch.Tensor,
) -> torch.Tensor:
    hook_name, head_idx = component
    act = activations[hook_name]
    if act.dim() == 4:
        assert head_idx is not None, f"Head index must be provided for {hook_name}"
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
    hook_name, head_idx = component
    if z.dim() == 4:
        assert head_idx is not None, f"Head index must be provided for {hook_name}"
        z[batch_idx, positions, head_idx] = patched_value.to(z.device)
    elif z.dim() == 3:
        z[batch_idx, positions] = patched_value.to(z.device)
    else:
        raise ValueError(f"Unsupported activation shape {z.shape} for {hook_name}")
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
        shared_positions.append(
            (
                orig_trimmed[:shared_len],
                new_trimmed[:shared_len],
            )
        )
    return shared_positions


def _get_dataset_positions(
    dataset,
    position_key: str = "end",
    device=None,
) -> torch.Tensor:
    positions = dataset.word_idx[position_key]
    if not isinstance(positions, torch.Tensor):
        positions = torch.tensor(positions, dtype=torch.long)
    if device is not None:
        positions = positions.to(device)
    return positions


def task_steering_subspace_identification_generated(
    model,
    dataset_pos,
    dataset_neg,
    component: Component,
    rank: int = 1,
):
    """
    Identify a task-steering subspace for a component over the full generated span.

    Each generated prediction position contributes one row to the contrastive matrix.
    """
    assert len(dataset_pos) == len(dataset_neg), "Contrastive datasets must align"
    hook_name, _ = component

    pos_cache = {}
    model.reset_hooks()
    model.cache_some(pos_cache, lambda name: name == hook_name, suppress_warning=True)
    with torch.no_grad():
        _ = model(dataset_pos.toks.long())
    model.reset_hooks()

    neg_cache = {}
    model.cache_some(neg_cache, lambda name: name == hook_name, suppress_warning=True)
    with torch.no_grad():
        _ = model(dataset_neg.toks.long())
    model.reset_hooks()

    pos_positions = get_generated_prediction_positions(dataset_pos)
    neg_positions = get_generated_prediction_positions(dataset_neg)
    shared_positions = _shared_generated_positions(
        pos_positions,
        neg_positions,
        orig_seq_len=pos_cache[hook_name].shape[1],
        new_seq_len=neg_cache[hook_name].shape[1],
    )

    delta_rows = []
    for batch_idx, (pos_idx, neg_idx) in enumerate(shared_positions):
        if pos_idx.numel() == 0:
            continue
        pos_idx = pos_idx.to(pos_cache[hook_name].device)
        neg_idx = neg_idx.to(neg_cache[hook_name].device)
        a_pos = _extract_component_sequence(component, pos_cache, batch_idx, pos_idx)
        a_neg = _extract_component_sequence(component, neg_cache, batch_idx, neg_idx)
        delta_rows.append(a_pos - a_neg)

    if not delta_rows:
        raise ValueError(f"No shared generated positions found for component {component}")

    deltas = torch.cat(delta_rows, dim=0)
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


def subspace_intervene_path_patching(
    model,
    dataset_orig,
    dataset_new,
    components: List[Component],
    subspace_dict: Dict[Component, torch.Tensor],
    epsilon: float = 1e-8,
    receiver_hooks: Optional[List[Component]] = None,
    positions: List[str] = ["end"],
    freeze_mlps: bool = False,
    have_internal_interactions: bool = False,
    metric_fn: Optional[Callable[[torch.Tensor, object], torch.Tensor]] = None,
):
    """
    Subspace path patching on explicit dataset positions, scored by perplexity-derived metrics.

    This mirrors `translation_utils.subspace_intervene_path_patching(...)` but defaults
    to generated-span perplexity from logits as the output metric.
    """
    assert len(dataset_orig) == len(dataset_new), "Contrastive datasets must align"
    if receiver_hooks is None:
        raise ValueError("receiver_hooks must be provided to limit patched paths")
    if metric_fn is None:
        metric_fn = get_generated_token_perplexity_from_logits

    receiver_hook_names = [x[0] for x in receiver_hooks]

    model.reset_hooks()
    with torch.no_grad():
        logits_orig = model(dataset_orig.toks.long())
    y_orig = metric_fn(logits_orig, dataset_orig)

    pos_orig = {
        pos: _get_dataset_positions(dataset_orig, pos, device=logits_orig.device)
        for pos in positions
    }
    pos_new = {
        pos: _get_dataset_positions(dataset_new, pos, device=logits_orig.device)
        for pos in positions
    }

    deltas = {}
    for component in components:
        hook_name, _ = component
        if component not in subspace_dict:
            raise KeyError(f"No subspace provided for component {component}")
        W_c = subspace_dict[component].to(logits_orig.device)

        sender_cache = {}
        model.reset_hooks()
        model.cache_some(sender_cache, lambda name: name == hook_name, suppress_warning=True)
        with torch.no_grad():
            _ = model(dataset_new.toks.long())

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
        model.reset_hooks()
        model.cache_some(
            target_cache,
            lambda name: name in target_hook_names,
            suppress_warning=True,
        )
        with torch.no_grad():
            _ = model(dataset_orig.toks.long())

        def subspace_patch(
            z,
            hook,
            comp=component,
            W=W_c,
            src=sender_cache[hook_name],
            tgt=target_cache[hook_name],
            orig_positions=pos_orig,
            new_positions=pos_new,
        ):
            for pos in positions:
                orig_idx = orig_positions[pos]
                new_idx = new_positions[pos]
                valid_orig = orig_idx < z.shape[1]
                valid_new = new_idx < src.shape[1]
                valid = valid_orig & valid_new
                if not valid.any():
                    continue
                batch_indices = torch.arange(z.shape[0], device=z.device)[valid.to(z.device)]
                orig_sel = orig_idx[valid].to(z.device)
                new_sel = new_idx[valid].to(src.device)
                _, head_idx = comp
                batch_idx_tgt = batch_indices.to(tgt.device)
                batch_idx_src = batch_indices.to(src.device)
                orig_sel_tgt = orig_sel.to(tgt.device)
                if tgt.dim() == 4:
                    a_plus = tgt[batch_idx_tgt, orig_sel_tgt, head_idx]
                    a_minus = src[batch_idx_src, new_sel, head_idx]
                else:
                    a_plus = tgt[batch_idx_tgt, orig_sel_tgt]
                    a_minus = src[batch_idx_src, new_sel]
                patched = _project_patch(a_plus, a_minus, W)
                if z.dim() == 4:
                    z[batch_indices, orig_sel, head_idx] = patched.to(z.device)
                else:
                    z[batch_indices, orig_sel] = patched.to(z.device)
            return z

        receiver_cache = {}
        model.reset_hooks()
        model.cache_some(receiver_cache, lambda n: n in receiver_hook_names, suppress_warning=True)
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
                    model.add_hook(h_name, freeze_hook)
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
                model.add_hook(h_name, freeze_hook)
        model.add_hook(hook_name, subspace_patch)
        with torch.no_grad():
            _ = model(dataset_orig.toks.long())

        model.reset_hooks()
        for h_name, hk in freezer_hooks:
            model.add_hook(h_name, hk)
        model.add_hook(hook_name, subspace_patch)
        for r_name, r_head in receiver_hooks:
            cached_act = receiver_cache[r_name]

            def restore_positions(
                z,
                hook,
                alt_act=cached_act,
                comp=(r_name, r_head),
                orig_positions=pos_orig,
            ):
                for pos in positions:
                    pos_idx = orig_positions[pos]
                    valid = pos_idx < z.shape[1]
                    if not valid.any():
                        continue
                    batch_indices = torch.arange(z.shape[0], device=z.device)[valid.to(z.device)]
                    dst_idx = pos_idx[valid].to(z.device)
                    src_idx = pos_idx[valid].to(alt_act.device)
                    _, head_idx = comp
                    batch_idx_src = batch_indices.to(alt_act.device)
                    if alt_act.dim() == 4:
                        patched_val = alt_act[batch_idx_src, src_idx, head_idx]
                    else:
                        patched_val = alt_act[batch_idx_src, src_idx]
                    if z.dim() == 4:
                        z[batch_indices, dst_idx, head_idx] = patched_val.to(z.device)
                    else:
                        z[batch_indices, dst_idx] = patched_val.to(z.device)
                return z

            model.add_hook(r_name, restore_positions)

        with torch.no_grad():
            logits_new = model(dataset_orig.toks.long())

        delta = (metric_fn(logits_new, dataset_orig) - y_orig) / (y_orig + epsilon)
        deltas[component] = delta.mean().detach().cpu()
        model.reset_hooks()

    return deltas


def subspace_intervene_path_patching_generated_perplexity(
    model,
    dataset_orig,
    dataset_new,
    components: List[Component],
    subspace_dict: Dict[Component, torch.Tensor],
    epsilon: float = 1e-8,
    receiver_hooks: Optional[List[Component]] = None,
    freeze_mlps: bool = False,
    have_internal_interactions: bool = False,
    metric_fn: Optional[Callable[[torch.Tensor, object], torch.Tensor]] = None,
):
    """
    Subspace path patching over all generated prediction positions, scored by generated-span PPL.

    `dataset_new` provides the steering signal being patched into `dataset_orig`.
    """
    assert len(dataset_orig) == len(dataset_new), "Contrastive datasets must align"
    if receiver_hooks is None:
        raise ValueError("receiver_hooks must be provided to limit patched paths")
    if metric_fn is None:
        metric_fn = get_generated_token_perplexity_from_logits

    receiver_hook_names = [x[0] for x in receiver_hooks]

    model.reset_hooks()
    with torch.no_grad():
        logits_orig = model(dataset_orig.toks.long())
    y_orig = metric_fn(logits_orig, dataset_orig)

    orig_positions = get_generated_prediction_positions(dataset_orig)
    new_positions = get_generated_prediction_positions(dataset_new)

    deltas = {}
    for component in components:
        hook_name, _ = component
        if component not in subspace_dict:
            raise KeyError(f"No subspace provided for component {component}")
        W_c = subspace_dict[component].to(logits_orig.device)

        sender_cache = {}
        model.reset_hooks()
        model.cache_some(sender_cache, lambda name: name == hook_name, suppress_warning=True)
        with torch.no_grad():
            _ = model(dataset_new.toks.long())

        target_cache = {}
        model.reset_hooks()
        model.cache_all(target_cache, suppress_warning=True)
        with torch.no_grad():
            _ = model(dataset_orig.toks.long())

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
            W=W_c,
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

        receiver_cache = {}
        model.reset_hooks()
        model.cache_some(receiver_cache, lambda n: n in receiver_hook_names, suppress_warning=True)
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
                    model.add_hook(h_name, freeze_hook)
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
                model.add_hook(h_name, freeze_hook)
        model.add_hook(hook_name, subspace_patch)
        with torch.no_grad():
            _ = model(dataset_orig.toks.long())

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
            logits_new = model(dataset_orig.toks.long())

        delta = (metric_fn(logits_new, dataset_orig) - y_orig) / (y_orig + epsilon)
        deltas[component] = delta.mean().detach().cpu()
        model.reset_hooks()

    return deltas


def subspace_intervene_path_patching_generated_perplexity_batch(
    model,
    dataset_orig,
    dataset_new,
    components: List[Component],
    subspace_dict: Dict[Component, torch.Tensor],
    epsilon: float = 1e-8,
    receiver_hooks: Optional[List[Component]] = None,
    freeze_mlps: bool = False,
    have_internal_interactions: bool = False,
    batch_size: int = 128,
    metric_fn: Optional[Callable[[torch.Tensor, object], torch.Tensor]] = None,
):
    """Batched generated-span subspace patching scored by full-span perplexity."""
    if metric_fn is None:
        metric_fn = get_generated_token_perplexity_from_logits
    assert receiver_hooks is not None, "receiver_hooks must be provided"

    totals = {comp: 0.0 for comp in components}
    counts = 0

    for start in range(0, len(dataset_orig), batch_size):
        end = start + batch_size
        orig_batch = dataset_orig[start:end]
        new_batch = dataset_new[start:end]
        batch_N = len(orig_batch)
        batch_deltas = subspace_intervene_path_patching_generated_perplexity(
            model=model,
            dataset_orig=orig_batch,
            dataset_new=new_batch,
            components=components,
            subspace_dict=subspace_dict,
            epsilon=epsilon,
            receiver_hooks=receiver_hooks,
            freeze_mlps=freeze_mlps,
            have_internal_interactions=have_internal_interactions,
            metric_fn=metric_fn,
        )
        for comp, delta in batch_deltas.items():
            totals[comp] += float(delta) * batch_N
        counts += batch_N

    return {comp: totals[comp] / max(counts, 1) for comp in components}


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
                # `hook_result` only exists when use_attn_result=True, which is very
                # memory-expensive. Use `hook_z` so we can patch individual heads
                # without materializing full head results.
                sender_hooks.append((f"blocks.{layer}.attn.hook_z", head_idx))

        sender_hook_names = [x[0] for x in sender_hooks]
        receiver_hook_names = [x[0] for x in receiver_hooks]
        target_hook_names = set(sender_hook_names + receiver_hook_names)
        for layer in range(model.cfg.n_layers):
            for hook_template in [
                "blocks.{}.attn.hook_q",
                "blocks.{}.attn.hook_k",
                "blocks.{}.attn.hook_v",
            ]:
                target_hook_names.add(hook_template.format(layer))
            if freeze_mlps:
                target_hook_names.add(f"blocks.{layer}.hook_mlp_out")

        sender_cache = {}
        model.reset_hooks()
        for hook in extra_hooks:
            model.add_hook(*hook)
        model.cache_some(
            sender_cache,
            lambda x: x in sender_hook_names,
            device="cpu",
            suppress_warning=True,
        )
        with torch.no_grad():
            model(batch_D_new.toks.long().to(model_device))

        target_cache = {}
        model.reset_hooks()
        for hook in extra_hooks:
            model.add_hook(*hook)
        model.cache_some(
            target_cache,
            lambda x: x in target_hook_names,
            device="cpu",
            suppress_warning=True,
        )
        with torch.no_grad():
            model(batch_D_orig.toks.long().to(model_device))

        receiver_cache = {}
        model.reset_hooks()
        model.cache_some(
            receiver_cache,
            lambda x: x in receiver_hook_names,
            device="cpu",
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
            sender_act = sender_cache[hook_name]
            target_act = target_cache[hook_name]
            if sender_act.shape == target_act.shape:
                assert not torch.allclose(sender_act, target_act), (
                    hook_name,
                    head_idx,
                )
            hook = get_act_hook(
                patch_positions,
                alt_act=sender_act,
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

        cur_perplexity = get_generated_token_perplexity(
            model,
            batch_D_orig,
            all=True,
            batch_size=batch_size,
        )
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
        get_generated_token_perplexity(
            model,
            translation_dataset,
            batch_size=batch_size,
        ).item()
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
