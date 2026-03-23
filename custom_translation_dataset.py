from copy import deepcopy
from typing import Dict, List, Optional

import torch


class CustomTranslationDataset:
    """
    Lightweight dataset wrapper compatible with translation_perplexity_utils.py.

    Expected sample schema:
    {
        "prefix_prompt": str,           # optional, recommended for chat/instruct models
        "prompt": str,
        "out_token_str": str,
        "out_token_id": list[int],      # optional but recommended
        "in_token_str": str,            # optional
        "in_token_id": list[int],       # optional
        "template_idx": int,            # optional
    }

    Important constraint:
    - either provide sample["prefix_prompt"], or ensure
      sample["prompt"].endswith(sample["out_token_str"])
    """

    REQUIRED_FIELDS = ("prompt", "out_token_str")

    def __init__(
        self,
        samples: List[Dict],
        tokenizer,
        prepend_bos: bool = False,
        validate: bool = True,
        auto_fill_token_ids: bool = True,
    ):
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos
        self.preprocess_df_trans = deepcopy(samples)

        if validate:
            self._validate_samples()

        if auto_fill_token_ids:
            self._fill_missing_token_fields()

        self.sentences = [sample["prompt"] for sample in self.preprocess_df_trans]

        tokenized = self.tokenizer(
            self.sentences,
            padding=True,
            add_special_tokens=self.prepend_bos,
            return_tensors="pt",
        )
        self.toks = tokenized["input_ids"]
        self.attn_mask = tokenized["attention_mask"]

        self.N = len(self.preprocess_df_trans)
        self.max_len = int(self.toks.shape[1]) if self.N > 0 else 0

        self.in_tok_ids = [
            sample.get("in_token_id", []) for sample in self.preprocess_df_trans
        ]
        self.out_tok_ids = [
            sample.get("out_token_id", []) for sample in self.preprocess_df_trans
        ]
        self.translation_tokenIDs = self.out_tok_ids

        self.groups = self._build_groups()
        self.word_idx = self._build_word_idx()
        self.tokenized_prompts = self._build_tokenized_prompts()

    def _validate_samples(self):
        if not isinstance(self.preprocess_df_trans, list):
            raise TypeError("samples must be a list of dictionaries")

        for idx, sample in enumerate(self.preprocess_df_trans):
            if not isinstance(sample, dict):
                raise TypeError(f"Sample at index {idx} must be a dictionary")

            for field in self.REQUIRED_FIELDS:
                if field not in sample:
                    raise KeyError(f"Missing required field '{field}' in sample {idx}")

            prompt = sample["prompt"]
            out_token_str = sample["out_token_str"]

            if not isinstance(prompt, str):
                raise TypeError(f"sample[{idx}]['prompt'] must be a string")
            if not isinstance(out_token_str, str):
                raise TypeError(f"sample[{idx}]['out_token_str'] must be a string")
            if len(out_token_str) == 0:
                raise ValueError(f"sample[{idx}]['out_token_str'] must be non-empty")
            prefix_prompt = sample.get("prefix_prompt")
            if prefix_prompt is not None and not isinstance(prefix_prompt, str):
                raise TypeError(f"sample[{idx}]['prefix_prompt'] must be a string")
            if prefix_prompt is None and not prompt.endswith(out_token_str):
                raise ValueError(
                    f"sample[{idx}] must provide prefix_prompt or satisfy "
                    "prompt.endswith(out_token_str)"
                )

    def _fill_missing_token_fields(self):
        for sample in self.preprocess_df_trans:
            if "out_token_id" not in sample:
                sample["out_token_id"] = self.tokenizer.encode(
                    sample["out_token_str"], add_special_tokens=False
                )

            if "in_token_str" in sample and "in_token_id" not in sample:
                sample["in_token_id"] = self.tokenizer.encode(
                    sample["in_token_str"], add_special_tokens=False
                )

            if "template_idx" not in sample:
                sample["template_idx"] = 0

            if "prefix_prompt" not in sample:
                sample["prefix_prompt"] = sample["prompt"][
                    : len(sample["prompt"]) - len(sample["out_token_str"])
                ]

    def _build_groups(self):
        groups = {}
        for idx, sample in enumerate(self.preprocess_df_trans):
            template_idx = sample.get("template_idx", 0)
            groups.setdefault(template_idx, []).append(idx)
        return list(groups.values())

    def _build_word_idx(self):
        end_idxs = []
        tgt_idxs = []
        src_idxs = []

        for i, sample in enumerate(self.preprocess_df_trans):
            full_len = int(self.attn_mask[i].sum().item())
            end_idxs.append(full_len - 2)
            tgt_idxs.append(full_len - 1)

            prefix_text = sample["prefix_prompt"]
            prefix_ids = self.tokenizer(
                prefix_text,
                add_special_tokens=self.prepend_bos,
            )["input_ids"]
            src_idxs.append(list(range(len(prefix_ids))))

        return {
            "end": torch.tensor(end_idxs, dtype=torch.long),
            "tgt": torch.tensor(tgt_idxs, dtype=torch.long),
            "src": src_idxs,
            "starts": torch.zeros(len(self.preprocess_df_trans), dtype=torch.long),
        }

    def _build_tokenized_prompts(self):
        tokenized_prompts = []
        for i in range(self.N):
            decoded = [self.tokenizer.decode(tok) for tok in self.toks[i]]
            tokenized_prompts.append("|".join(decoded))
        return tokenized_prompts

    def slice(self, key):
        return CustomTranslationDataset(
            samples=self.preprocess_df_trans[key],
            tokenizer=self.tokenizer,
            prepend_bos=self.prepend_bos,
            validate=False,
            auto_fill_token_ids=False,
        )

    def to(self, device):
        self.toks = self.toks.to(device)
        self.attn_mask = self.attn_mask.to(device)
        self.word_idx["end"] = self.word_idx["end"].to(device)
        self.word_idx["tgt"] = self.word_idx["tgt"].to(device)
        self.word_idx["starts"] = self.word_idx["starts"].to(device)
        return self

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.slice(key)
        if isinstance(key, int):
            return self.preprocess_df_trans[key]
        raise TypeError(f"Unsupported key type: {type(key)}")

    def __len__(self):
        return self.N


def build_custom_translation_dataset(
    samples: List[Dict],
    tokenizer,
    prepend_bos: bool = False,
    validate: bool = True,
    auto_fill_token_ids: bool = True,
):
    return CustomTranslationDataset(
        samples=samples,
        tokenizer=tokenizer,
        prepend_bos=prepend_bos,
        validate=validate,
        auto_fill_token_ids=auto_fill_token_ids,
    )
