from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


LANG_CODE_MAP = {
    "am": "Amharic",
    "ar": "Arabic",
    "ba": "Bashkir",
    "be": "Belarusian",
    "br": "Breton",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "el": "Greek (Modern)",
    "en": "English",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Western Frisian",
    "ga": "Irish",
    "gl": "Galician",
    "ha": "Hausa",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "ht": "Haitian Creole",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "ig": "Igbo",
    "ilo": "Iloko",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "ka": "Georgian",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "lg": "Ganda",
    "ln": "Lingala",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "ms": "Malay",
    "my": "Burmese",
    "ne": "Nepali",
    "nl": "Dutch",
    "or": "Odia (Oriya)",
    "pa": "Punjabi",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sl": "Slovenian",
    "so": "Somali",
    "sq": "Albanian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "th": "Thai",
    "tl": "Tagalog",
    "tn": "Tswana",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "wo": "Wolof",
    "xh": "Xhosa",
    "zh": "Chinese",
    "zu": "Zulu",
}

DIRECT_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a professional translator. Output ONLY the translation. "
        "Do not explain. Do not add notes. Do not add extra text."
    ),
}


def _map_lang(code: str) -> str:
    if code is None:
        return ""
    code = str(code).strip().lower()
    return LANG_CODE_MAP.get(code, code)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_parquet_folder(folder_path: str, pattern: str = "*.parquet") -> pd.DataFrame:
    folder = Path(folder_path)
    parquet_files = sorted(folder.rglob(pattern))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {folder} with pattern {pattern}")

    frames = []
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        df = df.copy()
        df["source_file"] = str(file_path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_context_block(row: Dict[str, Any]) -> str:
    parts = []
    for lang_key, text_key in [
        ("p1_lang", "p1_text"),
        ("p2_lang", "p2_text"),
        ("p3_lang", "p3_text"),
    ]:
        text = _clean_text(row.get(text_key))
        lang_code = _clean_text(row.get(lang_key))
        if not text or not lang_code:
            continue
        parts.append(f"{_map_lang(lang_code)} : {text}")
    return "\n".join(parts)


def build_multipivot_instruction(row: Dict[str, Any]) -> str:
    src_lang = _map_lang(row.get("src_lang", ""))
    trg_lang = _map_lang(row.get("trg_lang", ""))
    src_text = _clean_text(row.get("src_text"))
    ref_block = build_context_block(row)

    return (
        f"Your task is to translate the SOURCE sentence into {trg_lang} ONLY.\n\n"
        f"IMPORTANT:\n"
        f"- The final output MUST be written in {trg_lang}.\n"
        f"- Do NOT output text in any other language.\n"
        f"- Do NOT repeat the intermediate translations.\n"
        f"- DO NOT EXPLAIN.\n\n"
        f"SOURCE ({src_lang}):\n"
        f"{src_text}\n\n"
        f"Intermediate translations (for reference only, DO NOT output these languages):\n"
        f"{ref_block}\n\n"
        f"FINAL TRANSLATION ({trg_lang}):"
    )


def build_direct_instruction(row: Dict[str, Any]) -> str:
    src_lang = _map_lang(row.get("src_lang", ""))
    trg_lang = _map_lang(row.get("trg_lang", ""))
    src_text = _clean_text(row.get("src_text"))

    return (
        f"Translate the following text from {src_lang} into {trg_lang}:\n"
        f"{src_lang}: {src_text}"
    )


def build_messages(
    row: Dict[str, Any],
    prompt_type: str,
    include_target: bool = False,
    target_text: Optional[str] = None,
) -> List[Dict[str, str]]:
    prompt_type = prompt_type.lower()
    if prompt_type not in {"direct", "multipivot"}:
        raise ValueError("prompt_type must be one of: 'direct', 'multipivot'")

    if prompt_type == "direct":
        messages: List[Dict[str, str]] = [
            DIRECT_SYSTEM_PROMPT,
            {"role": "user", "content": build_direct_instruction(row)},
        ]
    else:
        messages = [
            {"role": "user", "content": build_multipivot_instruction(row)},
        ]

    if include_target:
        if target_text is None:
            target_text = _clean_text(row.get("trg_text"))
        messages = messages + [{"role": "assistant", "content": target_text}]

    return messages


def render_chat_prompt(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    add_generation_prompt: bool,
) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def build_prompt_record(
    row: Dict[str, Any],
    prompt_type: str,
    tokenizer=None,
    target_col: str = "trg_text",
) -> Dict[str, Any]:
    """
    Build one prompt record from a parquet row.

    If tokenizer is provided, also render:
    - prefix_prompt: prompt before assistant answer
    - prompt: full prompt with ground-truth assistant answer
    """
    row = dict(row)
    target_text = _clean_text(row.get(target_col))
    prefix_messages = build_messages(row, prompt_type, include_target=False)
    full_messages = build_messages(
        row,
        prompt_type,
        include_target=True,
        target_text=target_text,
    )

    record: Dict[str, Any] = {
        "prompt_type": prompt_type,
        "src_lang": _clean_text(row.get("src_lang")),
        "trg_lang": _clean_text(row.get("trg_lang")),
        "src_text": _clean_text(row.get("src_text")),
        "out_token_str": target_text,
        "instruction": prefix_messages[-1]["content"],
        "messages_prefix": prefix_messages,
        "messages_full": full_messages,
        "template_idx": 0,
        "source_file": row.get("source_file"),
    }

    for column in [
        "p1_lang",
        "p2_lang",
        "p3_lang",
        "p1_text",
        "p2_text",
    ]:
        if column in row:
            record[column] = row.get(column)

    if tokenizer is not None:
        record["prefix_prompt"] = render_chat_prompt(
            tokenizer,
            prefix_messages,
            add_generation_prompt=True,
        )
        record["prompt"] = render_chat_prompt(
            tokenizer,
            full_messages,
            add_generation_prompt=False,
        )

    return record


def build_prompt_records_from_folder(
    folder_path: str,
    prompt_type: str,
    tokenizer=None,
    target_col: str = "trg_text",
    pattern: str = "*.parquet",
) -> List[Dict[str, Any]]:
    df = read_parquet_folder(folder_path, pattern=pattern)
    return [
        build_prompt_record(
            row=row,
            prompt_type=prompt_type,
            tokenizer=tokenizer,
            target_col=target_col,
        )
        for row in df.to_dict(orient="records")
    ]


def build_prompt_dataframe_from_folder(
    folder_path: str,
    prompt_type: str,
    target_col: str = "trg_text",
    pattern: str = "*.parquet",
) -> pd.DataFrame:
    records = build_prompt_records_from_folder(
        folder_path=folder_path,
        prompt_type=prompt_type,
        tokenizer=None,
        target_col=target_col,
        pattern=pattern,
    )
    return pd.DataFrame(records)


def preview_folder(
    folder_path: str,
    prompt_type: str = "direct",
    pattern: str = "*.parquet",
    n: int = 2,
):
    df = build_prompt_dataframe_from_folder(
        folder_path=folder_path,
        prompt_type=prompt_type,
        pattern=pattern,
    )
    preview_cols = [
        "src_lang",
        "trg_lang",
        "src_text",
        "out_token_str",
        "instruction",
        "source_file",
    ]
    preview_cols = [col for col in preview_cols if col in df.columns]
    print(df[preview_cols].head(n).to_string(index=False))


if __name__ == "__main__":
    import fire

    fire.Fire(
        {
            "preview_folder": preview_folder,
            "build_prompt_dataframe_from_folder": build_prompt_dataframe_from_folder,
            "build_prompt_records_from_folder": build_prompt_records_from_folder,
        }
    )
