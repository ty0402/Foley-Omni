import json
import os
import re
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _csv_cell_to_id_str(cell) -> Optional[str]:
    """Normalize a dataframe cell to a non-empty id string, or None if absent."""
    if cell is None:
        return None
    try:
        if pd.isna(cell):
            return None
    except TypeError:
        pass
    if isinstance(cell, (int, np.integer)):
        return str(int(cell))
    if isinstance(cell, (float, np.floating)):
        f = float(cell)
        if np.isfinite(f) and f == int(f):
            return str(int(f))
    s = str(cell).strip()
    return s or None


def _sanitize_output_stem(s: str, max_len: int = 200) -> str:
    """Create a filesystem-safe filename stem."""
    s = s.strip()
    if not s:
        return "unnamed"
    s = re.sub(r'[\\/:*?"<>|\x00\r\n]', "_", s)
    s = s.strip("._ ") or "unnamed"
    return s[:max_len]


def _output_stems_and_source_ids_from_dataframe(df: pd.DataFrame) -> Tuple[List[Optional[str]], List[Optional[str]]]:
    """Build per-row output stems from the optional `id` column."""
    n = len(df)
    if n == 0 or "id" not in df.columns:
        return [None] * n, [None] * n

    raw_ids: List[Optional[str]] = []
    for i in range(n):
        raw_ids.append(_csv_cell_to_id_str(df["id"].iloc[i]))

    counts = Counter(r for r in raw_ids if r is not None)
    used_stems: set = set()
    output_stems: List[Optional[str]] = []
    source_ids: List[Optional[str]] = []

    for i, rid in enumerate(raw_ids):
        if rid is None:
            output_stems.append(None)
            source_ids.append(None)
            continue
        base = _sanitize_output_stem(rid)
        stem = f"{base}__row{i}" if counts[rid] > 1 else base
        dup = 0
        while stem in used_stems:
            dup += 1
            stem = f"{base}__row{i}__{dup}"
        used_stems.add(stem)
        output_stems.append(stem)
        source_ids.append(rid)

    return output_stems, source_ids


def _load_dataframe_from_json_path(path: str) -> pd.DataFrame:
    """Load prompts from json/jsonl into a dataframe compatible with CSV/TSV input."""
    _, ext = os.path.splitext(path.lower())
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif ext == ".jsonl":
        payload = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported json extension: {ext}")

    records = []
    if isinstance(payload, dict):
        for _, value in payload.items():
            row = {"text_prompt": "", "id": ""}
            if isinstance(value, dict):
                row["text_prompt"] = value.get("text_prompt", value.get("resp", value.get("prompt", "")))
                row["id"] = value.get("id", "")
            elif isinstance(value, str):
                row["text_prompt"] = value
            else:
                raise ValueError(f"Unsupported JSON value type: {type(value)}")
            records.append(row)
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"JSON list element at index {i} is not an object.")
            records.append({
                "text_prompt": item.get("text_prompt", item.get("resp", item.get("prompt", ""))),
                "id": item.get("id", ""),
            })
    else:
        raise ValueError("Unsupported JSON root type. Expect object or array.")

    df = pd.DataFrame(records)
    if "text_prompt" not in df.columns:
        df["text_prompt"] = ""
    if "id" not in df.columns:
        df["id"] = ""
    return df.fillna("")


def validate_and_process_user_prompt(
    text_prompt: str,
    image_path: str = None,
    mode: str = "vt2a",
) -> Tuple[list, list, list, list]:
    del image_path, mode

    if not isinstance(text_prompt, str):
        raise ValueError("User input must be a string")

    text_prompt = text_prompt.strip()
    if os.path.isfile(text_prompt):
        _, ext = os.path.splitext(text_prompt.lower())
        if ext == ".csv":
            df = pd.read_csv(text_prompt).fillna("")
        elif ext == ".tsv":
            df = pd.read_csv(text_prompt, sep="\t").fillna("")
        elif ext in [".json", ".jsonl"]:
            df = _load_dataframe_from_json_path(text_prompt)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Only .csv, .tsv, .json and .jsonl are allowed.")

        if "text_prompt" not in df.columns:
            raise ValueError("Missing required `text_prompt` column in prompt file.")

        text_prompts = list(df["text_prompt"])
        image_paths = [None] * len(text_prompts)
        output_stems, source_row_ids = _output_stems_and_source_ids_from_dataframe(df)
        assert len(output_stems) == len(text_prompts) == len(source_row_ids)
    else:
        text_prompts = [text_prompt]
        image_paths = [None]
        output_stems = [None]
        source_row_ids = [None]

    return text_prompts, image_paths, output_stems, source_row_ids


def format_prompt_for_filename(text: str) -> str:
    no_tags = re.sub(r"<.*?>", "", text)
    safe = no_tags.replace(" ", "_").replace("/", "_")
    return safe[:50]
