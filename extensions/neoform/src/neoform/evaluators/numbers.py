from __future__ import annotations

import re

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalize_number(token: str) -> str:
    cleaned = token.replace(",", "").strip()
    if not cleaned:
        return cleaned
    if "." in cleaned:
        value = float(cleaned)
        if value.is_integer():
            return str(int(value))
        return format(value, "g")
    return str(int(cleaned))


def extract_first_number(text: str) -> str | None:
    match = NUMBER_RE.search(text.replace(",", ""))
    return normalize_number(match.group()) if match else None


def extract_final_number(text: str) -> str | None:
    compact = text.replace(",", "")
    if "####" in compact:
        tail = compact.rsplit("####", 1)[-1]
        match = NUMBER_RE.search(tail)
        if match:
            return normalize_number(match.group())
    matches = NUMBER_RE.findall(compact)
    if not matches:
        return None
    return normalize_number(matches[-1])


def numbers_equal(left: str, right: str) -> bool:
    try:
        return float(left) == float(right)
    except ValueError:
        return left.strip() == right.strip()
