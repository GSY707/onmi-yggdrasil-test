from __future__ import annotations

import re
from typing import Any

from .data import ANSWER_LABELS

PROMPT_CONTRACT_VERSION = "yggdrasil.v2-a.prompt.v3"

SYSTEM_PROMPT = (
    "You execute short symbolic register programs exactly. COPY source -> target "
    "writes the source symbol into target and leaves source unchanged. SWAP a <-> b "
    "exchanges both symbols. Never change instruction order or invent steps. "
    "Valid answer symbols are A, B, C, D, E, F, G, H, I, J."
)

FINAL_ANSWER_PATTERN = re.compile(
    r"(?:FINAL(?:\s+value(?:\s+of(?:\s+the)?\s+[a-z]+(?:\s+register)?)?)?"
    r"|final\s+value(?:\s+of(?:\s+the)?\s+[a-z]+(?:\s+register)?)?|answer)"
    r"\s*(?::|=|\bis\b)\s*([a-z]+)",
    flags=re.IGNORECASE,
)
REGISTER_ANSWER_PATTERN = re.compile(
    r"\b(?:symbol|register)\b[^\n]{0,50}?\s*(?::|=|\bis\b)\s*([a-z]+)",
    flags=re.IGNORECASE,
)


def build_messages(
    example: dict[str, Any],
    mode: str,
    demonstrations: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    demonstrations = demonstrations or []
    if mode not in {"direct", "answer_only", "text_cot", "encoder"}:
        raise ValueError(f"Unknown prompt mode: {mode}")

    if mode == "direct":
        instruction = "Return only the final symbol (one uppercase letter A-J), with no explanation."
    elif mode == "answer_only":
        instruction = "Follow the examples and return only `FINAL: <one uppercase letter A-J>`."
    elif mode in {"text_cot", "encoder"}:
        instruction = "Track all three registers after every listed step, then write `FINAL: <one uppercase letter A-J>`. Do not add steps."

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for demo in demonstrations:
        messages.append({"role": "user", "content": f"{instruction}\n\n{demo['question']}"})
        if mode in {"text_cot", "encoder"}:
            response = demo["trace_text"]
        else:
            response = f"FINAL: {demo['answer']}"
        messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": f"{instruction}\n\n{example['question']}"})
    return messages


def parse_answer(text: str) -> str | None:
    explicit = [
        match.group(1)
        for pattern in (FINAL_ANSWER_PATTERN, REGISTER_ANSWER_PATTERN)
        for match in pattern.finditer(text)
    ]
    for candidate in reversed(explicit):
        label = candidate.upper()
        if label in ANSWER_LABELS:
            return label
    stripped = text.strip()
    labels = "|".join(ANSWER_LABELS)
    bare = re.fullmatch(
        rf"(?:the\s+answer\s+is\s+)?({labels})[.!]?",
        stripped,
        flags=re.IGNORECASE,
    )
    if bare:
        label = bare.group(1).upper()
        return label if label in ANSWER_LABELS else None
    words = [word.upper() for word in re.findall(r"[A-Za-z]+", stripped)]
    if len(words) == 1 and words[0] in ANSWER_LABELS:
        return words[0]
    return None


def has_complete_answer(text: str, mode: str) -> bool:
    if FINAL_ANSWER_PATTERN.search(text) or REGISTER_ANSWER_PATTERN.search(text):
        return True
    if mode == "text_cot":
        return False
    return bool(
        re.fullmatch(
            rf"(?:the\s+answer\s+is\s+)?(?:{'|'.join(ANSWER_LABELS)})[.!]?",
            text.strip(),
            flags=re.IGNORECASE,
        )
    )
