from __future__ import annotations

"""检测接口输入校验：与训练预处理规则对齐，并限制异常输入。"""

import re
from dataclasses import dataclass

from src.data.preprocess import clean_text
from src.features.extractor import PUNCTUATIONS, _tokens

# 与 preprocess.load_and_merge 中 `str.len() > 5` 保持一致。
MIN_TEXT_LEN = 6
MAX_TEXT_LEN = 10_000
MIN_TOKEN_COUNT = 1
MAX_SPECIAL_RATIO = 0.85

# 允许常见空白；其余 C0/C1 控制符拒绝。
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


@dataclass(frozen=True)
class TextValidationResult:
    ok: bool
    text: str = ""
    error: str = ""


def _count_special(text: str) -> int:
    return sum(1 for c in text if (not c.isalnum() and c not in PUNCTUATIONS and not c.isspace()))


def validate_detect_text(raw: object) -> TextValidationResult:
    """校验待检测文本，成功时返回清洗后的文本。"""

    if raw is None:
        return TextValidationResult(False, error="请输入待检测文本。")
    if not isinstance(raw, str):
        return TextValidationResult(False, error="text 字段必须是字符串。")

    if "\x00" in raw:
        return TextValidationResult(False, error="文本包含非法空字符。")
    if _CONTROL_RE.search(raw):
        return TextValidationResult(False, error="文本包含非法控制字符。")

    text = clean_text(raw)
    if not text:
        return TextValidationResult(False, error="请输入待检测文本。")

    length = len(text)
    if length < MIN_TEXT_LEN:
        return TextValidationResult(
            False,
            error=f"文本过短，至少需要 {MIN_TEXT_LEN} 个字符（当前 {length} 个）。",
        )
    if length > MAX_TEXT_LEN:
        return TextValidationResult(
            False,
            error=f"文本过长，最多 {MAX_TEXT_LEN} 个字符（当前 {length} 个）。",
        )

    tokens = _tokens(text)
    if len(tokens) < MIN_TOKEN_COUNT:
        return TextValidationResult(
            False,
            error=f"有效内容不足，至少需包含 {MIN_TOKEN_COUNT} 个中文/英文/数字词。",
        )

    special_ratio = _count_special(text) / length
    if special_ratio > MAX_SPECIAL_RATIO:
        return TextValidationResult(False, error="特殊字符占比过高，请输入正常文本。")

    return TextValidationResult(True, text=text)
