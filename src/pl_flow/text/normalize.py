"""Training-compatible text normalization."""

import re

_TAG_RE = re.compile("<[^>\\n]*>")

_BRACED_CONTROL_RE = re.compile("\\{[^{}\\n]*\\}")

_WHITESPACE_RE = re.compile("\\s+")

_ALPHA_NUM_HYPHEN_RE = re.compile("(?<=[A-Za-z])\\s*-\\s*(?=\\d)|(?<=\\d)\\s*-\\s*(?=[A-Za-z])")

_NON_WORD_INTERNAL_HYPHEN_RE = re.compile("(?<!\\w)-|-(?!\\w)")


def normalize_kokoro_input_text(text: str) -> str:
    """
    在语言分段 / tokenizer 之前做统一输入清洗。

    规则:
    - 真实换行和字面 `\\n` -> 空格
    - `<...>` 标签、`{...}` 控制占位符 -> 空格
    - `#` / `\\` / `/` 控制符 -> 空格
    - 连续空白 -> 单空格
    """
    text = str(text)
    text = text.replace("\\n", " ").replace("\n", " ").replace("\r", " ")
    text = _TAG_RE.sub(" ", text)
    text = _BRACED_CONTROL_RE.sub(" ", text)
    text = text.replace("#", " ").replace("\\", " ").replace("/", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_kokoro_tokenizer_text(text: str) -> str:
    """
    在进入 Kokoro tokenizer / G2P 前做轻量符号归一化。

    规则:
    - 先执行 `normalize_kokoro_input_text`
    - `~` / `～` / `―` -> `..`
    - 非词内 `-` -> `..`，词内 `-` 保留给英文 G2P
    - `·` / `•` -> `;`
    - `<` / `《` / `【` / `[` / `{` / `（` / `『` / `「` -> `(`
    - `>` / `》` / `】` / `]` / `}` / `）` / `』` / `」` -> `)`
    - 连续空白 -> 单空格
    """
    text = normalize_kokoro_input_text(text)
    text = text.replace("％", "%").replace("＄", "$")
    replacements = {
        "~": "..",
        "～": "..",
        "―": "..",
        "·": ";",
        "•": ";",
        "<": "(",
        "《": "(",
        "【": "(",
        "[": "(",
        "{": "(",
        "（": "(",
        "『": "(",
        "「": "(",
        ">": ")",
        "》": ")",
        "】": ")",
        "]": ")",
        "}": ")",
        "）": ")",
        "』": ")",
        "」": ")",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = _ALPHA_NUM_HYPHEN_RE.sub(" ", text)
    text = _NON_WORD_INTERNAL_HYPHEN_RE.sub("..", text)
    return _WHITESPACE_RE.sub(" ", text).strip()
