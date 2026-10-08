# Adapted from Kokoro's vocabulary and frontend integration (Apache-2.0).
# Modified for PL-Flow: fixed vocabulary, language routing and normalization.
# See licenses/Apache-2.0.txt and THIRD_PARTY.md.
"""Kokoro/Misaki frontend with the fixed training vocabulary."""

import warnings
from typing import Dict, List, Optional

from .normalize import normalize_kokoro_input_text, normalize_kokoro_tokenizer_text


def normalize_kokoro_phonemes(phonemes: str) -> str:
    """
    在 G2P 产出音素后追加一轮轻量归一化。

    规则:
    - `«` -> `(`
    - `»` -> `)`
    """
    return phonemes.replace("«", "(").replace("»", ")")


KOKORO_VOCAB: Dict[str, int] = {
    ";": 1,
    ":": 2,
    ",": 3,
    ".": 4,
    "!": 5,
    "?": 6,
    "—": 9,
    "…": 10,
    '"': 11,
    "(": 12,
    ")": 13,
    "“": 14,
    "”": 15,
    " ": 16,
    "̃": 17,
    "ʣ": 18,
    "ʥ": 19,
    "ʦ": 20,
    "ʨ": 21,
    "ᵝ": 22,
    "ꭧ": 23,
    "A": 24,
    "I": 25,
    "O": 31,
    "Q": 33,
    "S": 35,
    "T": 36,
    "W": 39,
    "Y": 41,
    "ᵊ": 42,
    "a": 43,
    "b": 44,
    "c": 45,
    "d": 46,
    "e": 47,
    "f": 48,
    "h": 50,
    "i": 51,
    "j": 52,
    "k": 53,
    "l": 54,
    "m": 55,
    "n": 56,
    "o": 57,
    "p": 58,
    "q": 59,
    "r": 60,
    "s": 61,
    "t": 62,
    "u": 63,
    "v": 64,
    "w": 65,
    "x": 66,
    "y": 67,
    "z": 68,
    "ɑ": 69,
    "ɐ": 70,
    "ɒ": 71,
    "æ": 72,
    "β": 75,
    "ɔ": 76,
    "ɕ": 77,
    "ç": 78,
    "ɖ": 80,
    "ð": 81,
    "ʤ": 82,
    "ə": 83,
    "ɚ": 85,
    "ɛ": 86,
    "ɜ": 87,
    "ɟ": 90,
    "ɡ": 92,
    "ɥ": 99,
    "ɨ": 101,
    "ɪ": 102,
    "ʝ": 103,
    "ɯ": 110,
    "ɰ": 111,
    "ŋ": 112,
    "ɳ": 113,
    "ɲ": 114,
    "ɴ": 115,
    "ø": 116,
    "ɸ": 118,
    "θ": 119,
    "œ": 120,
    "ɹ": 123,
    "ɾ": 125,
    "ɻ": 126,
    "ʁ": 128,
    "ɽ": 129,
    "ʂ": 130,
    "ʃ": 131,
    "ʈ": 132,
    "ʧ": 133,
    "ʊ": 135,
    "ʋ": 136,
    "ʌ": 138,
    "ɣ": 139,
    "ɤ": 140,
    "χ": 142,
    "ʎ": 143,
    "ʒ": 147,
    "ʔ": 148,
    "ˈ": 156,
    "ˌ": 157,
    "ː": 158,
    "ʰ": 162,
    "ʲ": 164,
    "↓": 169,
    "→": 171,
    "↗": 172,
    "↘": 173,
    "ᵻ": 177,
}

KOKORO_LANG_CODE_MAP = {"zh": "z", "en": "a", "ja": "j"}

KOKORO_MIXED_LANGS = set(KOKORO_LANG_CODE_MAP)


class KokoroG2P:
    """
    Kokoro 的 G2P (Grapheme-to-Phoneme) 封装，支持多语言。

    初始化时一次性创建所有指定语言的 G2P 后端。
    调用时同时返回音素字符串和对应的 token ids。

    支持语言:
        'a' = American English    pip install misaki[en]
        'b' = British English     pip install misaki[en]
        'j' = Japanese            pip install misaki[ja]
        'z' = Mandarin Chinese    pip install misaki[zh]
        'e' = Spanish             需要 espeak-ng
        'f' = French              需要 espeak-ng
        'h' = Hindi               需要 espeak-ng
        'i' = Italian             需要 espeak-ng
        'p' = Portuguese          需要 espeak-ng
    """

    ESPEAK_LANG_MAP = {"e": "es", "f": "fr-fr", "h": "hi", "i": "it", "p": "pt-br"}

    def __init__(
        self, lang_codes: List[str] = ("a", "j", "z"), vocab: Optional[Dict[str, int]] = None
    ):
        self.vocab = vocab or KOKORO_VOCAB
        self._backends: Dict[str, object] = {}
        self._backend_types: Dict[str, str] = {}
        self._lang_segmenter = None
        for lc in lang_codes:
            self._init_backend(lc)

    def _init_backend(self, lang_code: str):
        if lang_code in self._backends:
            return
        if lang_code in ("a", "b"):
            from misaki import en, espeak

            british = lang_code == "b"
            self._backends[lang_code] = en.G2P(
                british=british, fallback=espeak.EspeakFallback(british=british)
            )
            self._backend_types[lang_code] = "en"
        elif lang_code == "j":
            from misaki import ja

            self._backends[lang_code] = ja.JAG2P()
            self._backend_types[lang_code] = "ja"
        elif lang_code == "z":
            from misaki import zh

            self._backends[lang_code] = zh.ZHG2P()
            self._backend_types[lang_code] = "zh"
        elif lang_code in self.ESPEAK_LANG_MAP:
            from misaki import espeak

            self._backends[lang_code] = espeak.EspeakG2P(language=self.ESPEAK_LANG_MAP[lang_code])
            self._backend_types[lang_code] = "espeak"
        else:
            raise ValueError(f"不支持的语言代码: {lang_code}")

    def __call__(self, text: str, lang_code: str = "a") -> tuple:
        """
        文本 → (音素字符串, token ids 列表)。

        Args:
            text: 输入文本
            lang_code: 语言代码

        Returns:
            (phonemes, ids) 元组
            - phonemes: IPA 音素字符串
            - ids: token id 列表 (已含 BOS/EOS)
        """
        return self.g2p(text, lang_code)

    def g2p(self, text: str, lang_code: str = "a") -> tuple:
        """
        文本 → (音素字符串, token ids 列表)。

        Args:
            text: 输入文本
            lang_code: 语言代码

        Returns:
            (phonemes, ids) 元组
        """
        if lang_code not in self._backends:
            raise ValueError(
                f"语言 '{lang_code}' 未初始化，已初始化的语言: {list(self._backends.keys())}"
            )
        original_text = text
        text = normalize_kokoro_tokenizer_text(text)
        if self._backend_types[lang_code] != "en":
            text = " ".join(text.replace("%", " ").replace("$", " ").split())
        phonemes = normalize_kokoro_phonemes(self._text_to_phonemes(text, lang_code))
        missing = sorted({p for p in phonemes if p not in self.vocab})
        if missing:
            kept_phonemes = "".join((p for p in phonemes if p in self.vocab))
            warnings.warn(
                f"检测到未收录音素，已从 token ids 中跳过。\nlang_code: {lang_code}\noriginal_text: {original_text!r}\nnormalized_text: {text!r}\nphonemes: {phonemes!r}\nmissing_phonemes: {', '.join((repr(p) for p in missing))}\nkept_phonemes: {kept_phonemes!r}",
                stacklevel=2,
            )
        ids = [0, *(self.vocab[p] for p in phonemes if p in self.vocab), 0]
        return (phonemes, ids)

    def _get_lang_segmenter(self):
        if self._lang_segmenter is None:
            try:
                import LangSegment
            except ImportError as exc:
                raise RuntimeError(
                    "LangSegment is required for mixed-language G2P. Install it with: pip install langsegment-backup"
                ) from exc
            self._lang_segmenter = LangSegment
        return self._lang_segmenter

    def _mixed_filters(self, primary_lang: str) -> List[str]:
        if primary_lang not in KOKORO_MIXED_LANGS:
            raise ValueError(f"不支持的混合语言优先级: {primary_lang}")
        return [primary_lang, *(lang for lang in ("zh", "ja", "en") if lang != primary_lang)]

    def split_by_lang(
        self, text: str, primary_lang: str = "zh", threshold: float = 0.95
    ) -> List[tuple]:
        """
        使用 LangSegment 将文本切成中/日/英片段。

        非中日英片段直接丢弃。
        """
        text = normalize_kokoro_input_text(text)
        segmenter = self._get_lang_segmenter()
        segmenter.setfilters(self._mixed_filters(primary_lang))
        segmenter.setPriorityThreshold(threshold)
        segments = []
        for item in segmenter.getTexts(text):
            segment_text = item["text"].strip()
            if not segment_text:
                continue
            segment_lang = item["lang"]
            if segment_lang in KOKORO_MIXED_LANGS:
                segments.append((segment_lang, segment_text))
        return segments

    def mixed_g2p(self, text: str, primary_lang: str = "zh", threshold: float = 0.95) -> tuple:
        """
        混合中/日/英文本 → (音素字符串, token ids 列表)。

        每个语言片段单独 G2P，再去掉片段级 BOS/EOS 后合并为一条序列。
        """
        phonemes = ""
        ids = [0]
        space_id = self.vocab[" "]
        for lang, segment_text in self.split_by_lang(text, primary_lang, threshold):
            (segment_phonemes, segment_ids) = self.g2p(segment_text, KOKORO_LANG_CODE_MAP[lang])
            segment_inner_ids = segment_ids[1:-1]
            if not segment_phonemes or not segment_inner_ids:
                continue
            if ids[-1] != 0 and ids[-1] != space_id and (segment_inner_ids[0] != space_id):
                phonemes += " "
                ids.append(space_id)
            phonemes += segment_phonemes
            ids.extend(segment_inner_ids)
        ids.append(0)
        return (phonemes, ids)

    def _text_to_phonemes(self, text: str, lang_code: str) -> str:
        g2p = self._backends[lang_code]
        if self._backend_types[lang_code] == "en":
            (_, tokens) = g2p(text)
            return "".join(
                (t.phonemes + (" " if t.whitespace else "") for t in tokens if t.phonemes)
            ).strip()
        else:
            (phonemes, _) = g2p(text)
            return phonemes

    def tokenize(self, text: str, lang_code: str = "a") -> list:
        """
        (仅英语) 返回 misaki 的 MToken 列表，保留词级信息和时间戳槽位。
        """
        if lang_code not in self._backends:
            raise ValueError(f"语言 '{lang_code}' 未初始化")
        if self._backend_types[lang_code] != "en":
            raise NotImplementedError("tokenize 仅支持英语 (lang_code='a' 或 'b')")
        text = normalize_kokoro_tokenizer_text(text)
        (_, tokens) = self._backends[lang_code](text)
        return tokens

    @property
    def languages(self) -> List[str]:
        """已初始化的语言代码列表。"""
        return list(self._backends.keys())
