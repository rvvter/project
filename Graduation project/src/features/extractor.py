from __future__ import annotations

"""文本特征工程模块：提取统计特征并拼接 TF-IDF 特征。"""

import math
import re
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List

import jieba
import numpy as np
import pandas as pd
import textstat
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


SENT_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")
TOKEN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+") #定义了一个正则表达式，用于提取中文/英文/数字

PUNCTUATIONS = set("，。！？；：、“”‘’（）()《》<>【】[],.!?;:'\"-") #定义了一个标点符号集合
EMOTIONAL_WORDS = {
    "积极": ["优秀", "成功", "快乐", "满意", "喜欢", "高兴", "赞"],
    "消极": ["失败", "痛苦", "担忧", "失望", "生气", "讨厌", "差"],
}

# 旧版英文特征名 -> 中文（兼容未重训的历史模型）
LEXICAL_FEATURE_LABELS_EN_TO_ZH = {
    "char_count": "字符数",
    "token_count": "词元数",
    "sentence_count": "句子数",
    "avg_sentence_len": "平均句长",
    "sentence_len_std": "句长标准差",
    "avg_word_len": "平均词长",
    "lexical_density": "词汇丰富度",
    "char_diversity": "字符多样性",
    "repeat_token_ratio": "重复词元占比",
    "punctuation_density": "标点密度",
    "special_density": "特殊符号密度",
    "stop_like_ratio": "短词元占比",
    "emotion_pos_count": "积极情感词数",
    "emotion_neg_count": "消极情感词数",
    "token_entropy": "词频熵",
    "top_token_freq": "最高词频",
    "readability_flesch": "可读性分数",
    "de_density": "的密度",
    "func_word_ratio": "功能词占比",
    "hapax_ratio": "单现词比例",
    "yule_k": "尤尔K值",
    "simpson_diversity": "辛普森多样性",
    "compression_ratio": "压缩比",
    "first_token_diversity": "句首词多样性",
    "word_len_cv": "词长波动",
    "digit_density": "数字密度",
    "avg_freq_rank": "平均词频排名",
}


def display_feature_name(name: str) -> str:
    """将特征名转为网页展示用的中文。"""

    if name in LEXICAL_FEATURE_LABELS_EN_TO_ZH:
        return LEXICAL_FEATURE_LABELS_EN_TO_ZH[name]
    if name.startswith("tfidf_"):
        return f"字符片段「{name[6:]}」"
    return name


def _safe_div(a: float, b: float) -> float:
    """安全除法，防止除零。"""

    return float(a) / float(b) if b else 0.0


def _sentences(text: str) -> List[str]:
    """按中英文标点进行句子切分。"""

    sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else [text.strip()] if text.strip() else [""]


def _tokens(text: str) -> List[str]:
    """使用 jieba 分词；英文/数字块统一小写，过滤空白与纯标点。"""

    text = text.strip()
    if not text:
        return []

    tokens: List[str] = []
    for word in jieba.lcut(text, cut_all=False):
        word = word.strip()
        if not word or word.isspace(): #除空
            continue
        if all(c in PUNCTUATIONS or c.isspace() for c in word): #除标点
            continue
        # regex二次提取
        parts = TOKEN_RE.findall(word.lower() if word.isascii() else word)
        for part in parts:
            if part:
                tokens.append(part.lower() if part.isascii() else part)
    return tokens


def _count_special(text: str) -> int:
    """统计非字母数字且非常见标点的特殊字符数量。"""

    return sum(1 for c in text if (not c.isalnum() and c not in PUNCTUATIONS and not c.isspace()))


def _safe_readability_score(text: str, token_count: int, sent_count: int) -> float:
    """计算可读性分数；若外部依赖失败则回退到近似分数。"""

    if not text.strip():
        return 0.0
    try:
        # textstat 在部分环境下会触发 nltk cmudict 资源问题，这里做容错。
        return float(textstat.flesch_reading_ease(text)) #计算英文可读性分数 准不准不重要，关键是ai文本和人类文本的分数差异作为稳定参数喂给模型
    except Exception:
        # 轻量回退：用句长与词长组合成粗略可读性分数，保持特征维度稳定。
        avg_sent_len = _safe_div(token_count, sent_count)
        avg_char_len = _safe_div(len(text), token_count)
        score = 100.0 - 1.8 * avg_sent_len - 12.0 * avg_char_len
        return float(max(min(score, 100.0), 0.0))


def _lexical_features(text: str) -> dict:
    """提取单条文本的统计/风格/熵相关特征。"""

    tokens = _tokens(text)  # 分词：jieba 中文切词 + 英文/数字块提取
    sents = _sentences(text)  # 分句：按中英文句号、问号等切分
    chars = len(text)  # 原文总字符数
    token_count = len(tokens)  # 词元总数
    sent_count = len(sents)  # 句子总数

    uniq = len(set(tokens))  # 不重复词元个数
    # 重复词元占比：出现次数>1 的词元数量之和 / 词元总数
    repeated_ratio = _safe_div(sum(v for v in Counter(tokens).values() if v > 1), token_count)
    # 平均词长；无词元时置 0，避免空列表求均值报错
    avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0.0
    avg_sent_len = _safe_div(token_count, sent_count)  # 平均每句词元数
    # 句长波动：各句词元数的标准差，反映句式是否整齐
    sent_len_std = np.std([len(_tokens(s)) for s in sents]) if sents else 0.0
    # 标点密度：标点字符数 / 总字符数
    punctuation_density = _safe_div(sum(1 for c in text if c in PUNCTUATIONS), chars)
    # 特殊符号密度：非常见标点且非空白字符 / 总字符数
    special_density = _safe_div(_count_special(text), chars)
    lexical_density = _safe_div(uniq, token_count)  # 词汇丰富度：去重词元 / 总词元
    char_diversity = _safe_div(len(set(text)), chars)  # 字符丰富度：去重字符 / 总字符
    # 短词元占比：长度<=1 的词元（近似停用/单字）/ 总词元
    stop_like_ratio = _safe_div(sum(1 for t in tokens if len(t) <= 1), token_count)

    pos_hits = sum(text.count(w) for w in EMOTIONAL_WORDS["积极"])  # 积极情感词命中次数
    neg_hits = sum(text.count(w) for w in EMOTIONAL_WORDS["消极"])  # 消极情感词命中次数

    freq_counter = Counter(tokens)  # 统计每个词元的出现频次
    top_freq = sorted(freq_counter.values(), reverse=True)  # 频次降序，用于取最高频
    entropy = 0.0  # 词频熵默认 0（空文本时保持维度稳定）
    if token_count > 0:
        probs = [v / token_count for v in freq_counter.values()]  # 各词元概率分布
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)  # Shannon 熵，衡量词汇分散程度

    func_words = {"的", "了", "是", "在", "和", "也", "就", "都", "把", "被",
                  "从", "对", "向", "与", "或", "但", "而", "且", "所", "之",
                  "其", "以", "及", "更", "很", "又", "才", "刚", "只", "个"}
    func_word_count = sum(1 for t in tokens if t in func_words)
    func_word_ratio = _safe_div(func_word_count, token_count)

    de_density = _safe_div(tokens.count("的"), token_count)

    hapax_count = sum(1 for v in freq_counter.values() if v == 1)
    hapax_ratio = _safe_div(hapax_count, len(freq_counter))

    m1 = sum(v for v in freq_counter.values())
    m2 = sum(v * v for v in freq_counter.values())
    yule_k = 1e4 * (m2 - m1) / (m1 * m1) if m1 > 1 else 0.0

    simpson = 0.0
    if token_count > 0:
        simpson = 1.0 - sum((v / token_count) ** 2 for v in freq_counter.values())

    raw_bytes = text.encode("utf-8", errors="ignore")
    comp_bytes = zlib.compress(raw_bytes, level=6)
    compression_ratio = _safe_div(len(comp_bytes), len(raw_bytes)) if raw_bytes else 0.0

    sent_starts: List[str] = []
    for s in sents:
        st = _tokens(s)
        if st:
            sent_starts.append(st[0])
    first_token_diversity = _safe_div(len(set(sent_starts)), len(sent_starts))

    token_lens = [len(t) for t in tokens]
    mean_wl = np.mean(token_lens) if token_lens else 0.0
    std_wl = np.std(token_lens) if token_lens else 0.0
    word_len_cv = _safe_div(std_wl, mean_wl)

    digit_chars = sum(1 for c in text if c.isdigit())
    digit_density = _safe_div(digit_chars, chars)

    rank_map: dict[str, int] = {}
    for idx, (tok, _) in enumerate(freq_counter.most_common()):
        rank_map[tok] = idx + 1
    total_rank = sum(rank_map.get(t, len(freq_counter)) for t in tokens)
    avg_freq_rank = _safe_div(total_rank, token_count)

    return {
        "字符数": chars,
        "词元数": token_count,
        "句子数": sent_count,
        "平均句长": avg_sent_len,
        "句长标准差": float(sent_len_std),
        "平均词长": float(avg_word_len),
        "词汇丰富度": lexical_density,
        "字符多样性": char_diversity,
        "重复词元占比": repeated_ratio,
        "标点密度": punctuation_density,
        "特殊符号密度": special_density,
        "短词元占比": stop_like_ratio,
        "积极情感词数": float(pos_hits),
        "消极情感词数": float(neg_hits),
        "词频熵": float(entropy),
        "最高词频": float(top_freq[0]) if top_freq else 0.0,
        "可读性分数": _safe_readability_score(text, token_count, sent_count),
        "功能词占比": func_word_ratio,
        "的密度": de_density,
        "单现词比例": hapax_ratio,
        "尤尔K值": yule_k,
        "辛普森多样性": simpson,
        "压缩比": compression_ratio,
        "句首词多样性": first_token_diversity,
        "词长波动": word_len_cv,
        "数字密度": digit_density,
        "平均词频排名": avg_freq_rank,
    }


@dataclass
class FeatureArtifacts:
    """预留的数据结构，便于后续扩展特征元信息。"""

    feature_names: List[str]


class HybridFeatureExtractor(BaseEstimator, TransformerMixin):
    """将手工特征与字符级 TF-IDF 特征拼接成统一向量，供 sklearn Pipeline 使用。"""

    def __init__(self, max_tfidf_features: int = 300, ngram_range: tuple = (2, 3)):
        """初始化特征器并配置 TF-IDF。"""

        self.max_tfidf_features = max_tfidf_features  # 保留的 TF-IDF 特征上限（默认 300 维）
        self.ngram_range = ngram_range
        self.tfidf = TfidfVectorizer(
            analyzer="char",  # 按字符切分，而非按词
            ngram_range=ngram_range,  # 提取 n-gram 字符片段
            max_features=max_tfidf_features,  # 仅保留语料中 TF-IDF 最高的前 N 个片段
            lowercase=True,  # 英文统一转小写，减少冗余维度
        )
        self._lexical_columns: List[str] = []  # 手工特征列名，fit 后写入，保证 transform 列顺序一致

    def fit(self, X: Iterable[str], y=None):
        """在训练语料上学习 TF-IDF 词典，并固定手工特征的列名顺序。"""

        texts = list(X)  # 将可迭代输入转为列表，便于多次遍历
        lexical_rows = [_lexical_features(t) for t in texts]  # 每条文本提取 N 维手工特征
        self._lexical_columns = list(lexical_rows[0].keys()) if lexical_rows else []  # 记录手工特征名（如 字符数、词元数）
        self.tfidf.fit(texts)  # 学习字符 n-gram 的 IDF 权重与词表
        return self  # 返回自身，符合 sklearn Transformer 约定

    def transform(self, X: Iterable[str]):
        """将文本列表转换为数值特征矩阵（手工特征 || TF-IDF 特征）。"""

        texts = list(X)  # 统一为列表输入
        lexical_rows = [_lexical_features(t) for t in texts]  # 对每条样本计算手工特征
        lex_df = pd.DataFrame(lexical_rows, columns=self._lexical_columns).fillna(0.0)  # 对齐列名并填充缺失为 0
        tfidf_matrix = self.tfidf.transform(texts).toarray()  # 稀疏矩阵转稠密数组，便于与手工特征拼接
        return np.hstack([lex_df.values, tfidf_matrix])  # 横向拼接：手工特征 + TF-IDF 特征

    def get_feature_names_out(self) -> List[str]:
        """返回与 transform 输出列一一对应的特征名称，用于解释与导出。"""

        tfidf_names = [f"tfidf_{n}" for n in self.tfidf.get_feature_names_out().tolist()]  # 为 n-gram 片段加前缀
        return self._lexical_columns + tfidf_names  # 手工特征名在前，TF-IDF 特征名在后
