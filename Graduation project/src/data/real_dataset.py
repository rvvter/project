from __future__ import annotations 

"""真实数据集构建模块：从多种格式中抽取人类/机器文本。"""  

import ast  # 用于安全解析字符串形式的 Python 字面量（如 "['a','b']"）
import json  # 用于处理 JSON / JSONL 文件
from pathlib import Path  # 更现代的路径处理方式（替代 os.path）
from typing import Iterable, List  # 类型注解：可迭代对象 / 列表
import pandas as pd  # 用于读取 CSV / 数据处理

from src.data.preprocess import clean_text  # 自定义文本清洗函数（核心依赖）

# -----------------------------
# 文本字段候选名（用于兼容不同数据集）
# -----------------------------
TEXT_CANDIDATE_COLUMNS = [
    "text",
    "content",
    "body",
    "answer",
    "response",
    "article",
    "passage",
]


# =============================
# 1. 将任意字段转成“文本列表”
# =============================
def _to_text_list(value) -> List[str]:
    """
    将 JSON/CSV 字段值统一转为「字符串列表」。

    为什么要这样做：
    - 有些字段是 list
    - 有些字段是字符串
    - 有些字段是 "['a','b']" 这种字符串化 list
    """

    if value is None:
        return []  # 空值直接返回空列表（无样本）

    if isinstance(value, str):  # 如果是字符串
        raw = value.strip()  # 去掉前后空格

        if not raw:
            return []  # 空字符串 → 不生成样本

        # 如果字符串看起来像 list / tuple
        if (raw.startswith("[") and raw.endswith("]")) or (
            raw.startswith("(") and raw.endswith(")")
        ):
            try:
                parsed = ast.literal_eval(raw)  # 安全解析字符串为 Python 对象

                if isinstance(parsed, (list, tuple)):
                    # 拆成多个文本样本，并过滤空字符串
                    return [str(x) for x in parsed if str(x).strip()]

            except Exception:
                pass  # 解析失败 → 当普通字符串处理

        return [raw]  # 普通字符串 → 直接作为一个样本

    if isinstance(value, (list, tuple)):
        # 如果本来就是 list/tuple（最常见情况）
        return [str(x) for x in value if str(x).strip()]  # 过滤空值

    return [str(value)]  # 兜底：数字 / 其他类型转字符串


# =============================
# 2. 读取 txt / md 文件
# =============================
def _read_txt_like(path: Path) -> List[str]:
    """读取 txt/md，按行提取文本。"""

    # 读取文件内容 → 按行切分
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    # 每一行都做清洗，并过滤空行
    return [clean_text(x) for x in lines if clean_text(x)]


# =============================
# 3. 读取 JSON 文件
# =============================
def _read_json(path: Path) -> List[dict]:
    """读取 JSON 文件并统一返回字典列表。"""

    # 读取整个 JSON 文件
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))

    if isinstance(data, list):
        # 如果是 list[dict]
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        # 如果是单个 dict → 包装成 list
        return [data]

    return []


# =============================
# 4. 读取 JSONL 文件
# =============================
def _read_jsonl(path: Path) -> List[dict]:
    """读取 JSONL 文件（每行一个 JSON）"""

    rows = []

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()  # 去空格

        if not line:
            continue  # 跳过空行

        try:
            obj = json.loads(line)  # 每行解析 JSON

            if isinstance(obj, dict):
                rows.append(obj)  # 只保留 dict

        except Exception:
            continue  # 忽略坏数据行

    return rows


# =============================
# 5. 从单条 record 中抽取 human / machine
# =============================
def _extract_from_record(
    record: dict,
    label_hint: str | None = None
) -> tuple[List[str], List[str]]:
    """
    从单条记录中抽取 human / machine 文本
    """

    human_items: List[str] = []  # human 文本容器
    machine_items: List[str] = []  # machine 文本容器

    # -----------------------------
    # 5.1 HC3 等数据中的 human 字段
    # -----------------------------
    for key in ["human_answers", "human", "human_text", "human_response"]:
        if key in record:
            human_items.extend(_to_text_list(record.get(key)))

    # -----------------------------
    # 5.2 machine / GPT 字段
    # -----------------------------
    for key in ["chatgpt_answers", "machine", "machine_text", "ai_response", "gpt_answers"]:
        if key in record:
            machine_items.extend(_to_text_list(record.get(key)))

    # -----------------------------
    # 5.3 通用字段 fallback
    # -----------------------------
    generic = []

    for key in TEXT_CANDIDATE_COLUMNS:
        if key in record:
            generic.extend(_to_text_list(record.get(key)))
            break  # 只取第一个匹配字段

    # -----------------------------
    # 5.4 label 辅助分流
    # -----------------------------
    row_label = str(record.get("label", "")).strip().lower()

    if generic:
        # 如果是 human
        if label_hint == "human" or row_label in {"0", "human"}:
            human_items.extend(generic)

        # 如果是 machine
        elif label_hint == "machine" or row_label in {"1", "machine", "ai", "chatgpt"}:
            machine_items.extend(generic)

    return human_items, machine_items


# =============================
# 6. 收集目录所有文件
# =============================
def _gather_files(directory: Path) -> List[Path]:
    """递归收集支持的语料文件。"""

    if not directory.exists():
        return []

    exts = {".txt", ".md", ".csv", ".json", ".jsonl"}

    # 递归遍历所有子目录
    return [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    ]


# =============================
# 7. 从目录加载文本
# =============================
def _load_texts_from_directory(
    directory: Path,
    label_hint: str | None
) -> tuple[List[str], List[str]]:
    """
    从目录加载 human / machine 文本
    """

    human_all: List[str] = []   # human 总列表
    machine_all: List[str] = []  # machine 总列表

    for file_path in _gather_files(directory):

        suffix = file_path.suffix.lower()

        # -------------------------
        # txt / md 文件
        # -------------------------
        if suffix in {".txt", ".md"}:

            texts = _read_txt_like(file_path)

            # 目录级别标签直接决定归类
            if label_hint == "human":
                human_all.extend(texts)

            elif label_hint == "machine":
                machine_all.extend(texts)

            continue

        # -------------------------
        # CSV 文件
        # -------------------------
        if suffix == ".csv":

            df = pd.read_csv(file_path)

            if "label" in df.columns:
                # 如果 CSV 自带 label → 逐行解析
                for _, row in df.iterrows():
                    rec = row.to_dict()
                    h, m = _extract_from_record(rec, label_hint=None)
                    human_all.extend(h)
                    machine_all.extend(m)

            else:
                # 没 label → 用目录 hint
                text_col = next(
                    (c for c in df.columns if c.lower() in TEXT_CANDIDATE_COLUMNS),
                    None
                )

                if text_col:
                    texts = [
                        clean_text(str(x))
                        for x in df[text_col].tolist()
                        if clean_text(str(x))
                    ]

                    if label_hint == "human":
                        human_all.extend(texts)
                    elif label_hint == "machine":
                        machine_all.extend(texts)

            continue

        # -------------------------
        # JSON / JSONL
        # -------------------------
        records = _read_json(file_path) if suffix == ".json" else _read_jsonl(file_path)

        for rec in records:
            h, m = _extract_from_record(rec, label_hint=label_hint)
            human_all.extend(h)
            machine_all.extend(m)

    return human_all, machine_all


# =============================
# 8. 清洗 + 去重 + 过滤
# =============================
def _normalize_texts(texts: Iterable[str], min_len: int = 20) -> List[str]:
    """
    清洗 + 去重 + 长度过滤
    """

    out = []  # 输出列表
    seen = set()  # 去重集合

    for t in texts:

        # 清洗文本
        nt = clean_text(str(t))

        # 长度过滤
        if len(nt) < min_len:
            continue

        # 去重
        if nt in seen:
            continue

        seen.add(nt)
        out.append(nt)

    return out


# =============================
# 9. 构建最终数据集
# =============================
def build_real_dataset(
    hc3_dir: Path | None,
    human_dir: Path | None,
    machine_dir: Path | None,
    output_dir: Path,
    min_len: int = 20,
    max_per_class: int | None = None,
) -> tuple[Path, Path]:
    """
    构建 human / machine 数据集 CSV
    """

    human_texts: List[str] = []
    machine_texts: List[str] = []

    # HC3 数据
    if hc3_dir and hc3_dir.exists():
        h, m = _load_texts_from_directory(hc3_dir, label_hint=None)
        human_texts.extend(h)
        machine_texts.extend(m)

    # human 数据
    if human_dir and human_dir.exists():
        h, _ = _load_texts_from_directory(human_dir, label_hint="human")
        human_texts.extend(h)

    # machine 数据
    if machine_dir and machine_dir.exists():
        _, m = _load_texts_from_directory(machine_dir, label_hint="machine")
        machine_texts.extend(m)

    # 清洗 + 去重
    human_final = _normalize_texts(human_texts, min_len=min_len)
    machine_final = _normalize_texts(machine_texts, min_len=min_len)

    # 控制最大样本数
    if max_per_class is not None and max_per_class > 0:
        human_final = human_final[:max_per_class]
        machine_final = machine_final[:max_per_class]

    # 防止空数据
    if not human_final or not machine_final:
        raise ValueError(
            "真实数据构建失败：至少一类样本为空。"
        )

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出 CSV
    human_csv = output_dir / "human_texts.csv"
    machine_csv = output_dir / "machine_texts.csv"

    pd.DataFrame({"text": human_final}).to_csv(human_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame({"text": machine_final}).to_csv(machine_csv, index=False, encoding="utf-8-sig")

    return human_csv, machine_csv