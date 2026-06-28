from __future__ import annotations

"""数据预处理模块：清洗、合并、划分、持久化。"""

import re
# 正则表达式模块：用于文本清洗（压缩空白等）

from dataclasses import dataclass
# dataclass：用于快速定义“只存数据”的结构体类

from pathlib import Path
# Path：面向对象的路径操作（比 os.path 更现代）

from typing import Tuple
# 类型提示：返回多个路径时使用

import pandas as pd
# pandas：核心数据处理库（DataFrame）

from sklearn.model_selection import train_test_split
# sklearn 工具：用于随机划分训练/验证/测试集


@dataclass
class SplitResult:
    """保存 train/test 两个数据集切分结果。"""

    train: pd.DataFrame
    # 训练集（含调参用的交叉验证）

    test: pd.DataFrame
    # 测试集


def clean_text(text: str) -> str:
    """统一文本格式：去首尾空白、压缩连续空白。"""

    if not isinstance(text, str):
        # 防御性编程：如果不是字符串（可能是 NaN/数字），直接返回空字符串
        return ""

    text = text.strip()
    # 去掉首尾空格、换行、tab 等

    text = text.replace("\u3000", " ")
    # 替换中文全角空格（常见于中文语料）

    text = re.sub(r"\s+", " ", text)
    # 把多个连续空白字符（空格/换行/tab）压缩成一个空格

    return text
    # 返回清洗后的文本


def load_and_merge(human_csv: Path, machine_csv: Path) -> pd.DataFrame:
    """读取人类/机器文本并合并为统一带标签数据表。"""

    human_df = pd.read_csv(human_csv)
    # 读取人类文本 CSV

    machine_df = pd.read_csv(machine_csv)
    # 读取机器生成文本 CSV

    # 保留统一字段名，便于后续流程固定读取。
    human_df = human_df.rename(columns={"text": "text"})
    machine_df = machine_df.rename(columns={"text": "text"})

    # 标签约定：人类=0，机器=1。
    human_df["label"] = 0
    # 人类样本打标签 0

    machine_df["label"] = 1
    # 机器样本打标签 1

    df = pd.concat(
        [human_df[["text", "label"]], machine_df[["text", "label"]]],
        ignore_index=True
    )
    # 合并数据：
    # - 只保留 text + label 两列
    # - ignore_index=True 重新生成索引

    df["text"] = df["text"].map(clean_text)
    # 对所有文本进行清洗（逐行 apply clean_text）

    df = df[df["text"].str.len() > 5]
    # 过滤掉过短文本（<=5字符通常信息量太低）

    df = df.dropna()
    # 删除空值（防止后续训练报错）

    df = df.reset_index(drop=True)
    # 重置索引，使其连续干净

    return df
    # 返回最终统一数据集


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.8,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitResult:
    """按指定比例分层切分数据，保持标签分布一致。"""

    if round(train_size + test_size, 6) != 1.0:
        raise ValueError("train/test 比例之和必须为 1.0")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    # 一次切分：
    # - train：80%（调参由交叉验证完成，无需单独验证集）
    # - test：20%（完全独立，最终评估用）
    # stratify：保证 label 分布一致（分类任务关键）

    return SplitResult(
        train=train_df.reset_index(drop=True),
        # 重置索引，避免拼接/切分后的乱序索引

        test=test_df.reset_index(drop=True),
        # 测试集
    )


def save_split(result: SplitResult, output_dir: Path) -> Tuple[Path, Path]:
    """将切分结果保存到磁盘，返回两个文件路径。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    # 如果目录不存在就创建（避免报错）

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    # 定义输出文件路径

    result.train.to_csv(train_path, index=False, encoding="utf-8-sig")
    # 保存训练集
    # utf-8-sig：Excel 兼容编码（避免中文乱码）

    result.test.to_csv(test_path, index=False, encoding="utf-8-sig")
    # 保存测试集

    return train_path, test_path
    # 返回路径，方便后续 pipeline 使用