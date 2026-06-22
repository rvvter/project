from __future__ import annotations

"""命令行脚本：将真实语料构建为标准训练输入（默认仅 HC3）。"""

import argparse
from pathlib import Path
import sys

# 获取项目根目录路径（当前文件向上两级目录）
ROOT = Path(__file__).resolve().parents[1]

# 将项目根目录加入 Python 模块搜索路径
# 目的是保证可以正常导入 src 包
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 导入核心数据构建函数
from src.data.real_dataset import build_real_dataset


def parse_args() -> argparse.Namespace:
    """定义并解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="从 HC3 等语料构建 human_texts.csv / machine_texts.csv"
    )

    # HC3 数据目录路径
    parser.add_argument(
        "--hc3-dir",
        type=str,
        default="data/raw_sources/hc3",
        help="HC3 目录（内含 .jsonl，需含 human_answers / chatgpt_answers）",
    )

    # 额外人类语料目录路径（可选）
    parser.add_argument(
        "--human-dir",
        type=str,
        default="",
        help="可选：额外人类语料目录；留空则不使用",
    )

    # 额外机器语料目录路径（可选）
    parser.add_argument(
        "--machine-dir",
        type=str,
        default="",
        help="可选：额外机器语料目录；留空则不使用",
    )

    # 输出目录路径
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="输出目录",
    )

    # 文本最小长度过滤阈值
    parser.add_argument(
        "--min-len",
        type=int,
        default=20,
        help="文本最小长度过滤阈值",
    )

    # 每类最大样本数限制
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="每类最大样本数（0 表示不限制）",
    )

    # 解析命令行参数并返回
    return parser.parse_args()


def _resolve_optional_dir(path_str: str) -> Path | None:
    """将可选路径字符串转换为 Path，如果为空或不存在则返回 None。"""

    # 如果路径为空字符串或只包含空白字符，则认为未提供该路径
    if not path_str or not path_str.strip():
        return None

    # 拼接为项目根目录下的绝对路径
    path = ROOT / path_str.strip()

    # 如果路径存在则返回，否则返回 None
    return path if path.exists() else None


def main() -> None:
    """主流程：执行数据构建任务。"""

    # 解析命令行参数
    args = parse_args()

    # 构造 HC3 数据目录路径
    hc3_dir = ROOT / args.hc3_dir

    # 解析可选的人类语料目录
    human_dir = _resolve_optional_dir(args.human_dir)

    # 解析可选的机器语料目录
    machine_dir = _resolve_optional_dir(args.machine_dir)

    # 构造输出目录路径
    output_dir = ROOT / args.output_dir

    # 将 0 转换为 None，表示不限制样本数量
    max_per_class = args.max_per_class if args.max_per_class > 0 else None

    # 用于记录实际使用的数据源
    sources = []

    # 检查 HC3 数据是否存在
    if hc3_dir.exists():
        sources.append(f"HC3: {hc3_dir}")

    # 检查是否提供人类语料
    if human_dir:
        sources.append(f"人类语料: {human_dir}")

    # 检查是否提供机器语料
    if machine_dir:
        sources.append(f"机器语料: {machine_dir}")

    # 如果没有任何数据源则直接报错
    if not sources:
        raise FileNotFoundError(
            "未找到任何可用数据源。请确认 data/raw_sources/hc3 存在且内含 .jsonl 文件。"
        )

    # 打印当前使用的数据源
    print("使用数据源:")
    for s in sources:
        print(" -", s)

    # 调用核心函数构建数据集
    human_csv, machine_csv = build_real_dataset(
        hc3_dir=hc3_dir if hc3_dir.exists() else None,
        human_dir=human_dir,
        machine_dir=machine_dir,
        output_dir=output_dir,
        min_len=args.min_len,
        max_per_class=max_per_class,
    )

    # 输出生成结果路径
    print("真实数据已生成:")
    print(" -", human_csv)
    print(" -", machine_csv)


# 当脚本被直接运行时执行主函数
if __name__ == "__main__":
    main()