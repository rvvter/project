"""
mcp_servers/filesystem_server.py

提供文件系统访问的 MCP 服务器，用于读取复习笔记。

此服务器将用户的复习笔记暴露给通过 MCP 连接的 Agent。
作为独立进程运行，通过 stdio 传输进行通信。

暴露的工具：
    list_study_files()         : 列出有哪些笔记材料可用
    read_study_file(filename)  : 读取指定的笔记文件
    search_notes(query)        : 按关键词搜索相关段落

暴露的资源：
    notes://index              : 所有可用材料的摘要

安全防护：
    路径遍历攻击防护——Agent 无法读取笔记目录以外的文件。
"""

import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# ─────────────────────────────────────────────────────────────────────────────
# 服务器初始化
#
# FastMCP("名称") 创建一个 MCP 服务器实例。
# 名称会出现在 Agent Card 和 Langfuse 追踪中。
# 一行代码，FastMCP 处理所有协议细节。
# ─────────────────────────────────────────────────────────────────────────────

mcp = FastMCP("Filesystem Server")

# 复习笔记的根目录。
# 从 .env 读取，方便不修改代码更换路径。
NOTES_BASE = Path(os.getenv("NOTES_PATH", "study_materials/sample_notes"))


# ─────────────────────────────────────────────────────────────────────────────
# 工具定义
#
# @mcp.tool() 将普通 Python 函数转为 MCP 工具。
# 函数的：
#   - 名称 → Agent 调用的工具名
#   - docstring → 工具描述（LLM 读取此描述来决定是否使用该工具）
#   - 类型注解 → 参数 schema
#   - 返回类型注解 → 返回值描述
#
# FastMCP 负责序列化、传输和错误传播。
# 你只需写普通函数。
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def list_study_files() -> list[str]:
    """
    列出所有可用的复习笔记文件。

    返回相对于笔记目录的文件名列表。
    示例：['MySQL索引原理.md', '操作系统基础.md', '计算机网络.md']

    始终先调用此函数了解有哪些材料可用，
    再尝试读取特定文件。
    """
    if not NOTES_BASE.exists():
        return []

    files = sorted([
        str(f.relative_to(NOTES_BASE))
        for f in NOTES_BASE.rglob("*.md")
    ])
    return files


@mcp.tool()
def read_study_file(filename: str) -> str:
    """
    读取指定笔记文件的完整内容。

    参数：
        filename: 要读取的文件名，必须与 list_study_files()
                  返回的名称完全一致。示例：'MySQL索引原理.md'
    返回：
        文件的完整文本内容。
        如果文件不存在或路径无效，返回错误信息字符串，
        永不抛出异常，让 Agent 可以优雅地处理错误。
    """
    file_path = NOTES_BASE / filename

    # ── 安全防护：路径遍历攻击防御 ──────────────────────────────────
    # 没有此检查，Agent 可以调用：
    #   read_study_file("../../.env")
    # 读取你的 API 密钥。我们分别解析两个路径，
    # 验证请求的文件确实位于笔记目录内。
    try:
        resolved = file_path.resolve()
        resolved.relative_to(NOTES_BASE.resolve())
    except ValueError:
        return (
            f"错误：路径遍历尝试已被阻止（'{filename}'）。"
            "只能访问笔记目录内的文件。"
        )

    if not file_path.exists():
        available = list_study_files()
        return (
            f"错误：'{filename}' 不存在。"
            f"可用文件：{available}"
        )

    if file_path.suffix != ".md":
        return f"错误：只能访问 .md 文件，收到的是 '{file_path.suffix}'"

    try:
        content = file_path.read_text(encoding="utf-8")
        return content
    except (PermissionError, OSError) as e:
        return f"读取 '{filename}' 时出错：{e}"


@mcp.tool()
def search_notes(query: str) -> list[dict]:
    """
    在所有复习笔记中搜索关键词或短语。

    对所有 .md 文件执行不区分大小写的子串搜索。
    返回包含文件和行号上下文的匹配行。

    参数：
        query: 搜索词，不区分大小写。
               示例：'B+树'、'索引优化'、'进程调度'
    返回：
        匹配结果列表，每条包含键：
            'file':        相对文件名
            'line_number': 1-based 行号
            'line':        匹配的行文本（已去除首尾空白）
        最多返回 20 条结果，避免超出上下文窗口。
        无匹配时返回空列表。
    """
    if not NOTES_BASE.exists():
        return []

    results = []
    query_lower = query.lower()

    # 按排序顺序搜索，确保结果可复现
    for file_path in sorted(NOTES_BASE.rglob("*.md")):
        rel_path = str(file_path.relative_to(NOTES_BASE))
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except (UnicodeDecodeError, PermissionError, OSError):
            continue   # 跳过无法读取的文件

        for line_num, line in enumerate(lines, 1):
            if query_lower in line.lower():
                results.append({
                    "file": rel_path,
                    "line_number": line_num,
                    "line": line.strip(),
                })
                # 硬上限，防止上下文窗口溢出
                if len(results) >= 20:
                    return results

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 资源定义
#
# @mcp.resource("uri_pattern") 将函数转为 MCP 资源。
# URI 是 Agent 标识资源的方式，类似 URL。
# 资源是只读的，Agent 不能写入。
# ─────────────────────────────────────────────────────────────────────────────

@mcp.resource("notes://index")
def get_notes_index() -> str:
    """
    所有可用复习材料的索引。

    返回格式化的 Markdown 摘要，展示所有文件及其大小。
    Agent 可以读取此资源获取概览，无需加载每个文件。

    URI: notes://index
    """
    files = list_study_files()
    if not files:
        return "# 复习材料索引\n\n未找到复习材料。"

    lines = ["# 复习材料索引\n"]
    for filename in files:
        file_path = NOTES_BASE / filename
        try:
            size_kb = file_path.stat().st_size / 1024
            lines.append(f"- **{filename}**（{size_kb:.1f} KB）")
        except OSError:
            lines.append(f"- **{filename}**（大小未知）")

    lines.append(f"\n共 {len(files)} 个文件")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 入口
#
# 作为脚本运行时，服务器以 stdio 模式启动。
# LangGraph Agent 通过子进程 + 管道连接。
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    notes_path_str = str(NOTES_BASE.resolve())
    # 启动信息写入 stderr。stdout 是 stdio 传输下的
    # JSON-RPC 帧通道，写入 stdout 会破坏协议。
    print("[文件系统 MCP] 正在启动服务器", file=sys.stderr)
    print(f"[文件系统 MCP] 服务目录：{notes_path_str}", file=sys.stderr)
    print("[文件系统 MCP] 传输方式：stdio", file=sys.stderr)
    print("[文件系统 MCP] 等待连接...", file=sys.stderr)
    mcp.run()
