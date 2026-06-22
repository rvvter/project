from __future__ import annotations

"""
Flask 系统入口文件

作用：
--------------------------------
1. 启动 Flask Web 服务
2. 提供网页页面
3. 提供 AI 文本检测 API
4. 加载训练好的模型
5. 校验用户输入
6. 返回预测结果

整体流程：
--------------------------------
用户输入文本
    ↓
前端发送 POST 请求
    ↓
Flask 接收请求
    ↓
文本合法性校验
    ↓
调用 TextDetector
    ↓
返回 JSON 检测结果
"""

# pathlib.Path
# 用于跨平台文件路径处理
from pathlib import Path

# Flask 核心模块
from flask import (
    Flask,          # Flask 应用对象
    jsonify,        # 返回 JSON 数据
    render_template,# 渲染 HTML 页面
    request,        # 获取 HTTP 请求
)

# 导入文本检测器
from src.models.predict import TextDetector

# 导入文本校验规则
from src.system.validation import (
    MAX_TEXT_LEN,       # 最大字符长度
    MIN_TEXT_LEN,       # 最小字符长度
    MIN_TOKEN_COUNT,    # 最少单词数
    validate_detect_text,
)

# =========================================================
# 1. 获取项目根目录
# =========================================================

# __file__
# 当前文件路径
#
# resolve()
# 获取绝对路径
#
# parents[2]
# 返回上两级目录
#
# 例如：
# src/web/app.py
#   ↑
#   ↑ parents[2]
# 项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]

# =========================================================
# 2. 模型文件路径
# =========================================================

# artifacts/models/detector_pipeline.joblib
#
# 保存训练好的 sklearn pipeline
MODEL_PATH = (
    BASE_DIR
    / "artifacts"
    / "models"
    / "detector_pipeline.joblib"
)

# =========================================================
# 3. 创建 Flask 应用
# =========================================================

# template_folder
# HTML 模板目录
#
# static_folder
# 静态资源目录（CSS/JS/图片）
app = Flask(
    __name__,

    template_folder=str(BASE_DIR / "templates"),

    static_folder=str(BASE_DIR / "static"),
)

# =========================================================
# 4. 设置请求大小限制
# =========================================================

# MAX_CONTENT_LENGTH
# 限制 HTTP 请求体最大大小
#
# 为什么 ×4？
#
# UTF-8 中：
# 一个中文字符可能占 4 字节
#
# +4096：
# 给 JSON 结构预留额外空间
app.config["MAX_CONTENT_LENGTH"] = (
    MAX_TEXT_LEN * 4 + 4096
)

# =========================================================
# 5. 全局模型变量
# =========================================================

# detector 初始为空
#
# 第一次请求时再加载模型
#
# 好处：
# 避免 Flask 启动时卡顿
detector: TextDetector | None = None

# =========================================================
# 6. 请求前自动执行
# =========================================================

@app.before_request
def ensure_model_loaded() -> None:
    """
    懒加载模型（Lazy Loading）

    作用：
    --------------------------------
    第一次请求到来时，
    才加载训练好的模型。

    优点：
    --------------------------------
    1. Flask 启动更快
    2. 避免程序启动阻塞
    3. 减少无意义加载
    """

    global detector

    # 如果模型尚未加载
    # 且模型文件存在
    if detector is None and MODEL_PATH.exists():

        # 加载模型
        detector = TextDetector(MODEL_PATH)

# =========================================================
# 7. 首页路由
# =========================================================

@app.get("/")
def index():
    """
    渲染主页面

    URL：
    --------------------------------
    GET /

    返回：
    --------------------------------
    index.html 页面
    """

    return render_template(
        "index.html",

        # 前端可显示：
        # “模型是否已准备好”
        model_ready=MODEL_PATH.exists(),

        # 前端表单校验参数
        min_text_len=MIN_TEXT_LEN,
        max_text_len=MAX_TEXT_LEN,
        min_token_count=MIN_TOKEN_COUNT,
    )

# =========================================================
# 8. 文本检测 API
# =========================================================

@app.post("/api/detect")
def detect():
    """
    AI 文本检测接口

    URL：
    --------------------------------
    POST /api/detect

    请求格式：
    --------------------------------
    {
        "text": "待检测文本"
    }

    返回格式：
    --------------------------------
    {
        "ok": true,
        "result": {...}
    }
    """

    global detector

    # =====================================================
    # 8.1 检查模型是否已加载
    # =====================================================

    if detector is None:

        return jsonify(
            {
                "ok": False,

                "error": "模型未找到，请先运行训练脚本。"
            }
        ), 400

    # =====================================================
    # 8.2 检查 Content-Type
    # =====================================================

    # request.is_json
    #
    # 检查请求头：
    # Content-Type: application/json
    if not request.is_json:

        return jsonify(
            {
                "ok": False,

                "error":
                    "请求 Content-Type 必须为 application/json。"
            }
        ), 415

    # =====================================================
    # 8.3 获取 JSON 数据
    # =====================================================

    # silent=True
    #
    # JSON 解析失败时不抛异常
    payload = request.get_json(silent=True)

    # JSON 无效
    if payload is None:

        return jsonify(
            {
                "ok": False,

                "error": "请求体必须是合法的 JSON。"
            }
        ), 400

    # 必须是 JSON 对象
    #
    # 正确：
    # {"text": "..."}
    #
    # 错误：
    # ["abc"]
    # 123
    if not isinstance(payload, dict):

        return jsonify(
            {
                "ok": False,

                "error": "请求体必须是 JSON 对象。"
            }
        ), 400

    # =====================================================
    # 8.4 文本合法性校验
    # =====================================================

    # validate_detect_text()
    #
    # 检查：
    # 1. 是否为空
    # 2. 长度是否合法
    # 3. token 数量是否合法
    checked = validate_detect_text(
        payload.get("text")
    )

    # 校验失败
    if not checked.ok:

        return jsonify(
            {
                "ok": False,

                "error": checked.error
            }
        ), 400

    # =====================================================
    # 8.5 调用模型预测
    # =====================================================

    # topn=8
    #
    # 返回最重要的8个特征
    result = detector.predict(
        checked.text,
        topn=8
    )

    # =====================================================
    # 8.6 返回结果
    # =====================================================

    return jsonify(
        {
            "ok": True,

            "result": result
        }
    )

# =========================================================
# 9. 请求体过大错误处理
# =========================================================

@app.errorhandler(413)
def request_too_large(_):
    """
    413 = Payload Too Large

    当请求超过 MAX_CONTENT_LENGTH 时触发
    """

    return jsonify(
        {
            "ok": False,

            "error":
                f"请求体过大，文本最多 {MAX_TEXT_LEN} 个字符。"
        }
    ), 413

# =========================================================
# 10. 程序入口
# =========================================================

if __name__ == "__main__":

    # host="127.0.0.1"
    # 仅本机访问
    #
    # port=5000
    # Flask 默认端口
    #
    # debug=True
    # 开启调试模式
    #
    # 功能：
    # 1. 自动热更新
    # 2. 显示详细错误信息
    app.run(
        host="127.0.0.1",

        port=5000,

        debug=True
    )