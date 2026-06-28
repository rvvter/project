from __future__ import annotations

"""
模型推理模块：加载训练好的机器学习流水线，
对输入文本进行预测，并返回分类结果与局部可解释信息。
"""

# Path 用于处理文件路径
from pathlib import Path

# Dict、List 用于类型注解
from typing import Dict, List

# joblib 用于加载训练好的 sklearn 模型
import joblib

# numpy 用于数组运算
import numpy as np

# 导入自定义特征提取器
from src.features.extractor import (
    HybridFeatureExtractor,
    display_feature_name,
)


class TextDetector:
    """
    文本检测器。

    功能：
    1. 加载训练好的 pipeline
    2. 对文本进行预测
    3. 输出预测概率
    4. 输出局部解释（哪些特征影响最大）
    """

    def __init__(self, model_path: Path):
        """
        初始化检测器。

        参数：
            model_path: 训练完成后导出的模型路径
        """

        # =========================
        # 加载训练好的完整 pipeline
        # =========================
        #
        # pipeline 内部一般包含：
        #
        # 1. features -> 特征提取器
        # 2. select   -> 特征选择器
        # 3. clf      -> 分类器
        #
        # 训练时已经保存成 .joblib 文件
        #
        self.pipeline = joblib.load(model_path)

        # =========================
        # 取出 pipeline 中的各个模块
        # =========================

        # 特征提取器
        self.features: HybridFeatureExtractor = (
            self.pipeline.named_steps["features"]
        )

        # 特征选择器
        self.selector = self.pipeline.named_steps["select"]

        # 分类模型
        self.clf = self.pipeline.named_steps["clf"]

    def predict(self, text: str, topn: int = 8) -> Dict:
        """
        对单条文本进行预测。

        参数：
            text : 用户输入文本
            topn : 返回前几个重要特征

        返回：
            Dict 格式结果，例如：

            {
                "label": 1,
                "label_name": "疑似机器生成",
                "probability_machine": 0.93,
                "confidence": 0.93,
                "top_features": [...]
            }
        """

        # ==================================================
        # 第一步：预测“机器生成”的概率
        # ==================================================
        #
        # predict_proba 返回二维数组：
        #
        # [
        #   [人类概率, 机器概率]
        # ]
        #
        # 例如：
        #
        # [[0.12, 0.88]]
        #
        # 表示：
        #   人类文本概率 = 12%
        #   机器文本概率 = 88%
        #
        # [0, 1]：
        #   第0行，第1列
        #   即机器文本概率
        #
        prob = float(
            self.pipeline.predict_proba([text])[0, 1]
        )

        # ==================================================
        # 第二步：根据概率判断类别
        # ==================================================
        #
        # 二分类常用规则：
        #
        # prob >= 0.5 → 判定为机器生成
        # prob <  0.5 → 判定为人类文本
        #
        label = int(prob >= 0.5)

        # 中文类别名称
        label_name = (
            "疑似机器生成"
            if label == 1
            else "疑似人类书写"
        )

        # ==================================================
        # 第三步：获取所有特征名称
        # ==================================================
        #
        # 例如可能得到：
        #
        # [
        #   "avg_sentence_length",
        #   "tfidf_人工智能",
        #   "punct_ratio",
        #   ...
        # ]
        #
        full_feature_names = np.array(
            self.features.get_feature_names_out()
        )

        # ==================================================
        # 第四步：获取特征选择器的 mask
        # ==================================================
        #
        # get_support() 返回布尔数组：
        #
        # [True, False, True, ...]
        #
        # True  表示该特征被保留
        # False 表示该特征被删除
        #
        selected_mask = self.selector.get_support()

        # 根据 mask 提取最终保留的特征名称
        selected_names = full_feature_names[selected_mask]

        # ==================================================
        # 第五步：提取当前文本的完整特征
        # ==================================================
        #
        # transform([text]) 返回二维矩阵：
        #
        # [[0.2, 0.7, 1.3, ...]]
        #
        # [0] 取第一条样本
        #
        row_full = self.features.transform([text])[0]

        # ==================================================
        # 第六步：仅保留“被选择”的特征
        # ==================================================
        #
        row_selected = row_full[selected_mask]

        # ==================================================
        # 第七步：获取模型中的特征重要性
        # ==================================================
        #
        # 对于 RandomForest / XGBoost 等树模型：
        #
        # feature_importances_
        #
        # 表示每个特征的重要程度
        #
        importance = self.clf.feature_importances_

        # ==================================================
        # 第八步：计算局部贡献值
        # ==================================================
        #
        # 简化版解释方法：
        #
        # contribution =
        #     当前样本特征值
        #     ×
        #     全局特征重要性
        #
        # 贡献越大：
        #   对“机器文本”判定影响越强
        #
        contrib = row_selected * importance

        # ==================================================
        # 第九步：找贡献最大的 topn 个特征
        # ==================================================
        #
        # np.argsort(contrib)
        #     返回从小到大的索引
        #
        # [::-1]
        #     倒序 -> 从大到小
        #
        # [:topn]
        #     取前 topn 个
        #
        idx = np.argsort(contrib)[::-1][:topn]

        # ==================================================
        # 第十步：构建可解释结果
        # ==================================================
        #
        top_features: List[Dict] = []

        for i in idx:

            # 添加一个特征解释
            top_features.append(
                {
                    # 特征名称（美化显示）
                    "feature": display_feature_name(
                        str(selected_names[i])
                    ),

                    # 当前样本中的特征值
                    "value": float(row_selected[i]),

                    # 对最终结果的贡献
                    "contribution": float(contrib[i]),
                }
            )

        # ==================================================
        # 第十一步：返回最终结果
        # ==================================================
        #
        return {
            # 数值类别
            # 0 = 人类
            # 1 = 机器
            "label": label,

            # 中文类别名
            "label_name": label_name,

            # 机器文本概率
            "probability_machine": prob,

            # 置信度
            #
            # 如果：
            #   prob = 0.93
            #
            # 则：
            #   confidence = 0.93
            #
            # 如果：
            #   prob = 0.08
            #
            # 则：
            #   confidence = 0.92
            #
            # 因为模型更确信它是“人类文本”
            #
            "confidence": max(prob, 1 - prob),

            # 重要特征解释
            "top_features": top_features,
        }