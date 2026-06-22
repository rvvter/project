from __future__ import annotations

"""模型训练模块：调参、评估、导出模型与报告。
该模块负责：
1. 构建 ML pipeline（特征 + 模型）
2. 超参数调优
3. 模型评估
4. baseline 对比
5. 特征重要性导出
6. 模型与结果保存
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 特征选择：挑选最有信息量的特征
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# 模型：逻辑回归（baseline）+ 决策树（主模型）
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# 评价指标
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# 随机搜索调参
from sklearn.model_selection import RandomizedSearchCV

# Pipeline：把“特征工程 + 模型”串起来
from sklearn.pipeline import Pipeline

# 自定义特征提取器（TF-IDF + 其他特征）
from src.features.extractor import HybridFeatureExtractor


# =========================================================
# 1. 统一指标计算函数
# =========================================================
def _metric_dict(y_true, y_pred, y_prob) -> Dict[str, float]:
    """
    统一计算分类模型的评估指标

    参数:
    - y_true: 真实标签
    - y_pred: 预测标签（0/1）
    - y_prob: 预测为正类的概率

    返回:
    - 各种指标的字典
    """

    return {
        # 准确率：预测正确的比例
        "accuracy": float(accuracy_score(y_true, y_pred)),

        # 精确率：预测为正的样本中有多少是真的正
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),

        # 召回率：真实正样本中被预测出来的比例
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),

        # F1：precision 和 recall 的调和平均
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),

        # ROC-AUC：基于概率排序能力
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


# =========================================================
# 2. 混淆矩阵可视化
# =========================================================
def _save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    """
    保存混淆矩阵图像（用于可视化模型错误分布）
    """

    # 计算混淆矩阵：
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)

    # 创建画布
    plt.figure(figsize=(4, 3))

    # 用热力图展示
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

    plt.xlabel("Predicted")  # 预测标签
    plt.ylabel("True")       # 真实标签

    plt.tight_layout()

    # 确保目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存图片
    plt.savefig(output_path, dpi=150)

    # 关闭画布，释放内存
    plt.close()


# =========================================================
# 3. 构建主模型 Pipeline
# =========================================================
def build_main_pipeline(random_state: int = 42) -> Pipeline:
    """
    构建完整训练流水线：

    输入文本 →
    特征提取 →
    特征选择 →
    决策树分类器
    """

    return Pipeline(
        steps=[
            # 1. 特征工程：TF-IDF + 自定义特征
            ("features", HybridFeatureExtractor(max_tfidf_features=300, ngram_range=(2, 4))),

            # 2. 特征选择：保留最重要的120个特征
            ("select", SelectKBest(mutual_info_classif, k=120)),

            # 3. 分类器：决策树
            ("clf", DecisionTreeClassifier(
                random_state=random_state,  # 保证可复现
                class_weight="balanced",    # 处理类别不平衡
            )),
        ]
    )


# =========================================================
# 4. 超参数调优
# =========================================================
def tune_pipeline(
    pipeline: Pipeline,
    X_train,
    y_train,
    random_state: int = 42
) -> Pipeline:
    """
    使用 RandomizedSearchCV 对 pipeline 进行超参数调优
    """

    # 定义搜索空间
    param_dist = {
        # 特征选择：保留多少特征
        "select__k": [60, 100, 140, 180, 250],

        # TF-IDF 维度
        "features__max_tfidf_features": [200, 400, 600, 800],

        # 决策树深度
        "clf__max_depth": [6, 10, 15, 25, None],

        # 分裂条件
        "clf__min_samples_split": [2, 4, 8, 16],
        "clf__min_samples_leaf": [1, 2, 4, 8, 16],

        # 分裂标准
        "clf__criterion": ["gini", "entropy"],

        # 分裂策略
        "clf__splitter": ["best", "random"],

        # 代价复杂度剪枝
        "clf__ccp_alpha": [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01],
    }

    # 随机搜索（比 GridSearch 更快）
    search = RandomizedSearchCV(
        estimator=pipeline,

        # 参数空间
        param_distributions=param_dist,

        # 随机采样次数
        n_iter=60,

        # 用 F1 作为优化目标（适合不平衡数据）
        scoring="f1",

        # 5折交叉验证
        cv=5,

        # 随机种子
        random_state=random_state,

        # 多核并行
        n_jobs=6,

        # 输出训练过程
        verbose=10,
    )

    # 训练搜索
    search.fit(X_train, y_train)

    # 返回最优模型
    return search.best_estimator_


# =========================================================
# 5. 模型评估
# =========================================================
def evaluate_model(
    model: Pipeline,
    X_test,
    y_test
) -> Tuple[Dict[str, float], str, np.ndarray, np.ndarray]:
    """
    在测试集上评估模型表现
    """

    # 输出概率（用于 ROC AUC）
    y_prob = model.predict_proba(X_test)[:, 1]

    # 将概率转为类别（默认阈值0.5）
    y_pred = (y_prob >= 0.5).astype(int)

    # 计算指标
    metrics = _metric_dict(y_test, y_pred, y_prob)

    # 生成分类报告（文本形式）
    report = classification_report(y_test, y_pred, digits=4)

    return metrics, report, y_pred, y_prob


# =========================================================
# 6. Baseline 模型（逻辑回归）
# =========================================================
def train_baselines(X_train, y_train, X_test, y_test) -> Dict[str, Dict[str, float]]:
    """
    训练简单 baseline 模型，用于对比主模型性能
    """

    # 特征提取（比主模型更简单）
    feat = HybridFeatureExtractor(max_tfidf_features=250)

    X_train_feat = feat.fit_transform(X_train)
    X_test_feat = feat.transform(X_test)

    # 逻辑回归模型
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")

    result = {}

    # 可以扩展多个 baseline 模型
    for name, model in [("logistic_regression", lr)]:

        # 训练
        model.fit(X_train_feat, y_train)

        # 概率预测
        prob = model.predict_proba(X_test_feat)[:, 1]

        # 分类
        pred = (prob >= 0.5).astype(int)

        # 计算指标
        result[name] = _metric_dict(y_test, pred, prob)

    return result


# =========================================================
# 7. 导出特征重要性
# =========================================================
def export_feature_importance(
    trained_pipeline: Pipeline,
    output_csv: Path,
    topn: int = 40
) -> pd.DataFrame:
    """
    导出决策树 Top-N 重要特征
    """

    # 获取 pipeline 各阶段
    feature_extractor: HybridFeatureExtractor = trained_pipeline.named_steps["features"]
    selector: SelectKBest = trained_pipeline.named_steps["select"]
    clf: DecisionTreeClassifier = trained_pipeline.named_steps["clf"]

    # 所有特征名
    full_names = np.array(feature_extractor.get_feature_names_out())

    # 只保留被选中的特征
    selected_names = full_names[selector.get_support()]

    # 决策树特征重要性
    importances = clf.feature_importances_

    # 组织成 DataFrame
    fi_df = pd.DataFrame({
        "feature": selected_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # 保存 CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fi_df.head(topn).to_csv(output_csv, index=False, encoding="utf-8-sig")

    return fi_df


# =========================================================
# 8. 完整训练流程入口
# =========================================================
def train_and_save(
    train_csv: Path,
    test_csv: Path,
    artifacts_dir: Path
) -> None:
    """
    完整训练流程：

    1. 读取数据
    2. 训练 + 5折交叉验证调参
    3. 测试集评估
    4. baseline 对比
    5. 保存模型 + 报告 + 图
    """

    # 读取数据
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # 训练集（80%，内部由交叉验证完成调参）
    X_train = train_df["text"]
    y_train = train_df["label"]

    # 测试集（20%，完全独立，最终评估）
    X_test = test_df["text"]
    y_test = test_df["label"]

    # 构建 pipeline
    pipeline = build_main_pipeline()

    # 调参
    best_model = tune_pipeline(pipeline, X_train, y_train)

    # 评估主模型
    metrics, cls_report, y_pred, _ = evaluate_model(best_model, X_test, y_test)

    # baseline 对比
    baseline_scores = train_baselines(X_train, y_train, X_test, y_test)

    # 创建输出目录
    model_dir = artifacts_dir / "models"
    report_dir = artifacts_dir / "reports"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型
    joblib.dump(best_model, model_dir / "detector_pipeline.joblib")

    # 保存特征重要性
    export_feature_importance(best_model, report_dir / "feature_importance.csv", topn=50)

    # 保存混淆矩阵图
    _save_confusion_matrix(y_test, y_pred, report_dir / "confusion_matrix.png")

    # 保存所有指标 + 报告
    metrics_payload = {
        "main_model": metrics,
        "baselines": baseline_scores,
        "classification_report": cls_report,
    }

    (report_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("训练完成，结果保存在:", artifacts_dir)