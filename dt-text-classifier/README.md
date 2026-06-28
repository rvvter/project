# 基于决策树的机器生成文本检测系统设计与实现

本项目围绕毕业设计主线构建：**特征工程 + 决策树 + 系统实现**，用于识别”疑似人类文本”与”疑似机器生成文本”。

## 1. 项目目标

- 设计多层次文本特征，捕捉人类与大模型文本差异
- 使用 `DecisionTreeClassifier` 训练检测器
- 输出可解释结果：特征重要性、单样本贡献特征
- 提供可运行的 Flask 检测系统（文本输入 + 实时预测 + 可视化）

## 2. 项目结构

```text
.
├─ scripts/
│  ├─ build_real_dataset.py             # 从 HC3 等语料构建训练数据
│  └─ run_pipeline.py                   # 一键执行：数据→训练→评估→导出
├─ src/
│  ├─ data/
│  │  ├─ preprocess.py                  # 清洗、标准化、划分数据集
│  │  └─ real_dataset.py                # 多格式原始语料解析（JSON/JSONL/CSV/TXT）
│  ├─ features/
│  │  └─ extractor.py                   # 多层次文本特征工程（27维手工 + 300维TF-IDF）
│  ├─ models/
│  │  ├─ train.py                       # 训练、调参、评估、特征选择、基线对比
│  │  └─ predict.py                     # 推理与可解释性输出
│  └─ system/
│     ├─ app.py                         # Flask 系统后端
│     └─ validation.py                  # 输入校验（长度/字符/词元/特殊符号）
├─ static/
│  ├─ style.css                         # 页面样式
│  └─ validation.js                     # 前端输入校验
├─ templates/
│  └─ index.html                        # Web 前端页面
├─ data/
│  ├─ raw_sources/
│  │  └─ hc3/
│  │     └─ hc3_train.jsonl             # HC3 原始语料
│  ├─ raw/
│  │  ├─ human_texts.csv                # 预处理后的人类文本
│  │  └─ machine_texts.csv              # 预处理后的机器文本
│  └─ processed/
│     ├─ train.csv                      # 训练集
│     └─ test.csv                       # 测试集
├─ artifacts/
│  ├─ models/
│  │  └─ detector_pipeline.joblib       # 训练好的完整流水线
│  └─ reports/
│     ├─ metrics.json                   # 评估指标
│     ├─ feature_importance.csv         # 特征重要性 Top-50
│     └─ confusion_matrix.png           # 混淆矩阵图
├─ requirements.txt
└─ README.md
```

## 3. 环境安装

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

## 4. 在线体验

已部署至 Render，可直接访问试用：

🔗 **https://project-nufq.onrender.com**

> 注意：免费部署实例在 15 分钟无请求后会自动休眠，首次访问需等待 30-50 秒冷启动。

## 5. 快速开始

### Step 1: 准备原始数据

将 HC3 数据集放入 `data/raw_sources/hc3/`，支持 `.jsonl`、`.json`、`.csv` 格式。

HC3 数据集中需包含 `human_answers` 和 `chatgpt_answers` 字段，脚本会自动识别并分离人类/机器文本。

### Step 2: 构建训练数据

```bash
python scripts/build_real_dataset.py \
  --hc3-dir data/raw_sources/hc3 \
  --output-dir data/raw \
  --min-len 20
```

生成：
- `data/raw/human_texts.csv`
- `data/raw/machine_texts.csv`

### Step 3: 训练与评估

```bash
python scripts/run_pipeline.py \
  --human-csv data/raw/human_texts.csv \
  --machine-csv data/raw/machine_texts.csv \
  --train-size 0.6 \
  --valid-size 0.2 \
  --test-size 0.2 \
  --seed 42
```

输出：
- `artifacts/models/detector_pipeline.joblib`（模型流水线）
- `artifacts/reports/metrics.json`（Accuracy / Precision / Recall / F1 / ROC-AUC）
- `artifacts/reports/feature_importance.csv`（特征重要性）
- `artifacts/reports/confusion_matrix.png`（混淆矩阵）

### Step 4: 启动系统

```bash
python -m src.system.app
```

浏览器打开：`http://127.0.0.1:5000`

## 6. 方法说明

### 6.1 特征工程

从四个层次提取文本特征，共 327 维（27 维手工 + 300 维 TF-IDF）：

**手工统计特征（27 维）：**
- 表面特征：字符数、词元数、句子数、平均句长、句长标准差、平均词长
- 词汇特征：词汇丰富度（TTR）、字符多样性、重复词元占比、短词元占比、词频熵、最高词频、功能词占比、”的”密度、单现词比例、尤尔K值、辛普森多样性、平均词频排名
- 风格特征：标点密度、特殊符号密度、积极/消极情感词数、可读性分数、压缩比、句首词多样性、词长波动、数字密度

**字符级 TF-IDF 特征（300 维）：**
- 按字符切分的 2-gram 到 4-gram 片段
- 通过 TF-IDF 加权后保留权重最高的 300 个片段

**特征选择：**
- 使用 `SelectKBest(mutual_info_classif)` 从 327 维中保留最具判别力的 140 维

### 6.2 模型

- 主模型：`DecisionTreeClassifier`（CART 算法 + 代价复杂度剪枝）
- 基线模型：`LogisticRegression`
- 调参：`RandomizedSearchCV`（60 轮随机搜索，5 折交叉验证）
- 优化目标：F1 分数

### 6.3 可解释性

- 全局解释：决策树特征重要性排序（导出为 CSV）
- 局部解释：单次预测返回”贡献最大的前 N 个特征值”（特征值 × 全局重要性）

## 7. 实验结果

| 模型 | 准确率 | 精确率 | 召回率 | F1 分数 | ROC-AUC |
|------|--------|--------|--------|---------|---------|
| 决策树（主模型） | 90.59% | 89.49% | 90.09% | 89.79% | 94.89% |
| 逻辑回归（基线） | 94.53% | 93.63% | 94.52% | 94.07% | 98.69% |

特征重要性 Top-5：

| 排名 | 特征 | 重要度 |
|------|------|--------|
| 1 | tfidf_，并（字符片段） | 13.99% |
| 2 | 句首词多样性 | 11.69% |
| 3 | tfidf_果您（字符片段） | 11.21% |
| 4 | tfidf_我无（字符片段） | 6.62% |
| 5 | tfidf_。它（字符片段） | 5.39% |

## 8. 常见问题

- 运行报缺少 NLTK 资源：首次运行会自动下载 `punkt` 和 `stopwords`
- 数据量较小时分数波动大：建议每类至少 2000+ 样本
- 中文分词效果一般：可替换为更高质量分词器
- 构建数据报”至少一类样本为空”：检查原始目录文件是否包含可识别字段（如 `text`、`human_answers`、`chatgpt_answers`）
