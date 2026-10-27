# Tabular ML Model Validation Benchmark

面向模型验证学习的可复现 Python portfolio 项目：在 **UCI Statlog German Credit（ID 144）** 上，公平比较 Logistic Regression 与 Random Forest。重点是数据边界、指标、成本和结论的证据，而不是追求最高分。

这是经典表格机器学习基准，**不是深度学习基础模型验证**。未使用 TD 专有数据，不证明银行生产就绪、合规性或临床适用性。

## 环境与复现

本项目实测 Python **3.12.14**。在项目根目录运行（macOS/Linux shell）：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/inspect_data.py
python -m pytest -q
python -m scripts.benchmark
cat reports/validation_report.md
```

Windows 可用 `py -3.12 -m venv .venv`，随后用 `.venv\Scripts\Activate.ps1` 激活。
在 VS Code 中打开本项目目录，选择 `.venv` 解释器，然后预览 `reports/validation_report.md`。

直接依赖固定在 `requirements.txt`，包括 numpy、pandas、scikit-learn、matplotlib、pytest；完整环境版本保留在 `requirements-lock.txt`。此次额外约束已有间接依赖 scipy=1.14.1、pyparsing=3.2.3，以修复原锁文件与旧版 sklearn/matplotlib 的兼容警告；未引入新框架。不同硬件／底层数值库仍可能存在微小浮点差异。

可选快速运行（不覆盖正式报告）：

```bash
python -m scripts.benchmark --folds 2 --trees 8 --output-dir reports/smoke
```

## 数据来源与含义

官方来源：[UCI ID 144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)。
Hofmann, H. (1994), DOI: [10.24432/C5NC77](https://doi.org/10.24432/C5NC77)。UCI 标注许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，共享时须保留来源与许可说明并注明改动。

使用官方原始混合类型文件 `german.data`：1,000 行、20 个特征，7 个数值和 13 个类别特征。原始文件随项目保存，每次加载校验 SHA-256；缺失时从官方 ZIP 下载，下载或校验失败直接报错。没有合成数据替代路径。详见 [数据溯源](data/README.md) 和 [原始字段说明](data/raw/german.doc)。

**原始标签 1=good、2=bad → 项目标签 0=good、1=bad（正类）**。实际检查得到 700 个 good、300 个 bad，无缺失值、无完全重复特征行。`A65` 等“未知／无账户”编码属于有效类别，不自动当成缺失值。

## 验证方法

- 固定随机种子 42，分层划分 800 行训练集、200 行留出集；分层帮助维持类别比例。
- 训练集内 5 折 `StratifiedKFold(shuffle=True, random_state=42)`。两模型共享完全相同的折。
- `ColumnTransformer` 按预先指定的字段类型处理；数值中位数填补，类别众数填补和 one-hot 编码。LR 额外进行数值标准化，RF 无需缩放。
- 预处理与模型都放在 `Pipeline` 中：每折只对该折训练部分调用 `fit`，验证／留出数据只调用变换和预测。未见类别编码为全零。
- LR 使用 C=1、lbfgs、max_iter=2000；RF 使用 200 棵树、不限制深度。其余参数使用锁定 sklearn 版本的默认值。没有超参数搜索、类别重加权或阈值优化。
- 分类规则固定为 `P(bad) >= 0.5`；precision、recall、F1 的正类为 bad。ROC-AUC 使用 bad 的预测概率。无正类预测时 precision/F1 设为 0。
- 报告 accuracy、precision、recall、F1、ROC-AUC、混淆矩阵和成本；交叉验证提供折均值与样本标准差（ddof=1），不将其解释为置信区间。
- 训练分数用于观察拟合程度；留出分数用于独立评估。留出集不用于选超参数、阈值或预处理方案。

混淆矩阵为 `[[TN, FP], [FN, TP]]`，行是真实类别，列是预测类别，顺序均为 `[good, bad]`。
官方基准总成本为 **FP + 5×FN**，平均成本为总成本／样本数。该成本不是金额，不代表实际银行损失。
0.5 是固定基线阈值，并非成本最优阈值；理论 1/6 阈值需要概率校准与成本适用性等假设，本项目没有据此做生产决策。

## 阅读顺序与文件结构

1. `scripts/inspect_data.py`：保留原有 `load_data()`、`summarize()` 结构，迁移到经过校验的 UCI 数据。
2. `scripts/benchmark.py`：划分 → 构建 Pipeline → 训练集 CV → 最终训练 → 留出评估。
3. `scripts/reporting.py`：把实际结果生成 Markdown 表格、错误分析和 matplotlib 混淆矩阵图片。
4. `tests/test_data.py`、`tests/test_benchmark.py`：数据契约、异常处理、指标已知答案、训练边界和两次小规模完整运行。
5. `reports/validation_report.md`：正式运行的报告；`benchmark_results.json` 包含环境、配置、行索引和逐折结果；`holdout_predictions.csv` 可追溯逐行预测。

代码注释使用简单英文。类型提示如 `tuple[pd.DataFrame, pd.Series]` 描述函数返回类型，不自动执行运行时校验；`clone()` 创建未拟合的模型副本，避免不同折复用已学习状态。

## 局限与后续学习

这是 1994 年捐赠的小规模历史德国信贷数据；不能证明对当前加拿大客群的适用性。随机划分不能替代时间外、机构外验证。个人状态／性别、年龄、外籍劳动者字段涉及公平性及代理变量问题；尚未完成分组公平性、概率校准、漂移或真实业务成本评估。

无完全重复行不等于客户独立；未使用可验证的客户分组信息。缺失值测试仅检验处理机制，不代表真实缺失分布下的稳健性。错误数量和训练／留出差距不能单独解释错误成因。

后续如需调参或选阈值，应仅使用训练数据；反复依据当前留出结果修改方案会削弱其独立性。

## 保留的项目历史

原乳腺癌数据第一阶段的 README、检查脚本、测试及报告保存在 `reports/history/stage1_breast_cancer/`，仅用于追溯，不参与当前测试收集。`pytest.ini` 明确只收集 `tests/`。

原目录未初始化 Git；本次未创建 GitHub 仓库，也未执行推送。`.gitignore` 排除虚拟环境、缓存、密钥和本机设置；经许可保存的公共原始数据与正式报告可以提交。
