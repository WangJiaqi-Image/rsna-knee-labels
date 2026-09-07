# rsna-knee-labels

RSNA Knee Abnormality Detection比赛里两个可以独立复用的模块：

1. **打标签模块**（`rsna_knee_labels.labeling`）：从放射科报告原文里提取12个诊断项的弱标签，有规则版和本地LLM版两种实现
2. **本地评分模块**（`rsna_knee_labels.evaluation`）：算macro AUC，以及拿标签源去跟少量gold真值对比、看哪里错得多

背景：官方train.csv里只有一小部分study（当前数据集是58个）在12个诊断项上有完整的真实标注（下面统称"gold"），其余study只有报告原文，没有真值标签。这两个模块就是为了解决"剩下的study怎么弄到能训练用的标签"和"怎么知道弄出来的标签靠不靠谱"这两个问题。

## 安装

```bash
pip install -e .            # 只要规则版打标签 + 评分模块，依赖很轻（pandas/numpy/scikit-learn）
pip install -e ".[llm]"     # 还要用LLM版打标签，会多装vllm/transformers，需要GPU
```

## 模块一：打标签

### 规则版（`labeling.rule_based`）

9种语言的正则规则提取器（英/西/法/荷/德/土/克塞波/希腊/保俄），不需要GPU，跑得很快，可以当默认baseline或者LLM版跑不起来时的备用。

```python
from rsna_knee_labels.labeling import extract, TARGETS

result = extract("No evidence of ACL tear. Mild medial meniscus degeneration.")
# {"ACL": 0.09, "ACL__conf": 0.57, "Medial Meniscus": 0.71, ...}
```

`extract()`对每个诊断项返回`(score, confidence)`：`score`是0~1的连续打分（不是硬0/1），`confidence`是这次判断的置信度，训练时可以用来给样本加权（置信度低的样本权重小一点）。`TARGETS`是12个诊断项名字的列表，顺序固定。

### LLM版（`labeling.llm_based`）

本地跑Qwen2.5-14B-Instruct（通过vLLM），让模型直接读报告判断"临床上算不算阳性"，而不是字面上有没有出现关键词。在咱们58个gold study上验证过，比规则版准（0.8424 vs 0.8170 macro AUC，见下面"可靠性"一节）。**报告文本全程本地处理，不会发到任何外部API**，如果你的数据有合规要求这点比较重要。

```python
import pandas as pd
from rsna_knee_labels.labeling.llm_based import run

studies_df = pd.read_csv("train.csv")   # 至少要有 StudyInstanceUID, Report 两列
label_table = run(studies_df, tensor_parallel_size=2, batch_size=64)
label_table.to_csv("label_table_llm.csv")
```

或者直接用命令行（先 `pip install -e ".[llm]"`）：

```bash
# 先在58个gold study上跑，验证这次的抽取效果
python -m rsna_knee_labels.labeling.llm_based --train-csv train.csv --studies gold

# 确认没问题后跑全量
python -m rsna_knee_labels.labeling.llm_based --train-csv train.csv --studies all -o label_table_llm.csv
```

输出的CSV里每个诊断项两列：`<target>` (score) 和 `<target>__conf` (confidence)，跟规则版格式一致，可以直接互相替换。

## 模块二：本地评分

### `evaluation.macro_auc`

比赛评分口径的复现：每个诊断项单独算ROC-AUC，再取平均。某一列如果只有一个类别（全阳或全阴），这一列记nan、不参与平均，不会假装算出0.5或1.0。

```python
from rsna_knee_labels.evaluation import macro_auc

score = macro_auc(y_true, y_pred)   # 两个都是 (n_studies, n_targets) 的array
```

### `evaluation.score_against_gold`：标签源整体质量检查

拿任意一份标签（规则版，或LLM版的输出CSV）去跟gold真值比，按诊断项拆开看AUC、阳性率、平均置信度——不要只看一个macro数字，因为不同语言、不同诊断项的覆盖率差很多，一个宏观平均会掩盖"到底哪几项其实没读对"。

```python
import pandas as pd
from rsna_knee_labels.evaluation import score_against_gold

train_df = pd.read_csv("train.csv")
score_against_gold(train_df)                              # 检查规则版
score_against_gold(train_df, "label_table_llm.csv")        # 检查某个LLM标签表
```

命令行：

```bash
python -m rsna_knee_labels.evaluation.gold_check score --train-csv train.csv
python -m rsna_knee_labels.evaluation.gold_check score --train-csv train.csv --label-table label_table_llm.csv
```

### `evaluation.worst_misses`：逐条看错在哪

针对某一个诊断项，列出"gold是阳性、但标签源打分最低"和"gold是阴性、但标签源打分最高"的几条study，连同报告原文一起打印出来，方便直接读报告、判断是漏词还是误判，而不是只盯着一个AUC数字猜。

```bash
python -m rsna_knee_labels.evaluation.gold_check diagnose ACL --train-csv train.csv
python -m rsna_knee_labels.evaluation.gold_check diagnose ACL --train-csv train.csv --label-table label_table_llm.csv -n 5
```

## 数据格式要求

两个模块都只依赖一个DataFrame（或对应CSV），至少要有：

- `StudyInstanceUID`：study的唯一ID
- `Report`：放射科报告原文（打标签模块用）
- 12个`TARGETS`列（`rsna_knee_labels.TARGETS`）：gold真值，没有标注的study这些列留空/NaN（评分模块用，判定"这个study是不是gold"就是看这12列是否全部非空）

跟具体项目的DICOM读取、训练pipeline完全解耦，不需要装torch/vLLM之外的比赛专属依赖。

## 关于可靠性，用之前请先了解

这两个模块本身也是有噪声的近似工具，不是真值：

- **打标签模块**：LLM版在58个gold study上验证的macro AUC约0.84，规则版约0.82——能用，但远不是100%准。而且这个0.84本身只是从58个study估出来的，样本量小，置信区间不窄。真正训练时，98%+的study标签完全来自这两个模块之一，没有任何真值可以核对。
- **本地评分模块**：`macro_auc`本身的实现是对的（跟比赛官方口径一致），但它算出来的"holdout准不准"这件事，取决于holdout studies的标签是不是真值——如果holdout大部分也是弱标签，那本地AUC衡量的是"预测有多接近弱标签"，不是"预测有多接近真实情况"。只有真正落在gold子集里的study，本地AUC才是跟真值比的；这部分study数量通常很少（个位数到几十），单次统计意义有限。

实践中比较稳妥的用法：**本地这两个模块的数字，只用来判断方向（这个改动是变好还是变差），不要直接拿绝对值当作模型真实水平**。要确认真实效果，最终还是要看下游任务（比如训练出来的模型）在真正的hold-out/测试集上的表现。
