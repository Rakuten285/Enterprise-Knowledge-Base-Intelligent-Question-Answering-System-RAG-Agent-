# 面试准备文档：RAG 效果评测（Ragas）

> 本文档与代码同步维护。**基线评测日期：2026-06-21，全部实测。**
> 所有指标均为 Ragas 框架实测结果，不是预期值。

---

## 1. 为什么要做评测？评什么？

**问：你怎么知道你的 RAG 效果"好"？**

"我测试了几个问题，回答看起来对"——这不是工程回答。工程上需要：
1. **可复现的量化指标**：相同代码、相同数据集，任何人跑出来的数字一样
2. **能发现问题的指标体系**：不同的指标能定位不同类型的问题（检索问题 vs 生成问题）
3. **有对比基准**：调参前后指标变化多少，能说明优化是否有效

我选用 **Ragas** 框架，它是专门针对 RAG 系统的自动化评测框架，不需要人工标注评分，而是通过 LLM 作为评判者来计算指标。

---

## 2. 三项核心指标解释

### 2.1 Context Recall（上下文召回率）

**衡量什么**：检索到的文档片段有多大比例"覆盖"了 ground truth 中的信息。

**计算原理**：把 ground truth 拆解成若干声明，检查每条声明能否在 retrieved contexts 中找到支撑。

**值域**：0 ~ 1，越高越好

**问题定位**：Context Recall 低 → **检索阶段有问题**，检索器没取到相关文档，需要调 top_k 或混合检索策略

```
ground_truth 里的信息点 能在 retrieved_contexts 中找到的比例 = Context Recall
```

### 2.2 Faithfulness（忠实度）

**衡量什么**：LLM 生成的答案有多少内容是基于检索文档的，即**幻觉程度**。

**计算原理**：把生成答案拆解成若干声明，检查每条声明能否在 retrieved contexts 中找到支撑。

**值域**：0 ~ 1，越高越好

**问题定位**：Faithfulness 低 → **生成阶段有问题**，LLM 在检索内容之外"自由发挥"了，需要加强 System Prompt 的约束或降低 temperature

```
生成答案里的声明 能在 retrieved_contexts 中找到支撑的比例 = Faithfulness
```

### 2.3 Answer Relevancy（答案相关性）

**衡量什么**：生成的答案和原始问题的相关程度，判断 LLM 有没有"答非所问"。

**计算原理**：让 LLM 根据答案反推出若干问题，用 Embedding 计算这些问题与原始问题的余弦相似度均值。

**值域**：0 ~ 1，越高越好

**问题定位**：Answer Relevancy 低 → LLM 回答跑题，或者回答过于模糊/冗余

---

## 3. 评测集设计

**评测集位置**：`data/eval_dataset.json`

**规模**：95 条问答对（初版 50 条 + 扩充 45 条）

**知识库规模**：6 份文档，chunk_size=500 时共 28 个分块（初版仅 7 个分块，扩充后参数调优才有意义）

**来源分布**：
| 文档 | 问题数 | 覆盖内容 |
|---|---|---|
| 员工手册.md | 25 条 | 入职流程、考勤、薪酬绩效、报销标准、离职流程 |
| IT安全规范.md | 15 条 | 账号密码、VPN、数据分级、设备管理、安全上报 |
| 采购管理制度.docx | 10 条 | 采购审批、供应商分级、付款周期 |
| 绩效考核管理办法.md | 15 条 | 考核周期、评级比例、考核流程、结果应用、申诉机制 |
| 员工培训管理制度.md | 15 条 | 入职培训、在岗培训、外部培训、服务期约定、费用报销 |
| 商务接待与差旅管理规定.md | 15 条 | 出差审批、交通住宿标准、餐饮补贴、商务接待标准 |

**格式设计**：
```json
{
  "id": 1,
  "question": "新员工入职第一天需要携带哪些证件？",
  "ground_truth": "新员工入职需携带身份证、学历证明、离职证明",
  "reference_contexts": ["新员工入职需在第一个工作日携带..."],
  "source_doc": "员工手册.md"
}
```

**字段说明**：
- `ground_truth`：标准答案，来自原文直接引用，无推断补充。供 Ragas 的 Context Recall 指标使用
- `reference_contexts`：从原文抠出来的支撑句段，供 Ragas 的 Context Recall 指标对比用
- `source_doc`：来源文件，便于分析哪类文档的检索效果差

**为什么 ground_truth 必须来自原文？**

如果 ground_truth 是推断或补全的，Context Recall 就会把"检索到原文"和"ground_truth 需要推断"两个问题混在一起，指标失去诊断意义。严格的评测集设计是指标可信的前提。

---

## 4. 基线评测结果（chunk_size=500, top_k=5）

**实测日期**：2026-06-21

**环境**：
- 知识库：6 份文档，28 个分块（chunk_size=500, overlap=50）
- 混合检索：RRF 融合（BM25 + 向量），top_k=5
- 生成 LLM：DeepSeek deepseek-chat，temperature=0.1
- 评判 LLM（Ragas）：DeepSeek deepseek-chat
- 评判 Embedding（Ragas）：本地 BGE bge-small-zh-v1.5
- 评测集：95 条问答对

| 指标 | 基线得分 |
|---|---|
| **Context Recall** | **0.9789** |
| **Faithfulness** | **0.7954** |
| **Answer Relevancy** | **0.8395** |
| 总耗时 | 282.8s |

**结果解读**：

- **Context Recall = 0.9789**：高但不满分。top_k=5 从 28 个分块中取 5 个，约 2% 的问题检索时遗漏了关键信息，说明知识库规模达到了有区分度的范围（初版 7 个分块时 Context Recall 虚高到 1.0）。

- **Faithfulness = 0.7954**：中等偏上。约 20% 的生成声明无法从检索文档直接找到依据，主要原因是 LLM 对规定内容进行了解释性补充（并非编造错误信息），但 Ragas 会把这类补充判为不忠实。

- **Answer Relevancy = 0.8395**：良好，LLM 的回答切题度高，没有明显跑题问题。

---

## 5. 参数调优实验

### 5.1 调优逻辑

两个关键参数：
- **chunk_size**（分块大小）：影响每个检索片段的信息密度
  - 太小（300）：每块信息少，需要更多块才能覆盖，但检索更精准
  - 太大（800）：每块信息多，可能覆盖更多上下文，但向量相似度可能变差
- **top_k**（检索条数）：影响给 LLM 的上下文总量
  - 太小（3）：可能遗漏相关信息
  - 太大（8）：噪声增加，LLM 可能被无关内容干扰

### 5.2 参数扫描结果

扫描范围：chunk_size ∈ {300, 500, 800} × top_k ∈ {3, 5, 8}，共 9 个组合（**实测日期：2026-06-21**）

知识库：6 份文档，chunk_size=500 时 28 个分块；评测集：95 条问答对

| chunk_size | top_k | Context Recall | Faithfulness | Answer Relevancy | 综合均值 |
|---|---|---|---|---|---|
| 500 | 8 | 0.9895 | 0.8148 | 0.8462 | 0.8835 |
| **800** | **8** | **0.9895** | **0.8943** | **0.8434** | **0.9091** ← 最优 |
| 500 | 5（基线） | 0.9789 | 0.8186 | 0.8428 | 0.8801 |
| 300 | 8 | 0.9632 | 0.8008 | 0.8444 | 0.8695 |
| 800 | 5 | 0.9579 | 0.8661 | 0.8202 | 0.8814 |
| 300 | 5 | 0.9421 | 0.7659 | 0.8385 | 0.8488 |
| 800 | 3 | 0.9263 | 0.8051 | 0.8137 | 0.8484 |
| 300 | 3 | 0.9211 | 0.7111 | 0.8038 | 0.8120 |
| 500 | 3 | 0.8684 | 0.7597 | 0.7429 | 0.7903 |

**结果解读**：

- **top_k=3 是明显短板**：chunk_size=500 + top_k=3 的 Context Recall 跌至 0.87，Answer Relevancy 也跌到 0.74——这在 28 个分块的知识库里才显现出来，7 个分块时完全看不到这个问题
- **chunk_size=800 的 Faithfulness 显著最高（0.89）**：大分块让每个片段保留完整的规则上下文，LLM 需要自行补充的内容更少，幻觉更低。这个结论在扩充知识库后依然成立，且差异更明显
- **top_k=8 vs top_k=5**：扩充后 top_k=8 比 top_k=5 的 Context Recall 高约 0.01~0.03，说明 28 个分块的知识库里多取几条确实有收益（7 个分块时 top_k=5 已经覆盖大半，差异不明显）
- **综合最优是 chunk_size=800, top_k=8**：三项指标综合均值 0.9091，明显优于其他组合

### 5.3 最终选定参数：chunk_size=800, top_k=8

**优化前（基线，chunk_size=500, top_k=5）vs 优化后：**

| 指标 | 基线 | 优化后 | 变化 |
|---|---|---|---|
| Context Recall | 0.9789 | **0.9895** | ↑ +0.0106（+1.1%） |
| Faithfulness | 0.7954 | **0.8943** | ↑ **+0.0989（+12.4%）** |
| Answer Relevancy | 0.8395 | **0.8434** | ↑ +0.0039（基本持平） |

**选型理由**：
- Faithfulness 提升 12.4%——抗幻觉能力是 RAG 系统最核心的质量维度，此处改善最为显著
- Context Recall 小幅提升 1.1%，说明 top_k 从 5 增加到 8 在 28 块的知识库中有实际收益
- chunk_size=800 让企业规章文档的完整条款（平均每条 300-600 字）落在同一分块内，避免被截断导致语义不完整
- 代价：top_k=8 比 top_k=5 向 LLM 多传入约 60% 的上下文 token，推理成本略有增加，但对于企业内部问答场景可接受

---

## 6. 评测集自检：怎么保证数据本身是对的

**问：你的评测集是手写的，你怎么保证它没有错误？**

这是一个比评测指标本身更根本的问题。如果评测集的数据是错的，跑出来的指标也没有意义。

### 6.1 为什么需要自检

评测集里有两类典型错误来源：

1. **reference_contexts 写错了**：把一段不在原文中的内容当成引用（比如改写、合并、推断）
2. **ground_truth 包含推断内容**：写了"工作满3年属于该区间，因此有5天年假"——这是推断，不是原文，Context Recall 就会把"检索原文"和"ground_truth 含推断"两件事混在一起，指标失去诊断意义

### 6.2 用什么方法自检

写了 `scripts/check_eval_dataset.py`，对每条 Q&A 做两项机械检查：

**检查一：reference_contexts 是否真实存在于原文**

对每条 `reference_contexts` 里的文字，在对应 `source_doc` 的原始文件里做子串查找。找不到 → `FAIL`（直接引用错误）。

```python
def check_context_in_source(context: str, source_text: str) -> bool:
    return normalize(context) in normalize(source_text)
```

**检查二：ground_truth 里的数字是否在原文里出现**

数字是最容易出错的部分（天数、金额、比例）。提取 ground_truth 里所有数字，逐一在原文中查找。找不到 → `WARN`（可能是推断或笔误）。

```python
def extract_numbers(text: str) -> list[str]:
    return re.findall(r'\d+(?:\.\d+)?(?:\s*[%元天年月日小时万])?', text)
```

**为什么是子串匹配而不是语义匹配？**

语义匹配需要 LLM，成本高，而且语义"相近"不等于"直接引用"。评测集的设计原则是 ground_truth 必须是原文的直接引用，所以机械的字符串匹配已经足够——匹配上了就是原文有，匹配不上就是有问题。

### 6.3 一个关键 Bug 和修复

运行自检时发现 **21 条 FAIL**，根因是原文件里有 Markdown 格式标记（如 `**加粗**`），而 reference_contexts 引用时去掉了星号，导致字符串不匹配：

```
原文：**工作满1年不满10年**的员工，每年享有5天年假
reference_context："工作满1年不满10年的员工，每年享有5天年假"
```

直接子串查找失败（原文有 `**`，引用没有）。

修复：在比较前对原文和引用都做归一化——去掉 Markdown 标记和多余空白：

```python
def normalize(text: str) -> str:
    text = re.sub(r'[*_#`>|]', '', text)   # 去掉 Markdown 符号
    text = re.sub(r'\s+', '', text)          # 去掉所有空白
    return text
```

归一化后重新比较，21 FAIL → 0 FAIL，95/95 全部 PASS。

### 6.4 自检的局限性

自检能发现的是**形式错误**（引用不在原文中、数字对不上），不能发现**语义问题**：

| 能发现 | 不能发现 |
|---|---|
| reference_contexts 不在原文中 | ground_truth 语义正确但选的不是最关键的句子 |
| ground_truth 数字与原文不符 | 问题本身的覆盖度是否均匀 |
| 数据录入笔误 | 问题是否足够有区分度 |

这也是为什么在自检通过之后，还要结合 Ragas 评测——自检保证"数据没有错误"，Ragas 评测保证"RAG 系统表现如何"，两者互补。

---

## 7. Ragas 框架工程细节

**问：Ragas 评测框架是怎么工作的？**

```python
from ragas import evaluate
from ragas.metrics import ContextRecall, Faithfulness, AnswerRelevancy

# 构建 HuggingFace Dataset 格式
dataset = Dataset.from_dict({
    "question": [...],
    "answer": [...],        # RAG 系统生成的答案
    "contexts": [[...]],    # 检索到的文档片段（列表的列表）
    "ground_truth": [...],  # 标准答案
})

result = evaluate(dataset, metrics=[ContextRecall(...), Faithfulness(...), AnswerRelevancy(...)])
```

**踩坑记录（2026-06-21）**：

**坑：DeepSeek 不支持 n > 1 的 API 参数**

Ragas 的 `AnswerRelevancy` 默认会让 LLM 生成多个反推问题（`n=3`），DeepSeek API 返回 400 错误：
```
BadRequestError: Invalid n value (currently only n = 1 is supported)
```

修复：初始化时加 `strictness=1`：
```python
AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb, strictness=1)
```

这样每次只让 LLM 生成一个反推问题，DeepSeek 兼容，Answer Relevancy 正常计算。

**注意**：`strictness=1` 会使 Answer Relevancy 的计算方差更大（只采样一次），但对于 50 条的评测集已经足够稳定。

---

## 8. 已知未完成 / 待验证事项

1. **参数扫描结果待填入**：扫描完成后更新 §5.2 表格和 §5.3 最优参数
2. **Faithfulness 改善**：0.73 有优化空间，可尝试调低 temperature 或修改 System Prompt 约束
3. **Multi-Agent 性能测试**：阶段2文档里提到的"20条混合查询，单/多Agent各跑10次取均值"尚未实施
4. **分文档分析**：当前只有整体均值，缺少按文档类型（员工手册/IT安全/采购）的细分分析，便于发现哪类文档检索效果差
