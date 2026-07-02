# 项目总览：企业知识库智能问答系统（RAG + Multi-Agent）

> 本文档是整个项目的"地图"。任何人（包括后续接手的 Claude Code 会话、包括看代码的面试官）
> 应该先读这份文档，再去看具体模块代码或其他 docs/ 下的细节文档。
> **当前完成阶段：阶段1-4均已完整实现并端到端验证。**

---

## 一、项目最终要长成什么样

这是一个面向**开源展示 + 求职面试**的项目，不是单纯的练习 demo。最终交付物预期包含：

1. 一个能跑起来的企业知识库问答系统，支持上传 PDF/Word 文档，检索并基于检索内容回答问题
2. 多智能体协作能处理"需要调用工具"的复杂问题（搜索、计算、查数据库），不是单纯的检索+生成
3. 有量化的效果评估报告（不是"我觉得效果不错"，而是 Ragas 跑出来的具体指标和对比数据）
4. 能通过 Docker Compose 一键部署，别人 `git clone` 下来就能跑起来体验
5. README、架构图、API 文档齐全，**GitHub 仓库本身就是一份可以放进简历的作品**

**关键原则**：每一个量化指标（提升15%、降低30%、0.61→0.79 这些数字）**必须是真实跑出来的实测结果**，不是预先写在项目介绍里、之后再去凑数据。如果某个阶段还没跑出实测数据，宁可在文档里写"目标值，待验证"，也不能编一个数字。这是为了面试时经得起追问——如果面试官问"15%这个数字怎么测的，你的评测集是什么样的"，必须能拿出真实的评测脚本和结果，而不是哑口无言。

---

## 二、四大阶段总览与当前进度

| 阶段 | 内容 | 状态 | 对应代码目录 | 详细设计文档 |
|---|---|---|---|---|
| 1. RAG 核心链路 | 文档解析分块、Embedding、FAISS向量库、BM25混合检索、MMR去重、LLM生成 | ✅ 代码完成，✅ 端到端跑通验证（2026-06-18） | `app/rag/` | `docs/interview_prep_01_rag_pipeline.md` |
| 2. Multi-Agent 协作 | LangGraph搭建检索/执行/协调三角色，工具调用，跨轮次Memory | ✅ 代码完成，✅ 端到端跑通验证（2026-06-19） | `app/agents/`, `app/tools/` | `docs/interview_prep_02_multi_agent.md` |
| 3. 效果评测 | Ragas评测RAG链路，50条问答对评测集，参数调优 | ✅ 代码完成，✅ 端到端跑通验证（2026-06-21） | `app/eval/` | `docs/interview_prep_03_evaluation.md` |
| 4. 工程部署 | FastAPI流式接口、Prompt版本化、LangSmith追踪、Docker Compose | ✅ 代码完成，✅ 端到端跑通验证（2026-06-30） | `app/api/`, `docker/` | `docs/interview_prep_04_deployment.md` |

**当前真实进度**：**阶段1、2、3、4均已完整跑通**。阶段4（2026-06-30）：FastAPI服务启动，/health、/kb/status、/kb/build 全部验证通过；RAG模式非流式问答返回正确答案（年假15天）；Agent模式流式输出完整验证通过，tool_call/tool_result/token/done 事件均正确推送；运行命令：`.\venv\Scripts\uvicorn.exe app.api.main:app --host 0.0.0.0 --port 8000`。阶段1（2026-06-18）：RAG 检索+生成链路；阶段2（2026-06-19）：LangGraph Multi-Agent，三类工具串联 + Memory 跨轮次验证通过；阶段3（2026-06-21）：6份文档/28分块/95条评测集，基线(chunk_size=500,top_k=5) CR=0.9789/FA=0.7954/AR=0.8395，参数扫描后最优(chunk_size=800,top_k=8) Faithfulness提升至0.8943（+12.4%），Context Recall提升至0.9895（+1.1%）。详见各阶段面试文档。原阶段1说明——从文档解析、分块、BGE Embedding、FAISS+BM25混合检索，到 DeepSeek LLM 生成回答，完整链路全部验证通过。实测：3份示例文档建库7个分块，3条测试问题（年假天数/密码规则/知识库外问题）答案全部正确，其中"知识库外问题"触发了正确的拒答行为而非编造。新增文件：`app/rag/chain.py`（生成链）、`scripts/rag_query.py`（完整问答脚本）。阶段2/3/4 连代码都还没写。

如果有人（包括Claude Code）看到 `app/agents/`、`app/eval/`、`app/api/`、`docker/` 这些目录存在，**不代表里面已经有实现**——目前只是建项目骨架时预先创建的空目录占位，等阶段1验证通过后再按顺序实现。

---

## 三、各阶段的具体目标（来自最初的需求规划，含尚待验证的量化指标）

### 阶段1：RAG 核心链路
- 文档解析分块：PDF/Word → `RecursiveCharacterTextSplitter`
- Embedding：原始需求是 `text-embedding-3-small`，**实际改为可切换的Provider抽象**（默认本地BGE免费方案，OpenAI/智谱可选），原因见 `interview_prep_01_rag_pipeline.md` 第3节
- 向量存储：FAISS
- 混合检索：BM25 + 向量检索融合（RRF / 加权融合两种策略可切），目标"比纯向量检索准确率提升约15%" —— **待Ragas评测验证，目前是目标值**
- MMR去重，降低冗余召回

### 阶段2：Multi-Agent 协作（尚未开始）
- 基于 **LangGraph** 的 `StateGraph` 管理多智能体状态传递与任务分发
- 三类Agent角色：**检索Agent、执行Agent、协调Agent**
- 三类工具：搜索、计算器、数据库查询
- 跨轮次共享Memory（Context Engineering，需要设计具体怎么存、怎么在Agent间传递）
- 评测方式：设计20条混合查询测试集，单Agent/多Agent版本各跑10次取均值对比
- 目标："可并行任务响应时间降低约30%，强依赖串行任务因协调开销响应时间略有上升" —— **这是一个需要被验证、且预期会出现"有的场景变快、有的场景变慢"的真实工程结论，不是单纯吹多Agent架构多好。要验证清楚什么任务适合多Agent、什么任务不适合，这个结论本身比单纯的速度数字更有价值。**

### 阶段3：效果评测（尚未开始）
- 引入 **Ragas** 框架，覆盖三项指标：Context Recall、Faithfulness、Answer Relevancy
- 构建 **50条问答对评测集**（需要基于阶段1的示例文档真实编写，不是凑数）
- 迭代优化 chunk_size 与 top_k 参数
- 目标：Context Recall 从 0.61 提升至 0.79 —— **这是需要先跑出"优化前"基线、再跑"优化后"结果的真实对比实验，两个数字都要是实测的**

### 阶段4：工程部署（尚未开始）
- **FastAPI** 提供流式 REST 接口（Server-Sent Events），支持实时打字机输出
- Prompt模板版本化管理
- 关键节点接入 **LangSmith** 链路追踪
- **Docker Compose** 一键部署：API服务 + 向量库 + Redis会话缓存
- 实现知识库私有化部署，开源至GitHub

---

## 四、已经做出的关键技术选型（后续阶段要保持一致，不要重新讨论）

这些是已经和用户确认过的决策，Multi-Agent/评测/部署阶段的代码要延续这些选型，除非用户主动要求变更：

| 决策点 | 选型 | 原因 |
|---|---|---|
| LLM Provider | 默认 **DeepSeek**（deepseek-chat），可选智谱GLM/通义千问/OpenAI | 国内可用、有免费额度、性价比高；做成可切换接口体现工程能力 |
| Embedding Provider | 默认 **本地BGE**（bge-small-zh-v1.5），可选OpenAI text-embedding-3-small/智谱 | 完全免费离线，降低开源项目的使用门槛；可切换接口同时保留与原始需求(OpenAI)对标的能力 |
| 混合检索融合策略 | RRF（默认）+ 加权融合（可选） | 两种都实现，RRF更鲁棒因为不依赖原始分数量纲一致性，详见阶段1文档 |
| Agent框架 | LangGraph（StateGraph） | 按最初需求确定，未来如有变更需重新讨论 |
| API框架 | FastAPI + SSE | 按最初需求确定 |
| 部署方式 | Docker Compose（API + 向量库 + Redis） | 按最初需求确定 |
| 目标用户 | 开源到GitHub，面试展示用 | 决定了代码质量、文档完整度、可复现性的要求标准要往"生产级"靠，不是写完能跑就行 |

---

## 五、给后续会话（包括 Claude Code）的工作方式约定

1. **每完成一个模块，同步在对应的 `docs/interview_prep_0X_xxx.md` 里补充面试问答**，结构是：结论 → 为什么这么设计 → 可能被追问什么 → 已知未完成/待验证清单。这是用户明确要求的节奏，因为这个项目要拿去面试。
2. **任何调试/报错修复过程，都要按"现象→根本原因→修复方式→为什么这么修"的结构记录下来**，追加到相关文档里（比如环境问题记录在 `docs/windows_setup_troubleshooting.md`）。这些调试记录本身是面试时"项目中遇到的工程问题"的素材，不是可以省略的辅助信息。
3. **不编造尚未验证的量化数据**。所有效果对比数字（提升百分之多少、指标从多少到多少）必须基于真实跑出来的实验结果，没跑出来之前文档里要明确标注"目标值/待验证"。
4. **当前用户的本地环境**：Windows，正在从 Python 3.13 切换到 3.11 以避免依赖版本冲突（原因记录在 `docs/windows_setup_troubleshooting.md` 开头）。
5. 进入新会话/新工具（如Claude Code）时，建议先读本文档 + 已有的 `docs/interview_prep_01_rag_pipeline.md`，再开始写代码或调试，避免方向跑偏或重复讨论已经定好的设计决策。
