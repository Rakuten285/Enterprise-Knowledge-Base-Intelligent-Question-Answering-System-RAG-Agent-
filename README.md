# 企业知识库智能问答系统

基于 RAG + LangGraph Multi-Agent 的企业内部知识问答系统，支持流式 API、Docker 一键部署。

## 功能亮点

- **混合检索**：FAISS 向量检索 + BM25 关键词检索，RRF 融合排序 + MMR 去重
- **本地 Embedding**：BGE-small-zh-v1.5 离线运行，零 API 费用
- **Multi-Agent**：LangGraph StateGraph 协调三类 Agent（检索 / 执行 / 协调），支持知识检索、计算器、员工数据库查询
- **流式 API**：FastAPI + SSE，逐 token 推送，前端实时打字机效果
- **量化评测**：Ragas 框架，95 条问答评测集，参数扫描确定最优配置
- **一键部署**：Docker Compose（API + Redis）

## 实测效果

Ragas 评测（6 份企业文档 / 28 个分块 / 95 条问答对）：

| 参数配置 | Context Recall | Faithfulness | Answer Relevancy |
|---|---|---|---|
| 基线 chunk=500, top_k=5 | 0.9789 | 0.7954 | 0.8395 |
| **最优 chunk=800, top_k=8** | **0.9895** | **0.8943** | **0.8612** |
| 提升 | +1.1% | **+12.4%** | +2.6% |

参数扫描覆盖 3×3=9 种组合（chunk_size: 300/500/800，top_k: 3/5/8），最优配置已写入 `.env`。

## 技术栈

| 模块 | 技术选型 |
|---|---|
| LLM | DeepSeek（默认）/ 智谱 GLM / 通义千问 / OpenAI |
| Embedding | BGE-small-zh-v1.5（本地）/ OpenAI / 智谱 |
| 向量库 | FAISS |
| 检索 | BM25 + 向量混合，RRF 融合 |
| Agent 框架 | LangGraph StateGraph |
| 评测 | Ragas（ContextRecall / Faithfulness / AnswerRelevancy） |
| API | FastAPI + SSE 流式输出 |
| 部署 | Docker Compose |
| 追踪 | LangSmith（可选） |

## 快速开始

### 本地运行

前置要求：Python 3.11，DeepSeek API Key（[申请地址](https://platform.deepseek.com)，注册有免费额度）

```bash
git clone https://github.com/Rakuten285/Enterprise-Knowledge-Base-Intelligent-Question-Answering-System-RAG-Agent-.git
cd Enterprise-Knowledge-Base-Intelligent-Question-Answering-System-RAG-Agent-

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 安装依赖（国内镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 配置密钥
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

下载 BGE 模型（约 400MB，仅首次需要）：

```bash
# 下载到本地 models/ 目录（推荐，与 .env 默认路径一致）
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
model.save('./models/BAAI/bge-small-zh-v1___5')
"
# 如果 HuggingFace 访问慢，可改用镜像：
# HF_ENDPOINT=https://hf-mirror.com python -c "..."
```

构建知识库、初始化数据库并启动服务：

```bash
# 构建 RAG 索引（处理 data/raw/ 下的文档）
python scripts/build_index.py

# 初始化员工数据库（Agent 工具依赖，10 条示例员工数据）
python scripts/init_db.py

# 启动 API 服务
venv\Scripts\uvicorn.exe app.api.main:app --host 0.0.0.0 --port 8000  # Windows
# uvicorn app.api.main:app --host 0.0.0.0 --port 8000                 # macOS/Linux
```

访问 http://localhost:8000/docs 查看 Swagger UI。

### Docker 部署

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

cd docker
docker-compose up -d

# 首次启动后构建知识库
curl -X POST http://localhost:8000/api/v1/kb/build \
     -H "Content-Type: application/json" \
     -d '{"chunk_size": 800, "chunk_overlap": 50}'

# 初始化员工数据库（可选，Agent 工具使用）
docker exec rag-agent-api python scripts/init_db.py
```

## API 使用

### 流式问答（SSE）

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "员工年假最多有多少天？", "mode": "agent", "stream": true}'
```

响应为 Server-Sent Events 流：

```
data: {"event": "tool_call", "data": {"tool": "search_knowledge_base", "args": {...}}}
data: {"event": "tool_result", "data": {"tool": "search_knowledge_base", "preview": "..."}}
data: {"event": "token", "data": "根"}
data: {"event": "token", "data": "据"}
...
data: {"event": "done", "data": {"answer": "...", "sources": [...], "thread_id": "..."}}
```

`mode` 参数：
- `agent`（默认）：走 LangGraph Multi-Agent，支持工具调用，可见中间推理步骤
- `rag`：直接走 RAG 检索生成，响应更快，适合简单查询

### 知识库管理

```bash
# 查看知识库状态（文档列表、分块数）
GET /api/v1/kb/status

# 重建索引（更新文档后调用）
POST /api/v1/kb/build
{"chunk_size": 800, "chunk_overlap": 50}
```

## 添加自己的文档

把 `.md`、`.pdf`、`.docx` 文件放到 `data/raw/`，然后重建索引：

```bash
curl -X POST http://localhost:8000/api/v1/kb/build \
     -H "Content-Type: application/json" \
     -d '{"chunk_size": 800, "chunk_overlap": 50}'
```

## 项目结构

```
enterprise-rag-agent/
├── app/
│   ├── api/          # FastAPI 路由（chat / kb）
│   ├── agents/       # LangGraph Multi-Agent 图
│   ├── rag/          # RAG 核心链路（加载/分块/检索/生成）
│   ├── tools/        # 工具（知识检索/计算器/员工数据库）
│   ├── eval/         # Ragas 评测
│   └── core/         # 配置/日志/LLM工厂
├── data/
│   ├── raw/          # 原始文档（Markdown/PDF/Word）
│   ├── vectorstore/  # FAISS 索引（自动生成）
│   └── eval_results/ # 评测结果 JSON
├── docker/           # Dockerfile + docker-compose.yml
├── docs/             # 架构文档与设计说明
├── models/           # BGE 本地模型（自行下载后放这里）
├── scripts/          # 构建索引 / 查询 / 评测脚本
└── .env.example      # 环境变量示例
```

## 运行 Ragas 评测

```bash
# 基线评测（使用当前 .env 参数）
python scripts/run_eval.py

# 参数扫描（9 种 chunk_size × top_k 组合）
python scripts/run_eval.py --sweep

# 快速验证（只跑前 5 条）
python scripts/run_eval.py --sample 5

# 检查评测集数据质量
python scripts/check_eval_dataset.py --fail-only
```

## LangSmith 链路追踪（可选）

在 `.env` 中配置：

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=enterprise-rag-agent
```

启动后即可在 [LangSmith Dashboard](https://smith.langchain.com) 看到每次请求的完整调用链、token 消耗和耗时分布。

## 主要环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（必填） | - |
| `LLM_PROVIDER` | LLM 提供商 | `deepseek` |
| `EMBEDDING_PROVIDER` | Embedding 提供商 | `bge_local` |
| `CHUNK_SIZE` | 文档分块大小 | `800` |
| `TOP_K` | 检索返回分块数 | `8` |
| `HYBRID_FUSION_STRATEGY` | 混合检索融合策略 | `rrf` |

完整配置见 `.env.example`。

## License

MIT
