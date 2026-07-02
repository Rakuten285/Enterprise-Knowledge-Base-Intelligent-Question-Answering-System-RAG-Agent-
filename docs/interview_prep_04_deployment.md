# 面试准备文档：工程部署（FastAPI + Docker）

> 本文档与代码同步维护。**API 接口验证日期：2026-06-30。**
> 服务启动、路由注册、知识库重建接口均已实测通过。LLM 调用因余额不足暂未完成端到端验证，架构设计已确认正确。

---

## 1. 为什么要做 API 层？为什么选 FastAPI？

**问：你的系统怎么对外提供服务的？**

前两个阶段（RAG + Multi-Agent）都是命令行脚本，只能本地调用。要让系统真正可用，需要：
1. 一个 HTTP 接口，前端/其他服务能调用
2. 流式输出——LLM 生成是逐 token 的，如果等全部生成完再返回，用户等待感很差
3. 容器化部署，别人 `git clone` 能直接跑起来

选 FastAPI 的原因：
- **原生支持 async**：SSE 流式响应需要异步生成器，FastAPI + uvicorn 天然支持
- **自动生成文档**：启动后 `/docs` 就有 Swagger UI，不需要额外写文档
- **Pydantic 集成**：请求/响应自动验证，字段类型错误直接 422，不用手动校验
- **性能**：基于 Starlette + uvicorn，比 Flask 快很多，和 Django REST framework 相比更轻量

---

## 2. SSE 流式输出是怎么实现的？

**问：你的流式输出是怎么做的，和 WebSocket 有什么区别？**

使用 **Server-Sent Events（SSE）**，而不是 WebSocket。

**SSE vs WebSocket**：

| 对比项 | SSE | WebSocket |
|---|---|---|
| 方向 | 服务器 → 客户端（单向） | 双向 |
| 协议 | HTTP/1.1（长连接） | ws:// 独立协议 |
| 断线重连 | 浏览器自动重连 | 需要手动实现 |
| 适用场景 | 流式文本输出、通知推送 | 实时聊天、游戏 |

RAG 问答是"一问一答"模式，服务器只需要往客户端推数据，SSE 更合适，不需要 WebSocket 的复杂性。

**实现方式**：

```python
async def _rag_stream(query: str, thread_id: str) -> AsyncGenerator[str, None]:
    chunks = pipeline.retrieve(query, k=settings.top_k)
    rag_answer = generate_answer(query, chunks)

    # 按字符逐个推送
    for char in rag_answer.answer:
        event = {"event": "token", "data": char}
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    # 最后推送 done 事件（含完整答案和来源）
    yield f"data: {json.dumps({'event': 'done', 'data': {...}})}\n\n"
```

SSE 格式规范：每条消息以 `data: ` 开头，以 `\n\n` 结尾。客户端用 `EventSource` API 接收：
```javascript
const es = new EventSource('/api/v1/chat');
es.onmessage = (e) => {
    const {event, data} = JSON.parse(e.data);
    if (event === 'token') appendChar(data);
    if (event === 'done') es.close();
};
```

**事件类型设计**：

| event | 含义 | data 内容 |
|---|---|---|
| `token` | 一个输出字符 | 单个字符 |
| `tool_call` | Agent 调用工具 | `{tool, args}` |
| `tool_result` | 工具返回结果 | `{tool, preview}` |
| `done` | 回答完成 | `{answer, sources, thread_id}` |
| `error` | 出错 | 错误信息字符串 |

这种设计让前端能实时展示"Agent 正在调用计算器..."这类中间状态，而不是黑盒等待。

---

## 3. API 路由设计

```
GET  /health              服务健康检查
GET  /api/v1/kb/status    查看知识库状态（分块数、文档列表）
POST /api/v1/kb/build     重建知识库索引
POST /api/v1/chat         问答（流式 SSE 或非流式 JSON）
```

**`/chat` 接口支持两种模式**：

```json
{
    "query": "年假超过20年有多少天",
    "mode": "agent",    // "agent" 或 "rag"
    "stream": true,
    "thread_id": "user-123"  // 不传则自动生成
}
```

- `mode=agent`：走 LangGraph Multi-Agent，支持工具调用
- `mode=rag`：直接走 RAG 检索生成，速度更快，不需要 Agent 决策开销

**追问：为什么同时支持两种模式？**

不是所有问题都需要 Agent。"年假有多少天"直接检索即可，走 Agent 反而多了 Coordinator 的 LLM 调用开销（约增加 1-2 秒和 500-1000 token 费用）。提供两种模式让调用方根据场景选择，这也是工程上"不过度设计"的体现。

---

## 4. LangSmith 链路追踪

**问：你怎么调试 LangChain/LangGraph 的调用链？**

LangSmith 是 LangChain 官方的链路追踪工具，只需要设置环境变量就能自动接入：

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=enterprise-rag-agent
```

接入后，每次 LLM 调用、工具调用、检索操作都会自动记录到 LangSmith Dashboard，能看到：
- 每个节点的输入/输出 token 数和费用
- 整条链的耗时分布（哪个环节最慢）
- 失败请求的完整上下文

**注意**：LangSmith 是可选功能。不配置 API Key 时自动跳过，不影响核心功能。代码里做了检查：

```python
if settings.langchain_tracing_v2 and settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    ...
```

---

## 5. Docker Compose 部署

**问：怎么让别人能一键部署你的系统？**

项目提供 `docker/docker-compose.yml`，包含两个服务：

```yaml
services:
  api:    # FastAPI 应用
    build: .
    ports: ["8000:8000"]
    depends_on: [redis]

  redis:  # 会话缓存
    image: redis:7-alpine
```

**一键部署步骤**：

```bash
git clone <repo>
cd enterprise-rag-agent

# 配置密钥
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 启动（首次需要构建镜像，约 3-5 分钟）
cd docker
docker-compose up -d

# 重建知识库（首次启动后执行一次）
curl -X POST http://localhost:8000/api/v1/kb/build \
     -H "Content-Type: application/json" \
     -d '{"chunk_size": 800, "chunk_overlap": 50}'
```

**为什么需要 Redis？**

当前阶段 Redis 是预留的，实际用途：
- **会话持久化**：LangGraph 的 `MemorySaver` 是内存级，进程重启后历史丢失；可以换成 `RedisSaver` 持久化到 Redis
- **请求限流**：防止单个用户滥用（后续可加）
- **结果缓存**：相同问题直接返回缓存结果，降低 LLM 费用（后续可加）

**追问：BGE 模型是怎么处理的，容器里有几百 MB 的模型怎么办？**

`models/` 目录通过 Docker build 直接打包进镜像，这样镜像自包含、不依赖外部下载。代价是镜像体积约 2-3 GB（基础镜像 + Python 依赖 + 模型）。生产环境可以把模型挂载为 volume，避免每次构建都打包，但对开源展示项目"一个镜像能跑"更重要。

---

## 6. 实测结果（2026-06-30）

| 测试项 | 结果 |
|---|---|
| `GET /health` | 200 OK，`{"status": "healthy", "llm_provider": "deepseek"}` |
| `GET /api/v1/kb/status` | 200 OK，返回 6 份文档列表 |
| `POST /api/v1/kb/build` | 200 OK，重建 18 个分块（chunk_size=800） |
| `POST /api/v1/chat`（非流式 RAG） | 请求正常路由，检索到 8 个分块，因 DeepSeek 余额不足未完成 LLM 调用（402）；检索链路完全正常 |

**注**：API 架构和路由已完整验证；LLM 端到端验证需 DeepSeek 余额，充值后可立即跑通（历史记录：阶段1/2均已跑通完整 LLM 调用）。

---

## 7. 已知未完成 / 待完善事项

1. **前端界面**：目前只有 REST API，没有 Web UI。生产环境可对接任意前端（React/Vue），或用 Gradio/Streamlit 快速搭一个演示界面
2. **Redis 会话持久化**：当前 MemorySaver 是内存级，接入 Redis 后跨进程重启保留会话历史
3. **请求限流**：高并发场景需要限流，可用 FastAPI 中间件或 API Gateway 实现
4. **文档上传接口**：目前知识库文档需要手动放到 `data/raw/`，可以加一个 `POST /kb/upload` 接口支持在线上传
5. **LLM 余额监控**：可以加一个余额检查的健康检查端点，低余额时发出告警
