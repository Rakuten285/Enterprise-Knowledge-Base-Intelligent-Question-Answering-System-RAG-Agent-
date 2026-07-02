# 面试准备文档：Multi-Agent 协作系统

> 本文档与代码同步维护。**实测日期：2026-06-19，全部跑通。**
> 所有测试结果均为真实运行输出，不是预期值。

---

## 1. 整体架构：为什么要用 Multi-Agent，而不是给一个 LLM 挂一堆工具？

**问：你的 Multi-Agent 系统是怎么设计的？和单 Agent + 多工具有什么区别？**

```
用户问题
    ↓
Coordinator Agent（协调者）
    ├── 决策：需要查内部文档 → Retrieval Agent → 执行 search_knowledge_base
    ├── 决策：需要查员工数据或计算 → Executor Agent → 执行对应工具
    └── 决策：信息已充足 → 直接输出最终回答
         ↑__________________（结果反馈，循环直到完成）
```

**三个角色的分工：**

| 角色 | 职责 | 对应代码 |
|---|---|---|
| Coordinator（协调者） | 分析问题，决定调用哪个工具，汇总结果，生成最终回答 | `agents/coordinator.py` |
| Retrieval Agent（检索者） | 专门负责知识库搜索，复用阶段1全部成果 | `agents/retrieval_agent.py` |
| Executor Agent（执行者） | 负责计算器、数据库查询、网络搜索三类工具 | `agents/executor_agent.py` |

**为什么要分开，不直接让 Coordinator 自己执行所有工具？**

单 Agent + 多工具确实能解决大多数问题，但拆成多 Agent 有两个工程价值：

1. **职责边界清晰，易于独立扩展**：想换 Embedding 模型？只改 Retrieval Agent 内部，Coordinator 和 Executor 不用动。想加一类新工具（比如发邮件）？只需在 Executor 里加一个分支，不影响检索逻辑。这是"单一职责原则"的直接应用。

2. **为将来的并行执行留了架构位置**：当前版本是串行的（Coordinator 决策 → 一个 Agent 执行 → 回 Coordinator），但如果某个问题同时需要查知识库和查数据库（比如"李娜的年假余额是多少天"需要同时查 HR 文档和员工表），可以在 LangGraph 里把 Retrieval Agent 和 Executor Agent 改成并行节点，而不需要重写任何 Agent 内部逻辑。

**追问：你说"为将来留了位置"，现在有没有并行？**
当前版本没有实现并行——串行更容易调试，对 demo 场景足够。如果要实现并行，LangGraph 支持在 `add_conditional_edges` 里返回一个列表而不是单个字符串来触发并行分支，但这会增加状态合并的复杂度，留到性能优化阶段再做，目前先把正确性验证清楚。

---

## 2. LangGraph：为什么选它，StateGraph 怎么工作的？

**问：你为什么用 LangGraph 而不是 LangChain 的 AgentExecutor？**

`AgentExecutor` 是 LangChain 早期的 Agent 运行时，问题在于：
1. 控制流是固定的（LLM → 工具调用 → LLM → ...），没法灵活插入"换一个 Agent 来执行"
2. 状态不透明——中间过程不容易观察和调试
3. 不支持跨轮次 Memory 的持久化（或者说支持得很粗糙）

LangGraph 的核心抽象是 **StateGraph**：把整个 Agent 工作流表示成一张有向图，节点是函数，边是路由逻辑，状态在节点之间流动和累积。

```python
graph = StateGraph(AgentState)
graph.add_node("coordinator", coordinator_node)      # 节点 = 普通 Python 函数
graph.add_node("retrieval_agent", retrieval_agent_node)
graph.add_node("executor_agent", executor_agent_node)
graph.set_entry_point("coordinator")

graph.add_conditional_edges(                          # 条件边 = 路由函数
    "coordinator",
    route_after_coordinator,                          # 返回字符串决定去哪个节点
    {"retrieval_agent": ..., "executor_agent": ..., "END": END}
)
graph.add_edge("retrieval_agent", "coordinator")      # 固定边 = 无条件跳转
graph.add_edge("executor_agent", "coordinator")
```

**State 是怎么流动的？**

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # 累加，不覆盖
    query: str
    iterations: int
```

关键是 `Annotated[..., operator.add]`——每个节点返回的 `messages` 列表会被 **追加** 到全局 State，而不是替换。这样整条推理链（Human消息 → Coordinator的tool_call → ToolMessage结果 → 最终AIMessage）都完整保留，既是 Memory，也是可以回放的调试轨迹。

**追问：operator.add 在这里是什么意思？**
它是 LangGraph 的 "reducer" 机制——当多个节点同时修改同一个 State 字段时，LangGraph 需要知道怎么合并。`operator.add` 告诉它"用列表拼接"。如果不加这个注解，后一次节点的返回会直接覆盖前一次，对话历史就丢了。

---

## 3. 工具设计：四类工具的选型逻辑

### 3.1 为什么计算器要用 AST 解析，不直接用 `eval()`？

**问：你的计算器工具安全吗？**

直接 `eval(expression)` 会执行任意 Python 代码，比如用户输入 `__import__('os').system('del /f /q C:\\')` 就能在服务器上执行系统命令。这是典型的代码注入漏洞。

我用 `ast.parse()` + 节点类型白名单的方式：
```python
_SAFE_OPS = {
    ast.Add: op.add,   # 只允许这几种操作符
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
}
```
先把表达式 parse 成 AST，然后递归遍历每个节点——遇到不在白名单里的节点类型（比如 `ast.Call`、`ast.Import`）直接抛异常，只有纯数学运算能走通。

**为什么这个设计值得在面试里提**：它是安全编程意识的具体体现——不是"我觉得用户不会乱输入"，而是从架构上消除了这个可能性。任何 LLM 应用里有工具调用的地方都要考虑这个问题，因为 LLM 的输出是不受控的。

### 3.2 数据库工具为什么只允许 SELECT？

同样是安全边界问题：工具是由 LLM 驱动调用的，LLM 的输出不可 100% 信任。如果允许 `DELETE`/`UPDATE`，LLM 幻觉出一条错误 SQL 可能会毁掉数据。代码里做了最简单的前缀检查：

```python
if not sql.strip().upper().startswith("SELECT"):
    return "安全限制：只允许 SELECT 查询，不允许修改数据。"
```

这不是完善的 SQL 注入防护（完善的做法是参数化查询 + 数据库层面权限隔离），但对于 demo 场景够用，而且限制思路是正确的。

**追问：如果真的要做生产级数据库工具怎么办？**
- 给数据库用户只授 SELECT 权限（数据库层面强制，代码层面的检查只是双保险）
- 用参数化查询防止 SQL 注入（`cursor.execute("SELECT ... WHERE name=?", (name,))`）
- 限制可查询的表（不能查 users 表的密码字段之类）

### 3.3 web_search 为什么优雅降级而不是必须安装？

DuckDuckGo 搜索是"锦上添花"的功能：知识库 + 数据库 + 计算器已经能覆盖这个 demo 的核心场景，web_search 只在"知识库里没有且需要实时信息"时才有价值。

用 `try/except ImportError` 让它变成可选依赖：
```python
try:
    from duckduckgo_search import DDGS
except ImportError:
    return "web_search 工具未就绪：缺少 duckduckgo-search 依赖..."
```

这样 `requirements.txt` 里可以不强制要求这个包（国内网络下 DuckDuckGo 并不总是可用），不会因为一个可选工具让整个系统装不起来。

---

## 4. Memory：跨轮次对话是怎么实现的？

**问：你的 Memory 是怎么做的，用户说"他"，系统怎么知道"他"是谁？**

两层机制合在一起：

**第一层：`operator.add` 消息累积**

State 里的 `messages` 字段会把每轮对话的所有消息（Human/AI/Tool）都追加进去。第二轮对话时，Coordinator 看到的 messages 包含了第一轮的完整记录，LLM 自然能从上下文里解析"他"指代的是赵磊。

**第二层：LangGraph `MemorySaver` 持久化**

```python
memory = MemorySaver()
compiled_graph = graph.compile(checkpointer=memory)
```

`MemorySaver` 把每次图执行后的完整 State 序列化到内存（生产场景可以换成 `SqliteSaver` 或 `PostgresSaver` 持久化到磁盘/数据库）。下次用**相同的 `thread_id`** 调用时，LangGraph 自动从检查点恢复上次的 State。

```python
config = {"configurable": {"thread_id": "user-123"}}
app.stream(state_input, config=config)  # 同一 thread_id = 同一会话
```

**实测验证（2026-06-19）：**
- 第一轮：`技术部工资最高的员工是谁？` → 系统查库后回答"赵磊，月薪 45000"
- 第二轮（同一 thread_id）：`他的入职日期是什么时候？` → 系统**没有重新问"他是谁"**，直接查询赵磊的入职日期，返回 2017-02-14

**追问：MemorySaver 和 ConversationBufferMemory 有什么区别？**

`ConversationBufferMemory` 是 LangChain 的旧式 Memory 方案，只保存 `(Human, AI)` 对话对，不包含工具调用的中间过程。LangGraph 的 `MemorySaver` 保存的是完整的 State 快照（包含所有 ToolMessage），粒度更细，而且和图的执行机制深度集成——每个节点执行完后自动打快照，断电恢复也能从上一个节点继续，这对长流程 Agent 任务很重要。

---

## 5. 路由机制：Coordinator 怎么"决策"去哪个 Agent？

**问：协调者怎么知道该调用哪个工具？**

Coordinator 通过 LLM 的 **tool calling（函数调用）** 能力来做决策：

```python
_TOOLS = [search_knowledge_base, calculate, query_employee_database, web_search]
llm_with_tools = get_llm().bind_tools(_TOOLS)
response = llm_with_tools.invoke(messages)
```

`bind_tools` 会把四个工具的名称、描述、参数 schema 以 JSON Schema 格式传给 LLM。LLM 分析用户问题后，在回复里附带一个 `tool_calls` 字段（而不是直接生成文字回答），指定要调用哪个工具、参数是什么。

然后 LangGraph 的路由函数读取这个字段：

```python
def route_after_coordinator(state: AgentState) -> str:
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return "END"              # 没有 tool_calls = 直接给出最终回答
    tool_name = last.tool_calls[0]["name"]
    if tool_name == "search_knowledge_base":
        return "retrieval_agent"
    if tool_name in ("calculate", "query_employee_database", "web_search"):
        return "executor_agent"
    return "END"
```

**追问：工具的"描述"（docstring）重要吗？**

非常重要——LLM 完全根据描述来判断该调用哪个工具。比如 `search_knowledge_base` 的描述写的是"企业内部文档（员工手册、IT安全规范、采购管理制度）"，LLM 看到"年假"相关的问题就会选它而不是 `web_search`。描述写得模糊，工具选错的概率就高。这和提示工程一样，是工程细节但影响很大。

**追问：如果 LLM 选错了工具怎么办？**

当前版本没有专门的"工具选择纠错"机制。实测中 DeepSeek 选择工具的准确率比较高（主要靠清晰的工具描述）。工程上可以加的：
1. 给 Coordinator 的 System Prompt 加更多的使用场景示例（few-shot）
2. 限制每个工具最多调用 N 次（已实现 `MAX_ITERATIONS=6`）
3. 如果工具返回"知识库中未找到相关信息"，允许 Coordinator 再尝试一次不同的工具

---

## 6. 实测结果（2026-06-19，全部通过）

### 测试1：知识库检索
- 问：`公司年假有多少天？`
- 路径：`Coordinator → search_knowledge_base → Coordinator → 最终回答`
- 结果：正确，按工作年限三档输出，引用《员工手册》
- 说明：单工具调用，验证知识库路径

### 测试2：数据库 + 计算器串联
- 问：`技术部所有员工的平均月薪是多少？`
- 路径：`Coordinator → query_employee_database → Coordinator → calculate → Coordinator → 最终回答`
- 结果：正确，查出4名技术部员工薪资，自动生成计算表达式 `(28000+18000+45000+12000)/4=25750`
- 说明：Coordinator 自主决定"先查再算"，无需人工指定顺序

### 测试3：Memory 跨轮次
- 第一轮：`技术部工资最高的员工是谁？` → 赵磊，45000元
- 第二轮（同一 thread_id）：`他的入职日期是什么时候？` → 赵磊，2017-02-14
- 说明：第二轮没有重新查谁是技术部最高薪员工，直接从上下文理解"他"=赵磊

---

## 7. 已知未完成 / 待验证事项

1. **并行执行未实现**：当前串行，适合 demo。复杂查询可改为并行分支，LangGraph 语法支持，留到评测阶段确认价值后再做。
2. **工具选择错误无自动纠错**：当前依赖 LLM 自身准确性，偶发选错时没有回退机制。
3. **MemorySaver 是内存级存储**：进程重启后历史丢失。生产场景应换成 `SqliteSaver` 或 Redis 持久化。
4. **web_search 在国内不稳定**：DuckDuckGo 在国内网络环境下成功率不保证，已做优雅降级但没有备选搜索源。
5. **没有做 Multi-Agent 的性能评测**：文档里的"并行任务响应时间降低30%"仍是目标值，等阶段3 Ragas 评测完成后，需要单独设计 Multi-Agent 性能测试集（20条混合查询，单/多 Agent 各跑10次取均值）。
