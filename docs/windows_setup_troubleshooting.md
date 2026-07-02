# Windows 本地运行指南 & 常见问题排查

## 〇、为什么换成 Python 3.11（真实踩坑记录，可写进面试经历）

**背景**：第一次 `pip install -r requirements.txt` 时，在 `unstructured` 这个包上报错：

```
ERROR: Could not find a version that satisfies the requirement unstructured<0.15,>=0.14
```

**排查过程**：报错信息里其实给了关键线索——pip 列出了一长串"因为 Python 版本不满足而被忽略"的版本，例如：

```
0.14.x Requires-Python >=3.9.0,<3.13
0.16.x Requires-Python >=3.9.0,<3.13
```

再结合同一份日志里 faiss 解析出的 wheel 文件名 `faiss_cpu-1.14.3-cp313-cp313-win_amd64.whl`（`cp313` = CPython 3.13），可以确认：**本地用的是 Python 3.13，而 `unstructured` 在 0.12~0.16 这个版本段官方只支持到 3.12，导致 requirements.txt 里指定的版本区间和本地 Python 版本完全没有交集**。

**两种修复思路**：
1. 把有问题的包逐个改版本号，凑出一个 3.13 下能装的组合
2. 直接换一个生态更成熟、被更多库官方支持的 Python 版本

选了第二种——3.13 发布时间很新，`langchain` / `faiss` / `sentence-transformers` 这些库对它的适配还在跟进，逐个改版本号大概率会在后面别的包上反复遇到同样问题。**3.11 是目前 AI/数据类 Python 库支持最成熟的版本之一**，一次性解决比反复试错效率更高。

（这同时也顺手发现并清理了 `requirements.txt` 里两个实际代码中从未 `import` 过的多余依赖：`unstructured` 和 `FlagEmbedding`，是写依赖清单时手松加上、但后续没用到的——这个过程本身就是个不错的面试素材："如何定位版本冲突的根因，而不是盲目试错改版本号"。）

## 〇-B、Python 3.11 安装 & 依赖安装实测记录（2026-06-18 完成）

**操作环境**：Windows 11，原有 Python 3.13.7，通过 winget 安装 3.11.9。

**步骤与结果**：

```powershell
# 安装 Python 3.11.9（通过 winget，静默安装约 1 分钟）
winget install Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements

# 确认版本
py -3.11 --version   # → Python 3.11.9

# 在项目根目录创建虚拟环境（无旧 venv，直接创建）
py -3.11 -m venv venv

# 升级 pip
.\venv\Scripts\python.exe -m pip install --upgrade pip   # pip 24.0 → 26.1.2

# 安装依赖（使用清华镜像，避免 PyPI 国内访问慢）
.\venv\Scripts\pip.exe install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**结果**：`Successfully installed` 25 个包，零冲突、零报错。

**核心验证**：
```python
import langchain, faiss, sentence_transformers, rank_bm25, jieba
import pypdf, docx, pdfplumber, fastapi, ragas
# → ALL OK
```

**关键安装包版本**（实测安装版本，供后续排查参考）：
- `langchain 0.2.17` / `langchain-core 0.2.43` / `langchain-community 0.2.19`
- `langgraph 0.1.19` / `langchain-openai 0.1.25`
- `sentence-transformers 3.4.1` / `transformers 4.57.6` / `huggingface-hub 0.36.2`
- `faiss-cpu 1.14.3`（已在 pip 首次运行时装好）/ `ragas 0.1.22`

**面试素材**：用镜像源加速这一步本身可以顺带提，但更有价值的是前面"定位根本原因（Python 3.13 与 unstructured 版本区间无交集）→ 选择换版本而非逐包试错"的决策思路。

---

## 〇-C、BGE 模型下载：huggingface_hub 0.36.2 镜像失效问题（2026-06-18）

**现象**：设置 `$env:HF_ENDPOINT = "https://hf-mirror.com"` 后运行 `build_index.py`，仍然报错：
```
FileMetadataError: Distant resource does not seem to be on huggingface.co.
```

**根本原因**：`huggingface_hub 0.36.2` 引入了对 endpoint 域名的严格校验——下载文件前会验证 HTTP 响应头是否符合 `huggingface.co` 的特定格式，`hf-mirror.com` 的响应头不满足这个检查，导致直接 `FileMetadataError`，与网络是否通畅无关（浏览器访问 `hf-mirror.com` 完全正常）。

**修复方式**：改用 **ModelScope（魔搭）** 下载模型，绕过 huggingface_hub 库：
```powershell
# 安装 modelscope（约 100MB）
.\venv\Scripts\pip.exe install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载 BGE 模型到本地（约 91MB，速度 3-7 MB/s）
.\venv\Scripts\python.exe -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='./models')"
# 实际落盘路径：./models/BAAI/bge-small-zh-v1___5
# 注意：ModelScope 把模型名中的 "." 替换为 "___"
```

然后在 `.env` 中将 `BGE_MODEL_NAME` 从远程 ID 改为本地绝对/相对路径：
```
BGE_MODEL_NAME=./models/BAAI/bge-small-zh-v1___5
```

**为什么这么修**：不降级 `huggingface_hub`（降版本会与 `transformers 4.57.6` 的版本约束冲突），也不改代码逻辑——只需在 `.env` 里把模型名改成本地路径，`SentenceTransformer(model_name)` 对路径和 Hub ID 都支持，零代码改动。ModelScope 作为国内备选下载源，比绕过 huggingface_hub 内部检查更可靠。

**面试素材**："同一个 Python 包（huggingface_hub）的小版本升级（0.25 → 0.36）引入了 breaking change，现象是'网络完全通、镜像站也能 ping 通，但下载就是失败'——这类问题排查时要先看版本 changelog，而不是一直刷新网络或换 VPN"。

---

## 一、安装 Python 3.11 并重建虚拟环境

1. 去 [python.org 官网下载页](https://www.python.org/downloads/release/python-3119/) 下载 **Windows installer (64-bit)**，安装时勾选 "Add python.exe to PATH"
2. 安装完成后，因为 3.13 和 3.11 会同时存在，用 `py` 启动器明确指定版本，而不是直接用 `python`（否则可能仍调用到 3.13）：

```powershell
# 确认 3.11 已安装并被识别
py -3.11 --version
# 应输出: Python 3.11.x

# 删除旧的（基于 3.13 创建的）虚拟环境
deactivate
rmdir /s /q venv

# 用 3.11 重新创建虚拟环境
py -3.11 -m venv venv
venv\Scripts\activate

# 确认虚拟环境内的版本正确
python --version
# 应输出: Python 3.11.x

python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果 `py -3.11` 提示找不到该版本，说明安装时启动器没注册成功，重新跑一次 3.11 的安装程序，安装界面里确认勾选了 "py launcher" 相关选项。

---

## 一、完整环境搭建步骤（已在 Windows 11 + Python 3.11.9 实测通过，2026-06-18）

```powershell
# 0. 安装 Python 3.11（如果系统是 3.13 需要先装 3.11，详见 §〇 节）
winget install Python.Python.3.11 --silent --accept-package-agreements --accept-source-agreements
py -3.11 --version   # 验证：应输出 Python 3.11.x

# 1. 进入项目根目录
cd enterprise-rag-agent

# 2. 用 Python 3.11 创建虚拟环境（重要：必须用 py -3.11 明确指定）
py -3.11 -m venv venv
venv\Scripts\activate
python --version   # 验证：应输出 Python 3.11.x

# 3. 升级 pip
python -m pip install --upgrade pip

# 4. 安装依赖（使用清华镜像加速，避免 PyPI 国内访问慢）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 准备配置文件
copy .env.example .env
# 然后填入 .env 里的关键配置（见下文说明）

# 6. 下载 BGE 模型（使用 ModelScope，避免 HuggingFace 在国内的访问问题）
pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple -q
python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', cache_dir='./models')"
# 模型会下载到 ./models/BAAI/bge-small-zh-v1___5（注意：ModelScope 把 "." 替换为 "___"）

# 7. 修改 .env 里的 BGE 路径（改成本地路径）
# 把 BGE_MODEL_NAME=BAAI/bge-small-zh-v1.5
# 改为 BGE_MODEL_NAME=./models/BAAI/bge-small-zh-v1___5

# 8. 建库
python scripts\build_index.py
# 预期输出末尾：✅ 建库完成: 7 个分块, 耗时 ~12s

# 9. 测试纯检索
python scripts\query_index.py "公司年假有多少天"

# 10. 测试完整 RAG 问答（需要先在 .env 里配置 LLM API Key，见下文）
python scripts\rag_query.py "公司年假有多少天"
```

### .env 必填配置项说明

```ini
# LLM（选一个填入对应 Key）
LLM_PROVIDER=deepseek          # 或 zhipu / qwen / openai
DEEPSEEK_API_KEY=sk-xxx        # 去 platform.deepseek.com 申请，充值后可用

# Embedding（已改为本地路径，无需 API Key）
EMBEDDING_PROVIDER=bge_local
BGE_MODEL_NAME=./models/BAAI/bge-small-zh-v1___5   # ModelScope 下载后的实际路径
```

---

## 二、Windows 上大概率会遇到的坑（提前说明原因）

### 坑 1：`pip install faiss-cpu` 失败或卡住

**现象**：安装 faiss-cpu 时报错或卡死不动。

**原因**：faiss 本身是 C++ 实现，pip 安装的是预编译的 wheel 包，Windows 上有时和 Python 版本不匹配（比如用了 Python 3.13 但 faiss-cpu 还没出对应 wheel）。

**怎么办**：
- 先确认 Python 版本：`python --version`，推荐 3.10 或 3.11（faiss-cpu 对这两个版本支持最成熟）
- 如果版本太新导致没有预编译包，换成 conda 安装会更稳：`conda install -c conda-forge faiss-cpu`

### 坑 2：BGE 模型下载卡住或超时

**现象**：跑 `build_index.py` 时卡在 "加载本地 BGE Embedding 模型" 这一行不动，或者报 `ConnectionError` / `SSLError`。

**原因**：`sentence-transformers` 默认从 HuggingFace Hub（`huggingface.co`）下载模型，国内网络访问该域名经常不稳定或被限速。

**怎么办**：用 HuggingFace 国内镜像，在跑脚本前设置环境变量：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python scripts\build_index.py
```

或者直接写进 `.env` 文件（需要我加一行读取逻辑，到时候告诉我用了镜像我来补这个配置项）。

### 坑 3：中文路径/文件名报错

**现象**：`UnicodeDecodeError` 或者读取文件时报路径找不到。

**原因**：Windows 默认编码在某些场景下不是 UTF-8（尤其是控制台输出），加上项目里文件名用了中文（`员工手册.md`），如果你的系统区域设置不是中文，可能在某些 Python 版本下读取文件名时出现编码问题。

**怎么办**：如果遇到这个问题，先尝试在 PowerShell 里执行：
```powershell
chcp 65001
```
切到 UTF-8 代码页再跑脚本。如果还不行，把报错完整发我，我可能需要在代码里显式指定 `encoding="utf-8"`（部分地方可能漏了）。

### 坑 4：`ModuleNotFoundError: No module named 'app'`

**现象**：直接用 `python app/rag/pipeline.py` 之类的方式跑某个模块文件报这个错。

**原因**：项目用的是绝对导入（`from app.core.config import settings`），这要求**从项目根目录**运行，且根目录要在 `sys.path` 里。`scripts/build_index.py` 和 `scripts/query_index.py` 已经在文件开头加了 `sys.path.insert(...)` 处理这个问题，但如果你想直接运行 `app/` 下某个模块做调试，需要同样的处理，或者用 `python -m app.rag.pipeline` 这种模块运行方式（保证在根目录执行）。

### 坑 5-B：BGE 模型下载报 FileMetadataError（huggingface_hub 版本问题）

> 详见 §〇-C，这里只给快速结论。

**现象**：设置 `HF_ENDPOINT=https://hf-mirror.com` 后仍报：
```
FileMetadataError: Distant resource does not seem to be on huggingface.co.
```

**根本原因**：`huggingface_hub 0.36.2` 新增了 endpoint 域名校验，hf-mirror.com 的响应头不符合要求。

**修复**：改用 ModelScope 下载，步骤见 §一 第 6 步。

---

### 坑 5：依赖版本冲突报错（最可能出现，也最难提前预测）

**现象**：`pip install -r requirements.txt` 跑到某个包报 `ERROR: Cannot install ... because these package versions have conflicting dependencies`。

**原因**：`langchain` 系列包（`langchain` / `langchain-core` / `langchain-community` / `langchain-openai` / `langgraph`）之间版本耦合很紧，我在 `requirements.txt` 里给的版本范围是基于已知能配合工作的组合写的，但**这套版本组合从未在真实环境里跑过 pip resolve**，存在冲突的可能性不低。

**怎么办**：这是最需要你把完整报错贴给我的场景。把 pip 报错的完整输出（尤其是哪两个包冲突、各自要求什么版本）发给我，我会针对性调整 `requirements.txt` 里的版本号。

---

### 坑 6：LLM API 报 402 Insufficient Balance

**现象**：`rag_query.py` 到最后一步报错：
```
openai.APIStatusError: Error code: 402 - {'error': {'message': 'Insufficient Balance', ...}}
```

**根本原因**：API Key 本身有效，但账户余额为 0。DeepSeek 注册后默认余额为零，需要手动充值才能调用。

**修复**：登录 platform.deepseek.com → 充值（哪怕充 1 元就够跑大量测试，deepseek-chat 极便宜）。

**排查思路**（面试素材）：这个报错容易被新人误判为"Key 无效"或"网络问题"，但区分很简单——
- 401 = Key 无效或格式错误
- 402 = Key 有效但账户无钱
- 429 = 请求速率超限（Key 有效有钱但调太快）

看错误码可以直接定位问题层次，不需要反复检查 Key 格式或网络连接。

---

## 三、调试时怎么把信息给我，能让我最快定位问题

请尽量提供：

1. **完整的报错堆栈**（不要只发最后一行，Python 的 Traceback 从上到下能看出问题出在哪一层调用）
2. **你执行的具体命令**
3. **如果是 pip install 报错**：贴出报错前 10-20 行，里面通常会写清楚是哪个包版本冲突
4. **你的 Python 版本**（`python --version`）

我收到报错后会按这个结构回复你：
- **现象**：报错说的是什么
- **根本原因**：为什么会发生（不是泛泛地说"环境问题"，而是具体到是哪个模块的什么假设不成立）
- **修复方式**：改哪个文件、改成什么
- **为什么这么修**：不只是"改了能跑"，而是这个修复方案背后的考虑（比如是否有更好的替代方案、为什么选了当前这种）

这样你积累的每一次调试记录，本身都能变成面试时"项目里遇到的工程问题"的素材——比单纯说"项目跑起来了"更有说服力。
