# Super Develop Agents

> 超级智能体开发平台 — 用智能体开发智能体，让 AI 自己构建 AI。

## 愿景

Super Develop Agents 的终极目标是成为**智能体开发领域的超级智能体**。我们不仅提供工具，而是让 AI 本身参与到智能体的设计、开发、测试和迭代全流程中。就像编译器编译自身、AI 训练 AI，Super Develop Agents 让智能体开发智能体。

## 架构

```
┌─────────────────────────────────────────────────┐
│              Super Develop Agents               │
│                                                 │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ 产品经理  │ │ 混沌测试  │ │   智测引擎        │ │
│  │ PM Agent │ │Chaos Agent│ │  AI-TestPilot    │ │
│  │          │ │          │ │                  │ │
│  │ 需求澄清  │ │ 故障注入  │ │ 用例生成·脚本生成 │ │
│  │ 需求分析  │ │ 质量评估  │ │ 缺陷分析·自动建单 │ │
│  │ PRD生成  │ │ 报告输出  │ │ 导出Excel        │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│                                                 │
│  ┌─────────────────────────────────────────────┐│
│  │          LangGraph Agent 引擎               ││
│  │  Worker Registry · Checkpoint · SSE Stream ││
│  └─────────────────────────────────────────────┘│
│                                                 │
│  ┌─────────────────────────────────────────────┐│
│  │              FastAPI 服务层                  ││
│  │         REST API · SSE · Static             ││
│  └─────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
```

## 功能模块

### 📝 产品经理 Agent

AI 驱动的产品需求管理，从一句话想法到完整 PRD。

| 阶段 | 功能 | 说明 |
|------|------|------|
| 1. Understand | 需求理解 | 解析用户输入的产品想法 |
| 2. Clarify | 需求澄清 | 多轮提问，精准挖掘需求 |
| 3. Analyze | 需求分析 | 结构化分析：用户画像、功能优先级、风险识别 |
| 4. Generate | PRD 生成 | 输出完整 PRD 文档，支持确认、修订、下载 |

**核心特性：**
- 多轮对话澄清模糊需求
- 自动生成功能优先级（P0/P1/P2）
- 输出含用户故事和验收标准的完整 PRD
- 基于 LangGraph 的状态机编排，支持中断恢复
- 历史对话持久化，随时回溯

### ☢ 混沌测试 Agent

对智能体进行鲁棒性测试，确保 AI 应用在面对异常输入时依然可靠。

| 场景 | 说明 |
|------|------|
| 文本噪声 | 注入拼写错误、语序混乱、冗余填充词 |
| 工具故障 | 模拟工具调用失败、超时、返回异常 |
| 话题漂移 | 引入不相关话题，测试上下文保持能力 |
| 边界测试 | 试探业务规则边界，检测越狱风险 |

**核心特性：**
- 4 种混沌注入场景
- 硬规则 + LLM 裁判双重评估
- 3 级严重度分级（高/中/低）
- 并发执行，实时 SSE 进度反馈
- 生成结构化测试报告

### 🧪 智测引擎 AI-TestPilot

AI 驱动的智能测试研发协同平台，重塑测试用例生成、自动化执行与缺陷闭环。

**用例生成：**
- 上传 PRD 文档（.md/.docx）或 Swagger 文件（.json/.yaml）
- LLM 自动生成结构化测试用例（含 ID、模块、前置条件、步骤、预期结果、优先级）
- 可编辑表格，支持人工微调
- 一键导出 Excel

**脚本生成：**
- 基于用例生成 Playwright（UI 自动化）或 Pytest（接口自动化）脚本
- Few-shot Prompt 工程
- 输出完整可运行代码

**缺陷分析：**
- 粘贴错误堆栈，AI 自动根因分析
- 提供精确到行的修复建议（Diff 格式）
- 定位问题文件路径和行号

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 框架 | **FastAPI** | 异步 Web 框架，原生支持 SSE 流式响应 |
| Agent 引擎 | **LangGraph** | 状态机驱动的多智能体工作流编排 |
| LLM | **Claude / Qwen** | 通过 Anthropic API 或 DashScope 代理接入 |
| 持久化 | **SQLite + LangGraph Checkpoint** | 对话状态持久化，支持中断恢复 |
| 前端 | **原生 HTML/CSS/JS** | 零依赖，轻量级 SPA |
| 包管理 | **uv / pip** | Python 依赖管理 |

## 快速开始

### 前置条件

- Python >= 3.11
- LLM API Key（Anthropic 或 DashScope）

### 安装

```bash
# 克隆项目
git clone https://github.com/2279184729/super_develop_agents.git
cd super_develop_agents

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
pip install openpyxl  # Excel 导出支持
```

### 配置

复制 `.env.example` 为 `.env`，填入 API 配置：

```bash
cp .env.example .env
```

```env
# LLM API 配置
ANTHROPIC_API_KEY=sk-your-api-key
ANTHROPIC_BASE_URL=https://dashscope.aliyuncs.com/apps/anthropic  # 可选
LLM_MODEL=qwen3.7-max

# 服务配置
API_HOST=0.0.0.0
API_PORT=8000
```

### 运行

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

打开浏览器访问 **http://localhost:8000**

## 项目结构

```
super_develop_agents/
├── src/
│   ├── agent/                    # 智能体核心
│   │   ├── pm_graph.py           # PM Agent 状态图
│   │   ├── pm_workers.py         # PM Worker（澄清/分析/生成）
│   │   ├── pm_orchestrator.py    # PM 编排器
│   │   ├── pm_state.py           # PM 状态定义
│   │   ├── chaos_graph.py        # Chaos Agent 状态图
│   │   ├── chaos_workers.py      # Chaos Worker（注入/评估/报告）
│   │   ├── chaos_connector.py    # 外部 Agent 连接器
│   │   ├── chaos_state.py        # Chaos 状态定义
│   │   ├── testpilot_agent.py    # 智测引擎 Agent
│   │   ├── registry.py           # Worker 注册中心
│   │   ├── llm_utils.py          # LLM 工具函数
│   │   ├── sse_utils.py          # SSE 流式工具
│   │   └── state.py              # 共享状态定义
│   ├── api/
│   │   └── main.py               # FastAPI 应用 + 路由
│   ├── storage/
│   │   └── checkpoint.py         # SQLite 持久化
│   └── ui/
│       ├── index.html            # 主框架（导航 + iframe）
│       ├── pm.html               # 产品经理页面
│       ├── chaos.html            # 混沌测试页面
│       └── testpilot.html        # 智测引擎页面
├── tests/                        # 测试
│   ├── unit/                     # 单元测试
│   ├── integration/              # 集成测试
│   └── interface/                # 接口测试
├── data/                         # SQLite 数据库
├── pyproject.toml                # 项目配置
└── requirements.txt              # 依赖列表
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 主页面 |
| `GET` | `/api/health` | 健康检查 |
| `POST` | `/api/pm/stream` | PM Agent 流式对话 |
| `POST` | `/api/pm/answer/{id}` | 提交澄清回答 |
| `POST` | `/api/pm/confirm/{id}` | 确认/修订 PRD |
| `GET` | `/api/pm/state/{id}` | 获取 PM 状态 |
| `GET` | `/api/pm/download/{id}` | 下载 PRD |
| `GET` | `/api/pm/threads` | PM 对话列表 |
| `POST` | `/api/chaos/run` | 执行混沌测试 |
| `GET` | `/api/chaos/state/{id}` | 获取测试状态 |
| `GET` | `/api/chaos/report/{id}` | 获取测试报告 |
| `GET` | `/api/chaos/threads` | 测试记录列表 |
| `POST` | `/api/testpilot/generate-cases` | 生成测试用例 |
| `POST` | `/api/testpilot/generate-scripts` | 生成自动化脚本 |
| `POST` | `/api/testpilot/analyze-defect` | 缺陷分析 |
| `POST` | `/api/testpilot/export-cases` | 导出 Excel |

## 路线图

| 阶段 | 内容 | 状态 |
|------|------|------|
| ✅ MVP | PM Agent + Chaos Agent + 智测引擎 P0 | 已完成 |
| 🔜 V1.0 | CI/CD 集成、智能回归测试、多维度报告 | 规划中 |
| 🔜 V2.0 | 大模型专项测试、历史缺陷学习、UI 遍历测试 | 规划中 |
| 🎯 终极 | 智能体自举 — Agent 自动设计、开发和优化 Agent | 愿景 |

## License

MIT