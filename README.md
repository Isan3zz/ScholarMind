# 🎓 ScholarMind — 学术论文智能阅读助手

> **ScholarMind** 是一个基于 **LangGraph 多节点状态机** 的学术论文深度阅读与报告生成系统。支持论文上传、结构化检索、自动撰写与自校正闭环。
>
> ⚠️ **ScholarMind 是单 Agent 系统**——整个 StateGraph 编译产物就是那唯一的 Agent。Router、Planner、Researcher 等是它内部的**工作流节点**，各有独立 prompt 和职责，但不是独立 Agent。

### 演示截图

![ScholarMind 主界面](./docs/image.png)

---

## ✨ 核心特性

### 🧠 单 Agent · 多节点工作流 (LangGraph StateGraph)

整个 StateGraph 是一个 Agent，内部 6 个 LLM 节点各司其职，共享 `AgentState` 状态，外加一个后处理步骤持久化会话记忆：

| 节点 | 职责 | 模型 |
|------|------|------|
| **Router** | 意图识别，写入 state 供下游路由 | fast |
| **Planner** | 将用户问题拆解为 3-5 个检索 query | fast (temp=0.7) |
| **Researcher** | 本地检索 + 自主判断相关性 + 可选的网络搜索 | smart (temp=0) |
| **Writer** | 根据检索证据撰写完整报告 | fast |
| **Reviewer** | 审查报告质量（规则+LLM），PASS/FAIL | smart (temp=0) |
| **Refiner** | 在已有报告上编辑或增补 | fast |
| *(后处理)* | 持久化 short_memory 到 checkpoint（不在 graph 拓扑中） | — |

### 🔀 三条交互路径

```
new_topic:      router → planner → researcher → writer → reviewer ⇄ planner (≤3次) → END
augment_report: router → planner → researcher → refiner → reviewer ⇄ planner (≤3次) → END
edit_report:    router → refiner → END
```

- **开始新分析**: 完整检索+撰写+审查，最多 3 轮自校正
- **修改表达**: 仅编辑已有报告，不重新检索
- **补充证据**: 重新检索→在旧报告上增补→审查
- **回车自由输入**: 不点按钮直接 Enter → Router 的 LLM 自动判断意图

### 🔍 多层检索流水线

```
BM25(30条) + Dense(30条) → RRF融合 → qwen3-rerank精排 → Section Boost → 父块补充 → Diversity去重 → Top5
```

- **子块索引 + 父块返回**: 检索打在段落级（精准），LLM 看到段落+邻居（完整上下文）
- **Section Boost**: 根据问题类型（方法/实验/相关工作）给目标章节的 chunk 加分
- **Marker**: 高精度 PDF 解析（布局检测 + OCR + 章节层级），失败回退 PyPDFLoader + regex

### 🛡️ 防幻觉机制

- **文档仅模式**: LLM 判断检索结果不相关→熔断停止 + 弹窗提示切换混合模式
- **混合模式**: 本地证据不足→LLM 自主调用 Tavily 网络搜索
- **审查自校正**: Reviewer 检查引文格式 + 回答充分性，FAIL 时回 Planner 重检索（≤3 次）
- **3 次上限提示**: 达到最大修订上限时弹窗警告，强制展示最后一版草稿

### 📊 可追溯 (Trace)

每次对话自动保存 `traces/<thread_id>.json`，包含每轮的 query、plan、evidence（含 source/section/score/summary/raw_head）、report、review 结果。同一 thread_id 多轮对话自动累积。

### 🎨 前端

- Vue 3 + Tailwind CSS + 呼吸灯动效
- SSE 流式传输 + 打字机效果
- markdown-it + KaTeX 渲染

---

## 🏗️ 架构图

```text
                          ┌──────────┐
                          │  Router  │  ← 入口节点，写入 intent 到 state
                          └────┬─────┘
                         ┌─────┴─────┐
                         │ 条件路由   │
                         └──┬────┬──┘
                  new_topic │    │ edit_report
                 augment    │    │
                    ┌───────┘    └──────────┐
                    ▼                       ▼
               ┌─────────┐            ┌─────────┐
               │ Planner │            │ Refiner │
               └────┬────┘            └────┬────┘
                    │                      │
                    ▼                      │ edit_report
               ┌──────────┐                │ → END
               │Researcher│                │
               └────┬─────┘                │
                    │                      │
          ┌────┬────┴────┐                 │
          │    │         │                 │
     should_stop│   augment_report         │
        → END   │         │                │
                │         ▼                │
                │    ┌─────────┐           │
                │    │ Refiner │           │
                │    └────┬────┘           │
                │         │                │
                │    ┌────▼────┐           │
                │    │Reviewer │◄──────────┘
                │    └────┬────┘
                ▼         │
           ┌────────┐     │
           │ Writer │     │
           └───┬────┘     │
               │          │
          ┌────▼────┐     │
          │Reviewer │     │
          └────┬────┘     │
               │          │
     ┌────┬────┴────┐     │
     │    │         │     │
   FAIL  PASS     ≥3次    │
   →plan  │       END     │
   (≤3次) │               │
          ▼               ▼
         END             END
```

---

## 🛠️ 技术栈

### Backend
- **API**: Python 3.10+, FastAPI, SSE 流式
- **Agent**: LangChain, LangGraph (checkpoint + conditional edges)
- **检索引擎**: Elasticsearch (BM25 + 向量混合, RRF 融合)
- **重排序**: qwen3-rerank (DashScope OpenAI 兼容接口)
- **Embedding**: DashScope text-embedding-v4 / HuggingFace (可切换)
- **PDF 解析**: Marker (marker-pdf) → PyPDFLoader (降级)
- **持久化**: SQLite (AsyncSqliteSaver, LangGraph checkpoint)
- **LLM**: 双模型 — fast (qwen3-max) + smart (deepseek-r1)
- **网络搜索**: Tavily Search API

### Frontend
- Vue 3 (Composition API)
- Tailwind CSS
- markdown-it + KaTeX

### DevOps
- Docker Compose (3 服务: frontend + backend + ES)
- ES 连接池 (模块级单例复用)

---

## 🚀 快速开始

### Docker Compose (推荐)

```bash
# 1. 配置 API Keys
cp backend/.env.example backend/.env
# 编辑 backend/.env，填入 OPENAI_API_KEY 等

# 2. 启动全部服务
docker compose up -d --build

# 3. 打开浏览器
# http://localhost:5173
```

### 本地开发

```bash
# 后端
cd backend
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# 确保 ES 已启动 (docker compose up -d elasticsearch)
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 前端
cd frontend
npm install
npm run dev
```

---

## 📂 目录结构

```text
ScholarMind/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI 路由 + SSE 流式 + trace 集成
│   │   ├── graph/        # LangGraph 核心
│   │   │   ├── nodes/    # 6 个 LLM 节点 + memory 后处理（非独立 Agent）
│   │   │   ├── state.py  # AgentState 定义
│   │   │   └── graph.py  # 图拓扑 + 条件路由
│   │   ├── rag/          # 检索引擎 (ES 混合检索 + Marker + Chunking)
│   │   ├── tools/        # 外部工具 (Tavily)
│   │   ├── trace/        # 对话追溯 (collector + storage)
│   │   └── utils/        # LLM 工厂
│   ├── traces/           # 对话 trace 输出 (JSON, .gitignore)
│   ├── main.py           # 入口
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/   # Vue 组件
│       ├── services/     # API + SSE 流 + session 管理
│       └── App.vue
├── eval/                 # RAG 评测 (数据集 + 指标)
├── docs/                 # 设计文档 + 截图
├── docker-compose.yml
└── README.md
```
