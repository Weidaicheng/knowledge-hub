# 中心化知识库 MCP Server — MVP0 设计方案

## 1. 定位

一个独立运行的 MCP Server，为公司内部提供统一的知识检索能力。每次查询同时覆盖**代码知识图谱**和 **Markdown 文档**，返回结构化原始结果，由 AI 客户端的 LLM 负责整合成连贯回答。

## 2. 整体架构

```
┌──────────────────────────────────────────────────────┐
│               AI 客户端 (Qoder / etc.)                │
│                                                       │
│  用户: "App 的登录流程是什么样的？"                    │
│  ↓ 调用 MCP tool                                      │
└───────────────┬──────────────────────────────────────┘
                │  MCP Protocol (stdio / SSE)
                ▼
┌──────────────────────────────────────────────────────┐
│            Knowledge Hub MCP Server                   │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐ │
│  │ 图谱查询  │  │ 文档检索  │  │    统一查询         │ │
│  │  Engine   │  │  Engine   │  │   (合并两者)        │ │
│  └────┬─────┘  └────┬─────┘  └────────┬───────────┘ │
│       │              │                  │              │
│       ▼              ▼                  ▼              │
│  ┌────────────────────────────────────────────────┐  │
│  │            混合检索层 (Hybrid Retriever)         │  │
│  │                                                  │  │
│  │   ┌──────────────┐    ┌──────────────────────┐  │  │
│  │   │ FTS5 关键词   │    │ Embedding 语义检索    │  │  │
│  │   │ 精确匹配      │    │ 向量相似度 (cosine)   │  │  │
│  │   └──────┬───────┘    └──────────┬───────────┘  │  │
│  │          └────────┬──────────────┘               │  │
│  │                   ▼                              │  │
│  │          RRF 融合排序 → top_k                     │  │
│  └────────────────────────────────────────────────┘  │
│       │                              ▲                │
│       ▼                              │                │
│  ┌──────────────────────────────────────────────┐    │
│  │              SQLite 数据库                     │    │
│  │  repos 表 │ nodes 表 │ docs 表 │ FTS5 虚拟表  │    │
│  │  embeddings 表（向量存储）                      │    │
│  └──────────────────────────────────────────────┘    │
│       │                              ▲                │
│       ▼                              │                │
│  ┌──────────┐                  ┌──────────┐           │
│  │ graph.json│                  │ .md 文件  │           │
│  │ (按仓库)  │                  │ (按分类)  │           │
│  └──────────┘                  └──────────┘           │
└──────────────────────────────────────────────────────┘
```

## 3. 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 语言 | Python 3.11+ | 复用 Knowledge_Base 的 GraphEngine 逻辑；MCP SDK 官方支持 |
| MCP SDK | `mcp` (Python) | 标准 MCP 协议实现 |
| 图查询 | NetworkX | 复用现有 GraphEngine，支持节点遍历、邻居、社区、最短路径 |
| 关键词检索 | SQLite FTS5 | 零依赖、精确术语匹配，适合已知名称的查找 |
| 语义检索 | sentence-transformers / OpenAI Embeddings | 支持同义词、近义表达，解决"字面不同但语义相同"的问题 |
| 向量存储 | SQLite + numpy | embeddings 表存向量二进制；numpy 做 cosine 相似度计算；小规模够用，零额外服务依赖 |
| 混合排序 | RRF (Reciprocal Rank Fusion) | 融合 FTS5 排名和向量相似度排名，业界标准做法 |
| 存储 | 文件系统 + SQLite | graph.json 按仓库存放；md 原文存放于文件系统；索引统一进 SQLite |

## 4. 存储结构

服务运行时需要一个数据根目录（通过配置或启动参数指定，如 `./data/`）：

```
data/
├── knowledge_hub.db              # SQLite 主数据库（含 FTS5 索引 + 向量存储）
├── graphs/
│   ├── {owner1}/
│   │   ├── {repo1}/
│   │   │   ├── graph.json        # graphify 输出的原始图谱
│   │   │   └── report.md         # (可选) GRAPH_REPORT.md
│   │   └── {repo2}/
│   │       └── graph.json
│   └── ...
└── docs/
    ├── {category1}/              # 文档分类目录（可多级）
    │   ├── app-login-spec.md
    │   └── auth-flow-research.md
    └── {category2}/
        └── api-design.md
```

### SQLite 表结构

```sql
-- 仓库注册表
CREATE TABLE repos (
    id          INTEGER PRIMARY KEY,
    owner       TEXT NOT NULL,
    name        TEXT NOT NULL,
    graph_path  TEXT NOT NULL,         -- graph.json 的相对路径
    report_path TEXT,                  -- GRAPH_REPORT.md 的相对路径（可选）
    node_count  INTEGER DEFAULT 0,
    edge_count  INTEGER DEFAULT 0,
    community_count INTEGER DEFAULT 0,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(owner, name)
);

-- 图谱节点摘要表（从 graph.json 提取，用于快速检索）
CREATE TABLE graph_nodes (
    id          INTEGER PRIMARY KEY,
    repo_id     INTEGER REFERENCES repos(id),
    node_id     TEXT NOT NULL,          -- graphify 原始节点 ID
    name        TEXT NOT NULL,          -- 节点名称
    node_type   TEXT,                   -- 类型：function / class / file / module 等
    description TEXT,                   -- 节点描述
    file_path   TEXT,                   -- 所属文件路径
    community   INTEGER,               -- 所属社区编号
    community_name TEXT                  -- 社区名称（如有）
);

-- 节点索引（加速按名称/描述检索）
CREATE INDEX idx_nodes_name ON graph_nodes(name);
CREATE INDEX idx_nodes_repo ON graph_nodes(repo_id);
CREATE INDEX idx_nodes_type ON graph_nodes(node_type);

-- 文档表
CREATE TABLE docs (
    id          INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    file_path   TEXT NOT NULL UNIQUE,    -- 相对于 docs/ 的路径
    category    TEXT,                    -- 所属分类
    content     TEXT NOT NULL,           -- markdown 原文
    chunk_count INTEGER DEFAULT 0,
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now'))
);

-- 文档分块表
CREATE TABLE doc_chunks (
    id          INTEGER PRIMARY KEY,
    doc_id      INTEGER REFERENCES docs(id),
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,           -- 分块内容
    heading     TEXT                     -- 所属章节标题（如有）
);

-- FTS5 全文索引（覆盖节点和文档分块）
CREATE VIRTUAL TABLE fts_search USING fts5(
    source_type,     -- 'node' 或 'chunk'
    source_id,       -- graph_nodes.id 或 doc_chunks.id
    repo_owner,      -- 仓库 owner（节点有，文档填 NULL）
    repo_name,       -- 仓库 name（节点有，文档填 NULL）
    doc_title,       -- 文档标题（文档有，节点填 NULL）
    content,         -- 检索文本
    tokenize = 'unicode61'
);

-- 向量 embeddings 表（语义检索用）
CREATE TABLE embeddings (
    id          INTEGER PRIMARY KEY,
    source_type TEXT NOT NULL,          -- 'node' 或 'chunk'
    source_id   INTEGER NOT NULL,       -- graph_nodes.id 或 doc_chunks.id
    repo_id     INTEGER REFERENCES repos(id),  -- 节点有，文档填 NULL
    doc_id      INTEGER REFERENCES docs(id),   -- 文档有，节点填 NULL
    model       TEXT NOT NULL,          -- embedding 模型名称（如 'bge-small-zh-v1.5'）
    vector      BLOB NOT NULL,          -- 向量二进制（numpy float32 → bytes）
    dimensions  INTEGER NOT NULL,       -- 向量维度（如 512）
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(source_type, source_id, model)
);

CREATE INDEX idx_embeddings_source ON embeddings(source_type, source_id);
CREATE INDEX idx_embeddings_model ON embeddings(model);
```

### 向量存储说明

向量以 `numpy.float32` 数组序列化后存入 SQLite `BLOB` 字段。查询时加载到内存，用 numpy 做批量 cosine 相似度计算。小规模场景（<10 万条 embedding）下性能足够，无需引入额外的向量数据库服务。

## 5. MCP Tools 设计

### 5.1 统一查询

**`query`** — 同时查询代码图谱和文档，返回合并结果

```
输入:
  question: string       # 自然语言问题
  top_k: int = 10        # 每类返回的最大结果数

输出:
{
  "graph_results": [
    {
      "repo": "owner/repo",
      "node": { "name": "...", "type": "...", "description": "..." },
      "related_nodes": [...],      # 直接关联的邻居节点
      "file_path": "...",
      "community": "..."
    }
  ],
  "doc_results": [
    {
      "title": "App 登录用户故事",
      "category": "product",
      "chunk": "匹配到的文档片段内容...",
      "heading": "## 登录流程"
    }
  ],
  "hint": "以上结果来自代码图谱和文档两个维度，请综合整理为连贯回答"
}
```

### 5.2 图谱专用查询

**`query_code_graph`** — 仅查询代码知识图谱

```
输入:
  question: string       # 自然语言问题
  repos: [string] = []   # 限定仓库范围（空 = 全部）
  top_k: int = 10

输出: 同 query 中的 graph_results 部分
```

**`graph_stats`** — 查看图谱统计信息

```
输入:
  repo: string = ""      # 指定仓库（空 = 全部汇总）

输出:
{
  "repos": [
    { "name": "owner/repo", "nodes": 342, "edges": 891, "communities": 15 }
  ],
  "total_nodes": 1024,
  "total_edges": 2567
}
```

**`graph_neighbors`** — 获取某节点的邻居（展开图结构）

```
输入:
  node_name: string      # 节点名称
  repo: string = ""      # 仓库（空 = 在所有仓库中查找）
  depth: int = 1         # 展开深度

输出: 节点及其邻居的完整子图
```

### 5.3 文档专用检索

**`search_docs`** — 仅检索 Markdown 文档

```
输入:
  question: string       # 自然语言问题
  category: string = ""  # 限定文档分类
  top_k: int = 10

输出: 同 query 中的 doc_results 部分
```

**`list_docs`** — 列出所有文档及分类

```
输入:
  category: string = ""  # 限定分类

输出: 文档列表（标题、分类、路径、更新时间）
```

### 5.4 管理工具

**`upload_graph`** — 上传/更新仓库图谱

```
输入:
  owner: string
  repo: string
  graph_json: string     # graph.json 的内容

输出: 上传结果（节点数、边数、社区数）
```

**`upload_doc`** — 上传/更新 Markdown 文档

```
输入:
  title: string
  category: string       # 分类路径（如 "product/auth"）
  content: string        # markdown 内容

输出: 上传结果
```

**`remove_graph`** / **`remove_doc`** — 删除图谱或文档

## 6. 查询策略

### 6.1 混合检索流程（图谱 + 文档通用）

每次查询同时走两条检索路径，结果通过 RRF 融合排序后返回：

```
用户问题: "App 的登录流程是什么样的？"
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
┌──────────────────┐  ┌──────────────────────┐
│ 路径 A: FTS5     │  │ 路径 B: Embedding    │
│ 关键词检索        │  │ 语义检索              │
│                  │  │                      │
│ 1. 分词          │  │ 1. 问题 → 向量        │
│  "App" "登录"    │  │  (调 embedding 模型)  │
│  "流程"          │  │                      │
│                  │  │ 2. 加载所有候选向量     │
│ 2. FTS5 全文匹配 │  │  (从 embeddings 表)   │
│  返回匹配行 +    │  │                      │
│  BM25 排名分     │  │ 3. 计算 cosine 相似度  │
│                  │  │  返回 top 候选 +      │
│                  │  │  相似度分数            │
└────────┬─────────┘  └──────────┬───────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ RRF 融合排序           │
         │                       │
         │ score_rrf =           │
         │   1/(k+rank_fts) +    │
         │   1/(k+rank_embed)    │
         │                       │
         │ 去重后按 score_rrf     │
         │ 降序取 top_k          │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │ 上下文扩展             │
         │ (图遍历 / chunk 扩展)  │
         └───────────────────────┘
```

**RRF (Reciprocal Rank Fusion)** 公式：`score = 1/(k + rank_A) + 1/(k + rank_B)`，其中 k 通常取 60。如果某条结果只在一条路径中出现，另一条路径的 rank 视为 ∞（即该项贡献为 0）。

### 6.2 图谱查询流程

```
用户问题: "App 的登录流程是什么样的？"
                    │
                    ▼
        ┌───────────────────────────┐
        │  混合检索（见 6.1）         │
        │                           │
        │  候选池: graph_nodes 表    │
        │  检索文本: name +          │
        │           description +    │
        │           community_name   │
        └───────────┬───────────────┘
                    ▼
        ┌───────────────────────────┐
        │  图遍历扩展                │
        │  对 top_k 匹配到的节点:    │
        │  - 取 N 跳邻居            │
        │  - 取所属社区信息          │
        │  - 取关联边               │
        └───────────┬───────────────┘
                    ▼
        ┌───────────────────────────┐
        │  返回子图上下文            │
        │  含节点详情 + 邻居 + 社区   │
        └───────────────────────────┘
```

### 6.3 文档检索流程

```
用户问题: "App 的登录流程是什么样的？"
                    │
                    ▼
        ┌───────────────────────────┐
        │  混合检索（见 6.1）         │
        │                           │
        │  候选池: doc_chunks 表     │
        │  检索文本: chunk content + │
        │           heading +        │
        │           doc title        │
        └───────────┬───────────────┘
                    ▼
        ┌───────────────────────────┐
        │  上下文扩展                │
        │  匹配到的 chunk:           │
        │  - 取前后相邻 chunk        │
        │  - 取所属文档标题          │
        │  - 取章节 heading          │
        └───────────┬───────────────┘
                    ▼
        ┌───────────────────────────┐
        │  排序返回 top_k            │
        └───────────────────────────┘
```

### 6.4 Embedding 模型配置

支持两种 embedding 后端，通过配置文件切换：

| 模式 | 模型 | 优点 | 缺点 |
|------|------|------|------|
| 本地模型（默认） | `BAAI/bge-small-zh-v1.5` | 零 API 费用、无网络依赖、中文优化 | 首次加载需下载模型（~100MB） |
| API 模型 | OpenAI `text-embedding-3-small` | 无需本地 GPU、质量更高 | 需 API key、有调用成本、需网络 |

服务启动时加载 embedding 模型（本地模式下载到 `~/.cache/huggingface/`），所有文本统一用同一模型生成向量。

### 6.5 向量索引构建时机

embedding 向量在**上传时**生成并写入 `embeddings` 表：
- `upload_graph`：解析 graph.json → 提取节点 → 对每个节点的 `name + description` 生成 embedding → 入库
- `upload_doc`：解析 markdown → 分块 → 对每个 chunk 的 `content + heading` 生成 embedding → 入库

查询时只需加载向量做相似度计算，无需实时调 embedding 模型。

### 6.3 文档分块策略

Markdown 文档按 **二级标题（`##`）** 自然分块。每个 chunk 保留其所属的文档标题和章节 heading，便于 AI 客户端理解上下文。若某节过长（>2000 字），按段落进一步切分。

### 6.4 统一查询合并逻辑

`query` 工具内部并行执行图谱查询和文档检索，将两组结果合并到一个响应中返回。不做去重或融合——交给 AI 客户端的 LLM 来组织最终回答。

## 7. AI 客户端接入示例

员工在自己的 AI 客户端（如 Qoder）中配置 MCP 连接：

```json
{
  "mcpServers": {
    "company-kb": {
      "url": "http://knowledge-hub.internal:8080/mcp",
      "transport": "sse"
    }
  }
}
```

配置后，AI 客户端自动获得 `query`、`query_code_graph`、`search_docs` 等工具。当用户提问时，AI 客户端的 LLM 会：

1. 调用 `query("App 的登录流程是什么样的？")`
2. 拿到图谱结果（登录相关的函数、类、调用链）和文档结果（用户故事、需求文档片段）
3. 将两部分信息整合成一段连贯回答，涵盖代码实现和设计考量

## 8. 启动与配置

```bash
# 启动命令
python -m knowledge_hub.server \
    --data-dir ./data \
    --host 0.0.0.0 \
    --port 8080 \
    --transport sse
```

配置文件（可选，`config.yaml`）：

```yaml
data_dir: ./data
host: 0.0.0.0
port: 8080
transport: sse            # sse | stdio

# 图谱查询参数
graph:
  default_top_k: 10
  neighbor_depth: 1       # 匹配节点向外扩展的跳数

# 文档检索参数
docs:
  default_top_k: 10
  chunk_max_chars: 2000   # 单 chunk 最大字符数

# Embedding 语义检索配置
embedding:
  backend: local          # local | openai
  model: BAAI/bge-small-zh-v1.5   # 本地模型名（backend=local 时生效）
  # openai_api_key: sk-xxx  # backend=openai 时必填（建议用环境变量 OPENAI_API_KEY）
  # openai_model: text-embedding-3-small  # backend=openai 时可选
  batch_size: 32          # 批量生成 embedding 的批次大小
  enabled: true           # 设为 false 可退化为纯 FTS5 模式

# 混合检索参数
retrieval:
  rrf_k: 60               # RRF 融合公式中的 k 值
  fts_weight: 0.4         # FTS5 结果权重（0-1）
  embed_weight: 0.6       # Embedding 结果权重（0-1）
```

## 9. 项目结构

```
knowledge-hub/
├── pyproject.toml
├── config.yaml
├── knowledge_hub/
│   ├── __init__.py
│   ├── server.py            # MCP Server 入口，注册所有 tools
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── query.py         # 统一 query 工具
│   │   ├── graph_tools.py   # query_code_graph / graph_stats / graph_neighbors
│   │   ├── doc_tools.py     # search_docs / list_docs
│   │   └── admin_tools.py   # upload_graph / upload_doc / remove_*
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── graph_engine.py  # 图谱加载与查询（复用/改写自 Knowledge_Base）
│   │   ├── doc_engine.py    # 文档分块、索引、检索
│   │   ├── embedding_engine.py  # Embedding 模型加载、向量生成、相似度计算
│   │   └── hybrid_retriever.py  # 混合检索：FTS5 + Embedding + RRF 融合
│   ├── storage/
│   │   ├── __init__.py
│   │   └── db.py            # SQLite 连接管理、建表、FTS5 维护、向量存取
│   └── models.py            # 数据结构定义
└── data/                    # 运行时数据目录（示例）
    ├── knowledge_hub.db
    ├── graphs/
    └── docs/
```

## 10. MVP0 边界

**包含：**
- MCP Server 核心（stdio / SSE 两种传输）
- 图谱上传 + 节点 embedding 生成 + 混合检索（FTS5 + 向量） + 邻居扩展
- 文档上传 + 分块 + chunk embedding 生成 + 混合检索
- 统一 query 工具（RRF 融合排序）+ 独立查询工具
- 管理工具（上传/删除）
- Embedding 引擎（支持本地模型 / OpenAI API 两种后端，可配置切换）

**不包含（后续迭代）：**
- 自动从 GitHub 拉取代码并生成 graphify 图谱
- 自动从文档仓库同步 markdown
- 权限控制 / 多租户
- 全局索引 / 跨仓库智能路由
- Web 管理界面
- 向量数据库升级（如 Milvus / Qdrant，当前用 SQLite + numpy 足够）
