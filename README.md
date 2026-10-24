# Knowledge Hub

中心化知识库 MCP Server —— 让 AI 客户端同时检索代码图谱和文档，回答「这个功能怎么实现的」类问题。

## 特性

- **双源检索**：同时查询代码知识图谱（graphify）和 Markdown 文档
- **混合检索**：FTS5 关键词匹配 + Embedding 语义检索，RRF 融合排序
- **MCP 协议**：标准 MCP Server，支持 stdio / SSE 传输，可接入 Qoder、Claude Desktop 等 AI 客户端
- **多仓库支持**：独立管理多个代码仓库的图谱，支持跨仓库查询
- **零外部服务依赖**：SQLite + 文件系统存储，Embedding 模型可本地运行

## 架构

```
┌──────────────────────────────────────────────────────┐
│               AI 客户端 (Qoder / Claude / etc.)       │
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
│  │   ┌──────────────┐    ┌──────────────────────┐  │  │
│  │   │ FTS5 关键词   │    │ Embedding 语义检索    │  │  │
│  │   └──────┬───────┘    └──────────┬───────────┘  │  │
│  │          └────────┬──────────────┘               │  │
│  │                   ▼                              │  │
│  │          RRF 融合排序 → top_k                     │  │
│  └────────────────────────────────────────────────┘  │
│       │                              │                │
│       ▼                              ▼                │
│  ┌──────────┐                  ┌──────────┐           │
│  │ graph.json│                  │ .md 文件  │           │
│  │ (按仓库)  │                  │ (按分类)  │           │
│  └──────────┘                  └──────────┘           │
└──────────────────────────────────────────────────────┘
```

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/Weidaicheng/knowledge-hub.git
cd knowledge-hub

# 创建虚拟环境（需要 Python 3.11+）
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e .
```

### 配置

编辑 `config.yaml`：

```yaml
data_dir: ./data
host: 0.0.0.0
port: 8080
transport: sse            # sse | stdio

# Embedding 配置
embedding:
  backend: local          # local | openai
  model: BAAI/bge-small-zh-v1.5
  enabled: true

# 检索参数
retrieval:
  rrf_k: 60
  fts_weight: 0.4
  embed_weight: 0.6
```

### 启动

```bash
# stdio 模式（本地 AI 客户端直连）
python -m knowledge_hub --transport stdio

# SSE 模式（HTTP 服务，支持远程访问）
python -m knowledge_hub --transport sse --port 8080
```

### 接入 AI 客户端

**Qoder / Claude Desktop** 配置示例（`~/.qoder-cn/mcp.json` 或 `claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "knowledge-hub": {
      "url": "http://localhost:8080/mcp",
      "transport": "sse"
    }
  }
}
```

stdio 模式：

```json
{
  "mcpServers": {
    "knowledge-hub": {
      "command": "python",
      "args": ["-m", "knowledge_hub", "--transport", "stdio"],
      "cwd": "/path/to/knowledge-hub"
    }
  }
}
```

## 使用

### 1. 上传代码图谱

使用 [graphify](https://github.com/graphify/graphify) 生成代码图谱后，通过 MCP 工具上传：

```
工具: upload_graph
参数:
  owner: "your-org"
  repo: "your-repo"
  graph_json: "<graph.json 文件内容>"
```

### 2. 上传文档

```
工具: upload_doc
参数:
  title: "App 登录用户故事"
  category: "product/auth"
  content: "# 登录流程\n\n## 用户故事\n\n作为用户..."
```

### 3. 查询

```
工具: query
参数:
  question: "App 的登录流程是什么样的？"
  top_k: 10
```

返回结果包含代码图谱中的相关节点（函数、类、调用链）和文档中的相关片段（需求、设计文档），AI 客户端会整合成连贯回答。

## MCP 工具参考

### 统一查询

| 工具 | 说明 |
|------|------|
| `query` | 同时查询图谱和文档，返回合并结果 |

### 图谱查询

| 工具 | 说明 |
|------|------|
| `query_code_graph` | 仅查询代码图谱 |
| `graph_stats` | 查看图谱统计 |
| `graph_neighbors` | 获取节点邻居 |
| `graph_community` | 列出社区中的节点 |
| `graph_god_nodes` | 返回核心节点（度数最高） |

### 文档检索

| 工具 | 说明 |
|------|------|
| `search_docs` | 仅检索文档 |
| `list_docs` | 列出所有文档 |
| `get_doc` | 获取文档全文 |

### 管理

| 工具 | 说明 |
|------|------|
| `upload_graph` | 上传/更新仓库图谱 |
| `upload_doc` | 上传/更新文档 |
| `remove_graph` | 删除仓库图谱 |
| `remove_doc` | 删除文档 |
| `list_repos` | 列出所有仓库 |

## 项目结构

```
knowledge-hub/
├── pyproject.toml
├── config.yaml
├── knowledge_hub/
│   ├── server.py            # MCP Server 入口
│   ├── models.py            # 数据模型
│   ├── storage/
│   │   └── db.py            # SQLite + FTS5 + 向量存储
│   ├── engines/
│   │   ├── graph_engine.py      # 图谱加载与查询
│   │   ├── embedding_engine.py  # Embedding（本地/OpenAI）
│   │   ├── doc_engine.py        # 文档分块
│   │   └── hybrid_retriever.py  # 混合检索 + RRF
│   └── tools/
│       ├── query.py         # 统一查询工具
│       ├── graph_tools.py   # 图谱工具
│       ├── doc_tools.py     # 文档工具
│       └── admin_tools.py   # 管理工具
└── data/                    # 运行时数据（gitignore）
    ├── knowledge_hub.db
    ├── graphs/
    └── docs/
```

## 技术栈

- **Python 3.11+**
- **MCP SDK** (`mcp<2`) — MCP 协议实现
- **NetworkX** — 图数据结构与算法
- **SQLite FTS5** — 全文检索
- **sentence-transformers** / **OpenAI Embeddings** — 语义检索
- **numpy** — 向量计算

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 类型检查
mypy knowledge_hub/
```

## License

MIT
