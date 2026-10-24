"""数据模型定义"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class Repo:
    """仓库信息"""
    id: Optional[int] = None
    owner: str = ""
    name: str = ""
    graph_path: str = ""
    report_path: Optional[str] = None
    node_count: int = 0
    edge_count: int = 0
    community_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"


@dataclass
class GraphNode:
    """图谱节点"""
    id: Optional[int] = None
    repo_id: Optional[int] = None
    node_id: str = ""           # graphify 原始节点 ID
    name: str = ""              # 节点名称/label
    node_type: Optional[str] = None
    description: Optional[str] = None
    file_path: Optional[str] = None
    community: Optional[int] = None
    community_name: Optional[str] = None


@dataclass
class GraphEdge:
    """图谱边"""
    source: str = ""
    target: str = ""
    relation: str = ""
    confidence: str = ""


@dataclass
class Doc:
    """Markdown 文档"""
    id: Optional[int] = None
    title: str = ""
    file_path: str = ""
    category: Optional[str] = None
    content: str = ""
    chunk_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class DocChunk:
    """文档分块"""
    id: Optional[int] = None
    doc_id: Optional[int] = None
    chunk_index: int = 0
    content: str = ""
    heading: Optional[str] = None


@dataclass
class Embedding:
    """向量嵌入"""
    id: Optional[int] = None
    source_type: str = ""       # 'node' 或 'chunk'
    source_id: int = 0
    repo_id: Optional[int] = None
    doc_id: Optional[int] = None
    model: str = ""
    vector: bytes = b""
    dimensions: int = 0
    created_at: Optional[str] = None


@dataclass
class SearchResult:
    """搜索结果"""
    source_type: str            # 'node' 或 'chunk'
    source_id: int
    score: float
    rrf_score: Optional[float] = None
    # 节点相关字段
    repo: Optional[str] = None
    node: Optional[dict] = None
    related_nodes: list = field(default_factory=list)
    file_path: Optional[str] = None
    community: Optional[str] = None
    # 文档相关字段
    title: Optional[str] = None
    category: Optional[str] = None
    chunk: Optional[str] = None
    heading: Optional[str] = None


@dataclass
class QueryResult:
    """统一查询结果"""
    graph_results: list[dict] = field(default_factory=list)
    doc_results: list[dict] = field(default_factory=list)
    hint: str = "以上结果来自代码图谱和文档两个维度，请综合整理为连贯回答"


@dataclass
class Config:
    """服务配置"""
    data_dir: str = "./data"
    host: str = "0.0.0.0"
    port: int = 8080
    transport: str = "sse"

    # 图谱查询参数
    graph_top_k: int = 10
    graph_neighbor_depth: int = 1

    # 文档检索参数
    docs_top_k: int = 10
    docs_chunk_max_chars: int = 2000

    # Embedding 配置
    embedding_backend: str = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_openai_api_key: Optional[str] = None
    embedding_openai_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 32
    embedding_enabled: bool = True

    # 混合检索参数
    retrieval_rrf_k: int = 60
    retrieval_fts_weight: float = 0.4
    retrieval_embed_weight: float = 0.6

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置"""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        cfg = cls()
        cfg.data_dir = data.get("data_dir", cfg.data_dir)
        cfg.host = data.get("host", cfg.host)
        cfg.port = data.get("port", cfg.port)
        cfg.transport = data.get("transport", cfg.transport)

        graph = data.get("graph", {})
        cfg.graph_top_k = graph.get("default_top_k", cfg.graph_top_k)
        cfg.graph_neighbor_depth = graph.get("neighbor_depth", cfg.graph_neighbor_depth)

        docs = data.get("docs", {})
        cfg.docs_top_k = docs.get("default_top_k", cfg.docs_top_k)
        cfg.docs_chunk_max_chars = docs.get("chunk_max_chars", cfg.docs_chunk_max_chars)

        emb = data.get("embedding", {})
        cfg.embedding_backend = emb.get("backend", cfg.embedding_backend)
        cfg.embedding_model = emb.get("model", cfg.embedding_model)
        cfg.embedding_openai_api_key = emb.get("openai_api_key")
        cfg.embedding_openai_model = emb.get("openai_model", cfg.embedding_openai_model)
        cfg.embedding_batch_size = emb.get("batch_size", cfg.embedding_batch_size)
        cfg.embedding_enabled = emb.get("enabled", cfg.embedding_enabled)

        ret = data.get("retrieval", {})
        cfg.retrieval_rrf_k = ret.get("rrf_k", cfg.retrieval_rrf_k)
        cfg.retrieval_fts_weight = ret.get("fts_weight", cfg.retrieval_fts_weight)
        cfg.retrieval_embed_weight = ret.get("embed_weight", cfg.retrieval_embed_weight)

        return cfg
