"""混合检索引擎 — FTS5 + Embedding + RRF 融合"""

from typing import Optional

import numpy as np

from ..models import Config, QueryResult
from ..storage.db import Database
from .graph_engine import GraphEngine
from .embedding_engine import EmbeddingEngine
from .doc_engine import DocEngine


class HybridRetriever:
    """混合检索引擎

    同时使用 FTS5 关键词检索和 Embedding 语义检索，
    通过 RRF (Reciprocal Rank Fusion) 融合排序结果。
    """

    def __init__(self, db: Database, graph_engine: GraphEngine,
                 embedding_engine: EmbeddingEngine, doc_engine: DocEngine,
                 config: Config):
        self.db = db
        self.graph_engine = graph_engine
        self.embedding_engine = embedding_engine
        self.doc_engine = doc_engine
        self.config = config

    # ==================== 图谱检索 ====================

    def search_graph(self, query: str, top_k: int = 10,
                    repo_ids: Optional[list[int]] = None) -> list[dict]:
        """检索代码图谱

        混合使用 FTS5 和 Embedding 检索节点，然后扩展邻居。

        Returns:
            [{
                "repo": str,
                "node": dict,
                "related_nodes": list,
                "file_path": str,
                "community": str,
                "score": float,
            }, ...]
        """
        # 1. FTS5 检索
        fts_results = self.db.search_nodes_fts(query, limit=top_k * 2)

        # 2. Embedding 检索
        embed_results = []
        if self.config.embedding_enabled and self.embedding_engine.enabled:
            embed_results = self._search_nodes_embedding(query, top_k * 2, repo_ids)

        # 3. RRF 融合
        node_scores = self._rrf_fusion_nodes(fts_results, embed_results)

        # 4. 取 top_k 并扩展邻居
        results = []
        for node_id, score in node_scores[:top_k]:
            # 获取节点详情
            node_data = self._get_node_by_db_id(node_id)
            if not node_data:
                continue

            # 扩展邻居
            neighbors = self.graph_engine.get_neighbors(
                node_data["name"],
                repo_id=node_data.get("repo_id"),
                depth=self.config.graph_neighbor_depth,
                limit=10
            )

            results.append({
                "repo": node_data.get("repo", node_data.get("repo_name", "")),
                "node": {
                    "name": node_data.get("name", ""),
                    "type": node_data.get("node_type", ""),
                    "description": node_data.get("description", ""),
                },
                "related_nodes": neighbors.get("neighbors", []),
                "file_path": node_data.get("file_path", ""),
                "community": node_data.get("community_name", ""),
                "score": score,
            })

        return results

    def _search_nodes_embedding(self, query: str, top_k: int,
                               repo_ids: Optional[list[int]] = None) -> list[dict]:
        """Embedding 检索节点"""
        if not self.embedding_engine.enabled:
            return []

        # 获取所有向量
        node_ids, _, matrix = self.db.get_all_vectors(self.embedding_engine.model_name)

        if matrix is None or len(node_ids) == 0:
            return []

        # 过滤 repo
        if repo_ids:
            filtered_ids = []
            filtered_vecs = []
            for i, nid in enumerate(node_ids):
                if nid == -1:
                    continue
                # 查询节点所属 repo
                row = self.db.conn.execute(
                    "SELECT repo_id FROM graph_nodes WHERE id = ?", (nid,)).fetchone()
                if row and row["repo_id"] in repo_ids:
                    filtered_ids.append(nid)
                    filtered_vecs.append(matrix[i])
            if filtered_vecs:
                matrix = np.stack(filtered_vecs)
                node_ids = filtered_ids
            else:
                return []
        else:
            # 过滤掉 chunk 的 ID（node_ids 中 -1 表示 chunk）
            valid_mask = [nid != -1 for nid in node_ids]
            node_ids = [nid for nid, v in zip(node_ids, valid_mask) if v]
            matrix = matrix[valid_mask]

        # 向量搜索
        search_results = self.embedding_engine.search(
            query, matrix, node_ids, top_k)

        return search_results

    def _get_node_by_db_id(self, node_db_id: int) -> Optional[dict]:
        """根据数据库 ID 获取节点"""
        row = self.db.conn.execute("""
            SELECT gn.*, r.owner, r.name as repo_name
            FROM graph_nodes gn
            JOIN repos r ON r.id = gn.repo_id
            WHERE gn.id = ?
        """, (node_db_id,)).fetchone()
        if row:
            d = dict(row)
            d["repo"] = f"{d['owner']}/{d['repo_name']}"
            return d
        return None

    # ==================== 文档检索 ====================

    def search_docs(self, query: str, top_k: int = 10,
                   category: Optional[str] = None) -> list[dict]:
        """检索 Markdown 文档

        混合使用 FTS5 和 Embedding 检索分块。

        Returns:
            [{
                "title": str,
                "category": str,
                "chunk": str,
                "heading": str,
                "score": float,
            }, ...]
        """
        # 1. FTS5 检索
        fts_results = self.db.search_chunks_fts(query, limit=top_k * 2)

        # 2. Embedding 检索
        embed_results = []
        if self.config.embedding_enabled and self.embedding_engine.enabled:
            embed_results = self._search_chunks_embedding(query, top_k * 2)

        # 3. RRF 融合
        chunk_scores = self._rrf_fusion_chunks(fts_results, embed_results)

        # 4. 取 top_k 并扩展上下文
        results = []
        for chunk_id, score in chunk_scores[:top_k]:
            chunk_data = self._get_chunk_by_db_id(chunk_id)
            if not chunk_data:
                continue

            # 过滤 category
            if category and chunk_data.get("category") != category:
                continue

            results.append({
                "title": chunk_data.get("title", ""),
                "category": chunk_data.get("category", ""),
                "chunk": chunk_data.get("content", ""),
                "heading": chunk_data.get("heading", ""),
                "score": score,
            })

        return results

    def _search_chunks_embedding(self, query: str, top_k: int) -> list[dict]:
        """Embedding 检索文档分块"""
        if not self.embedding_engine.enabled:
            return []

        # 获取所有向量
        _, chunk_ids, matrix = self.db.get_all_vectors(self.embedding_engine.model_name)

        if matrix is None or len(chunk_ids) == 0:
            return []

        # 过滤掉 node 的 ID（chunk_ids 中 -1 表示 node）
        valid_mask = [cid != -1 for cid in chunk_ids]
        chunk_ids = [cid for cid, v in zip(chunk_ids, valid_mask) if v]
        matrix = matrix[valid_mask]

        if len(chunk_ids) == 0:
            return []

        # 向量搜索
        search_results = self.embedding_engine.search(
            query, matrix, chunk_ids, top_k)

        return search_results

    def _get_chunk_by_db_id(self, chunk_db_id: int) -> Optional[dict]:
        """根据数据库 ID 获取分块"""
        row = self.db.conn.execute("""
            SELECT dc.*, d.title, d.category
            FROM doc_chunks dc
            JOIN docs d ON d.id = dc.doc_id
            WHERE dc.id = ?
        """, (chunk_db_id,)).fetchone()
        if row:
            return dict(row)
        return None

    # ==================== RRF 融合 ====================

    def _rrf_fusion_nodes(self, fts_results: list[dict],
                         embed_results: list[dict]) -> list[tuple[int, float]]:
        """RRF 融合节点检索结果

        Returns:
            [(node_db_id, rrf_score), ...] 按分数降序
        """
        k = self.config.retrieval_rrf_k
        scores: dict[int, float] = {}

        # FTS5 结果
        for rank, r in enumerate(fts_results):
            node_id = r.get("id")
            if node_id:
                scores[node_id] = scores.get(node_id, 0.0) + \
                    self.config.retrieval_fts_weight / (k + rank + 1)

        # Embedding 结果
        for rank, r in enumerate(embed_results):
            node_id = r.get("source_id")
            if node_id:
                scores[node_id] = scores.get(node_id, 0.0) + \
                    self.config.retrieval_embed_weight / (k + rank + 1)

        # 排序
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_items

    def _rrf_fusion_chunks(self, fts_results: list[dict],
                          embed_results: list[dict]) -> list[tuple[int, float]]:
        """RRF 融合文档分块检索结果

        Returns:
            [(chunk_db_id, rrf_score), ...] 按分数降序
        """
        k = self.config.retrieval_rrf_k
        scores: dict[int, float] = {}

        # FTS5 结果
        for rank, r in enumerate(fts_results):
            chunk_id = r.get("id")
            if chunk_id:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + \
                    self.config.retrieval_fts_weight / (k + rank + 1)

        # Embedding 结果
        for rank, r in enumerate(embed_results):
            chunk_id = r.get("source_id")
            if chunk_id:
                scores[chunk_id] = scores.get(chunk_id, 0.0) + \
                    self.config.retrieval_embed_weight / (k + rank + 1)

        # 排序
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_items

    # ==================== 统一查询 ====================

    def query(self, question: str, top_k: int = 10) -> QueryResult:
        """统一查询 — 同时检索图谱和文档

        Returns:
            QueryResult 包含 graph_results 和 doc_results
        """
        graph_results = self.search_graph(question, top_k=top_k)
        doc_results = self.search_docs(question, top_k=top_k)

        return QueryResult(
            graph_results=graph_results,
            doc_results=doc_results,
            hint="以上结果来自代码图谱和文档两个维度，请综合整理为连贯回答"
        )
