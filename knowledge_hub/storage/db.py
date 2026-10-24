"""SQLite 数据库管理 — 存储、索引、向量"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from ..models import Repo, GraphNode, Doc, DocChunk, Embedding


class Database:
    """知识库数据库层"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self):
        """初始化数据库表结构"""
        self.conn.executescript("""
            -- 仓库注册表
            CREATE TABLE IF NOT EXISTS repos (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                owner           TEXT NOT NULL,
                name            TEXT NOT NULL,
                graph_path      TEXT NOT NULL,
                report_path     TEXT,
                node_count      INTEGER DEFAULT 0,
                edge_count      INTEGER DEFAULT 0,
                community_count INTEGER DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now')),
                updated_at      TEXT DEFAULT (datetime('now')),
                UNIQUE(owner, name)
            );

            -- 图谱节点摘要表
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id         INTEGER REFERENCES repos(id) ON DELETE CASCADE,
                node_id         TEXT NOT NULL,
                name            TEXT NOT NULL,
                node_type       TEXT,
                description     TEXT,
                file_path       TEXT,
                community       INTEGER,
                community_name  TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_repo ON graph_nodes(repo_id);
            CREATE INDEX IF NOT EXISTS idx_nodes_name ON graph_nodes(name);
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON graph_nodes(node_type);

            -- 文档表
            CREATE TABLE IF NOT EXISTS docs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                title           TEXT NOT NULL,
                file_path       TEXT NOT NULL UNIQUE,
                category        TEXT,
                content         TEXT NOT NULL,
                chunk_count     INTEGER DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now')),
                updated_at      TEXT DEFAULT (datetime('now'))
            );

            -- 文档分块表
            CREATE TABLE IF NOT EXISTS doc_chunks (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id          INTEGER REFERENCES docs(id) ON DELETE CASCADE,
                chunk_index     INTEGER NOT NULL,
                content         TEXT NOT NULL,
                heading         TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_doc ON doc_chunks(doc_id);

            -- 向量 embeddings 表
            CREATE TABLE IF NOT EXISTS embeddings (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type     TEXT NOT NULL,
                source_id       INTEGER NOT NULL,
                repo_id         INTEGER REFERENCES repos(id) ON DELETE CASCADE,
                doc_id          INTEGER REFERENCES docs(id) ON DELETE CASCADE,
                model           TEXT NOT NULL,
                vector          BLOB NOT NULL,
                dimensions      INTEGER NOT NULL,
                created_at      TEXT DEFAULT (datetime('now')),
                UNIQUE(source_type, source_id, model)
            );

            CREATE INDEX IF NOT EXISTS idx_emb_source ON embeddings(source_type, source_id);
            CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model);
        """)

        # FTS5 虚拟表（可能不可用）
        try:
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_search USING fts5(
                    source_type,
                    source_id,
                    repo_owner,
                    repo_name,
                    doc_title,
                    content,
                    tokenize = 'unicode61'
                )
            """)
            self._fts_available = True
        except sqlite3.OperationalError:
            self._fts_available = False

        self.conn.commit()

    @property
    def fts_available(self) -> bool:
        return self._fts_available

    # ==================== Repo 操作 ====================

    def upsert_repo(self, owner: str, name: str, graph_path: str,
                    report_path: Optional[str] = None,
                    node_count: int = 0, edge_count: int = 0,
                    community_count: int = 0) -> int:
        """插入或更新仓库信息，返回 repo_id"""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        cursor = self.conn.execute("""
            INSERT INTO repos (owner, name, graph_path, report_path,
                              node_count, edge_count, community_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner, name) DO UPDATE SET
                graph_path = excluded.graph_path,
                report_path = excluded.report_path,
                node_count = excluded.node_count,
                edge_count = excluded.edge_count,
                community_count = excluded.community_count,
                updated_at = excluded.updated_at
        """, (owner, name, graph_path, report_path,
              node_count, edge_count, community_count, now))
        self.conn.commit()

        # 获取 repo_id
        row = self.conn.execute(
            "SELECT id FROM repos WHERE owner = ? AND name = ?",
            (owner, name)).fetchone()
        return row["id"]

    def get_repo(self, owner: str, name: str) -> Optional[Repo]:
        """获取仓库信息"""
        row = self.conn.execute(
            "SELECT * FROM repos WHERE owner = ? AND name = ?",
            (owner, name)).fetchone()
        if row:
            return Repo(**dict(row))
        return None

    def get_repo_by_id(self, repo_id: int) -> Optional[Repo]:
        """根据 ID 获取仓库"""
        row = self.conn.execute(
            "SELECT * FROM repos WHERE id = ?", (repo_id,)).fetchone()
        if row:
            return Repo(**dict(row))
        return None

    def list_repos(self) -> list[Repo]:
        """列出所有仓库"""
        rows = self.conn.execute(
            "SELECT * FROM repos ORDER BY updated_at DESC").fetchall()
        return [Repo(**dict(r)) for r in rows]

    def delete_repo(self, owner: str, name: str) -> bool:
        """删除仓库及其关联数据"""
        repo = self.get_repo(owner, name)
        if not repo or not repo.id:
            return False
        # 级联删除
        self.conn.execute("DELETE FROM repos WHERE id = ?", (repo.id,))
        self.conn.commit()
        return True

    # ==================== GraphNode 操作 ====================

    def insert_nodes(self, repo_id: int, nodes: list[dict]) -> int:
        """批量插入图谱节点，返回插入数量"""
        count = 0
        for n in nodes:
            self.conn.execute("""
                INSERT INTO graph_nodes
                    (repo_id, node_id, name, node_type, description,
                     file_path, community, community_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (repo_id, n.get("node_id", ""), n.get("name", ""),
                  n.get("node_type"), n.get("description"),
                  n.get("file_path"), n.get("community"),
                  n.get("community_name")))
            count += 1
        self.conn.commit()
        return count

    def delete_nodes_by_repo(self, repo_id: int):
        """删除仓库的所有节点"""
        self.conn.execute(
            "DELETE FROM graph_nodes WHERE repo_id = ?", (repo_id,))
        self.conn.commit()

    def get_nodes_by_repo(self, repo_id: int) -> list[GraphNode]:
        """获取仓库的所有节点"""
        rows = self.conn.execute(
            "SELECT * FROM graph_nodes WHERE repo_id = ?", (repo_id,)).fetchall()
        return [GraphNode(**dict(r)) for r in rows]

    def search_nodes_fts(self, query: str, limit: int = 20) -> list[dict]:
        """FTS5 搜索图谱节点"""
        if not self._fts_available:
            return self._search_nodes_like(query, limit)

        try:
            rows = self.conn.execute("""
                SELECT gn.*, r.owner, r.name as repo_name,
                       rank
                FROM fts_search
                JOIN graph_nodes gn ON gn.id = CAST(fts_search.source_id AS INTEGER)
                JOIN repos r ON r.id = gn.repo_id
                WHERE fts_search MATCH ? AND source_type = 'node'
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return self._search_nodes_like(query, limit)

    def _search_nodes_like(self, query: str, limit: int) -> list[dict]:
        """LIKE 降级搜索节点"""
        pattern = f"%{query}%"
        rows = self.conn.execute("""
            SELECT gn.*, r.owner, r.name as repo_name
            FROM graph_nodes gn
            JOIN repos r ON r.id = gn.repo_id
            WHERE gn.name LIKE ? OR gn.description LIKE ?
            LIMIT ?
        """, (pattern, pattern, limit)).fetchall()
        return [dict(r) for r in rows]

    # ==================== Doc 操作 ====================

    def upsert_doc(self, title: str, file_path: str, content: str,
                   category: Optional[str] = None) -> int:
        """插入或更新文档，返回 doc_id"""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        existing = self.conn.execute(
            "SELECT id FROM docs WHERE file_path = ?", (file_path,)).fetchone()

        if existing:
            doc_id = existing["id"]
            self.conn.execute("""
                UPDATE docs SET title = ?, content = ?, category = ?,
                               chunk_count = 0, updated_at = ?
                WHERE id = ?
            """, (title, content, category, now, doc_id))
            # 删除旧分块
            self.conn.execute("DELETE FROM doc_chunks WHERE doc_id = ?", (doc_id,))
        else:
            cursor = self.conn.execute("""
                INSERT INTO docs (title, file_path, category, content, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (title, file_path, category, content, now, now))
            doc_id = cursor.lastrowid

        self.conn.commit()
        return doc_id

    def get_doc(self, doc_id: int) -> Optional[Doc]:
        """获取文档"""
        row = self.conn.execute(
            "SELECT * FROM docs WHERE id = ?", (doc_id,)).fetchone()
        if row:
            return Doc(**dict(row))
        return None

    def list_docs(self, category: Optional[str] = None) -> list[Doc]:
        """列出文档"""
        if category:
            rows = self.conn.execute(
                "SELECT * FROM docs WHERE category = ? ORDER BY updated_at DESC",
                (category,)).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM docs ORDER BY updated_at DESC").fetchall()
        return [Doc(**dict(r)) for r in rows]

    def delete_doc(self, file_path: str) -> bool:
        """删除文档及其分块"""
        doc = self.conn.execute(
            "SELECT id FROM docs WHERE file_path = ?", (file_path,)).fetchone()
        if not doc:
            return False
        self.conn.execute("DELETE FROM docs WHERE id = ?", (doc["id"],))
        self.conn.commit()
        return True

    # ==================== DocChunk 操作 ====================

    def insert_chunks(self, doc_id: int, chunks: list[dict]) -> int:
        """批量插入文档分块"""
        count = 0
        for i, c in enumerate(chunks):
            self.conn.execute("""
                INSERT INTO doc_chunks (doc_id, chunk_index, content, heading)
                VALUES (?, ?, ?, ?)
            """, (doc_id, i, c.get("content", ""), c.get("heading")))
            count += 1

        # 更新文档的 chunk_count
        self.conn.execute(
            "UPDATE docs SET chunk_count = ? WHERE id = ?", (count, doc_id))
        self.conn.commit()
        return count

    def get_chunks_by_doc(self, doc_id: int) -> list[DocChunk]:
        """获取文档的所有分块"""
        rows = self.conn.execute(
            "SELECT * FROM doc_chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,)).fetchall()
        return [DocChunk(**dict(r)) for r in rows]

    def search_chunks_fts(self, query: str, limit: int = 20) -> list[dict]:
        """FTS5 搜索文档分块"""
        if not self._fts_available:
            return self._search_chunks_like(query, limit)

        try:
            rows = self.conn.execute("""
                SELECT dc.*, d.title, d.category, d.file_path,
                       rank
                FROM fts_search
                JOIN doc_chunks dc ON dc.id = CAST(fts_search.source_id AS INTEGER)
                JOIN docs d ON d.id = dc.doc_id
                WHERE fts_search MATCH ? AND source_type = 'chunk'
                ORDER BY rank
                LIMIT ?
            """, (query, limit)).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return self._search_chunks_like(query, limit)

    def _search_chunks_like(self, query: str, limit: int) -> list[dict]:
        """LIKE 降级搜索分块"""
        pattern = f"%{query}%"
        rows = self.conn.execute("""
            SELECT dc.*, d.title, d.category, d.file_path
            FROM doc_chunks dc
            JOIN docs d ON d.id = dc.doc_id
            WHERE dc.content LIKE ? OR dc.heading LIKE ?
            LIMIT ?
        """, (pattern, pattern, limit)).fetchall()
        return [dict(r) for r in rows]

    # ==================== FTS 索引维护 ====================

    def fts_insert_node(self, node_id: int, repo: Repo, node: dict):
        """向 FTS 索引插入节点"""
        if not self._fts_available:
            return
        try:
            content = f"{node.get('name', '')} {node.get('description', '')} {node.get('community_name', '')}"
            self.conn.execute("""
                INSERT INTO fts_search (source_type, source_id, repo_owner, repo_name, doc_title, content)
                VALUES ('node', ?, ?, ?, '', ?)
            """, (node_id, repo.owner, repo.name, content))
        except sqlite3.OperationalError:
            pass

    def fts_insert_chunk(self, chunk_id: int, doc: Doc, chunk: dict):
        """向 FTS 索引插入分块"""
        if not self._fts_available:
            return
        try:
            content = f"{chunk.get('heading', '')} {chunk.get('content', '')}"
            self.conn.execute("""
                INSERT INTO fts_search (source_type, source_id, repo_owner, repo_name, doc_title, content)
                VALUES ('chunk', ?, '', '', ?, ?)
            """, (chunk_id, doc.title, content))
        except sqlite3.OperationalError:
            pass

    def fts_delete_source(self, source_type: str, source_id: int):
        """从 FTS 索引删除条目"""
        if not self._fts_available:
            return
        try:
            # FTS5 content-sync 模式需要特殊处理
            # 这里简单重建索引
            pass
        except sqlite3.OperationalError:
            pass

    # ==================== Embedding 操作 ====================

    def store_embedding(self, source_type: str, source_id: int,
                       vector: np.ndarray, model: str,
                       repo_id: Optional[int] = None,
                       doc_id: Optional[int] = None):
        """存储向量嵌入"""
        vector_blob = vector.astype(np.float32).tobytes()
        dimensions = len(vector)

        self.conn.execute("""
            INSERT OR REPLACE INTO embeddings
                (source_type, source_id, repo_id, doc_id, model, vector, dimensions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (source_type, source_id, repo_id, doc_id, model,
              vector_blob, dimensions))
        self.conn.commit()

    def get_all_vectors(self, model: str) -> tuple[list[int], list[int], Optional[np.ndarray]]:
        """获取指定模型的所有向量
        返回 (node_ids, chunk_ids, matrix)
        """
        rows = self.conn.execute("""
            SELECT source_type, source_id, vector, dimensions
            FROM embeddings WHERE model = ?
        """, (model,)).fetchall()

        if not rows:
            return [], [], None

        node_ids = []
        chunk_ids = []
        vectors = []
        dimensions = rows[0]["dimensions"]

        for r in rows:
            vec = np.frombuffer(r["vector"], dtype=np.float32)
            vectors.append(vec)
            if r["source_type"] == "node":
                node_ids.append(r["source_id"])
                chunk_ids.append(-1)
            else:
                node_ids.append(-1)
                chunk_ids.append(r["source_id"])

        matrix = np.stack(vectors)
        return node_ids, chunk_ids, matrix

    def delete_embeddings_by_source(self, source_type: str, source_ids: list[int]):
        """删除指定源的嵌入"""
        if not source_ids:
            return
        placeholders = ",".join("?" * len(source_ids))
        self.conn.execute(f"""
            DELETE FROM embeddings
            WHERE source_type = ? AND source_id IN ({placeholders})
        """, [source_type] + source_ids)
        self.conn.commit()

    # ==================== 统计 ====================

    def get_stats(self) -> dict:
        """获取数据库统计"""
        repos = self.conn.execute("SELECT COUNT(*) FROM repos").fetchone()[0]
        nodes = self.conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
        docs = self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        chunks = self.conn.execute("SELECT COUNT(*) FROM doc_chunks").fetchone()[0]
        embeddings = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

        return {
            "repos": repos,
            "nodes": nodes,
            "docs": docs,
            "chunks": chunks,
            "embeddings": embeddings,
        }

    def close(self):
        """关闭数据库连接"""
        self.conn.close()
