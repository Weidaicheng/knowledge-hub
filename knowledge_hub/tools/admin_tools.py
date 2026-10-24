"""管理工具 — 上传/删除图谱和文档"""

import json
from pathlib import Path
from typing import Optional

from ..storage.db import Database
from ..engines.graph_engine import GraphEngine
from ..engines.embedding_engine import EmbeddingEngine
from ..engines.doc_engine import DocEngine
from ..models import Config, Repo


def register_admin_tools(mcp, db: Database, graph_engine: GraphEngine,
                        embedding_engine: EmbeddingEngine, doc_engine: DocEngine,
                        config: Config):
    """注册管理工具"""

    @mcp.tool()
    def upload_graph(owner: str, repo: str, graph_json: str) -> str:
        """上传/更新仓库图谱

        将 graphify 生成的 graph.json 上传到知识库。

        Args:
            owner: 仓库所有者
            repo: 仓库名称
            graph_json: graph.json 的完整内容（JSON 字符串）

        Returns:
            JSON 字符串，包含上传结果（节点数、边数、社区数）
        """
        # 解析 graph.json
        try:
            data = json.loads(graph_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})

        # 保存到文件
        graph_dir = Path(config.data_dir) / "graphs" / owner / repo
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graph_dir / "graph.json"
        graph_path.write_text(graph_json, encoding="utf-8")

        # 统计节点和边
        if "edges" in data and "links" not in data:
            data["links"] = data.pop("edges")

        nodes = data.get("nodes", [])
        edges = data.get("links", [])
        communities = set()
        for n in nodes:
            cid = n.get("community")
            if cid is not None:
                communities.add(cid)

        # 注册仓库
        repo_id = db.upsert_repo(
            owner=owner,
            name=repo,
            graph_path=str(graph_path),
            node_count=len(nodes),
            edge_count=len(edges),
            community_count=len(communities),
        )

        # 加载图谱
        repo_info = db.get_repo(owner, repo)
        if repo_info:
            graph_engine.load_graph(repo_info, str(graph_path))

            # 提取节点到数据库
            db.delete_nodes_by_repo(repo_id)
            node_list = graph_engine.extract_nodes(repo_id)
            db.insert_nodes(repo_id, node_list)

            # 生成 embedding
            if config.embedding_enabled and embedding_engine.enabled:
                _generate_node_embeddings(db, graph_engine, embedding_engine,
                                         repo_id, node_list)

        return json.dumps({
            "ok": True,
            "repo": f"{owner}/{repo}",
            "nodes": len(nodes),
            "edges": len(edges),
            "communities": len(communities),
        }, ensure_ascii=False, indent=2)

    @mcp.tool()
    def upload_doc(title: str, category: str, content: str,
                  file_path: str = "") -> str:
        """上传/更新 Markdown 文档

        将 Markdown 文档上传到知识库。

        Args:
            title: 文档标题
            category: 分类路径（如 "product/auth"）
            content: Markdown 内容
            file_path: 文件路径（可选，默认使用 title）

        Returns:
            JSON 字符串，包含上传结果
        """
        # 确定文件路径
        if not file_path:
            safe_title = title.replace("/", "_").replace(" ", "_")
            file_path = f"{category}/{safe_title}.md" if category else f"{safe_title}.md"

        # 保存到文件
        doc_dir = Path(config.data_dir) / "docs"
        doc_path = doc_dir / file_path
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(content, encoding="utf-8")

        # 注册文档
        doc_id = db.upsert_doc(
            title=title,
            file_path=file_path,
            content=content,
            category=category,
        )

        # 分块
        chunks = doc_engine.chunk_markdown(content, title)
        db.insert_chunks(doc_id, chunks)

        # 生成 embedding
        if config.embedding_enabled and embedding_engine.enabled:
            _generate_chunk_embeddings(db, doc_engine, embedding_engine,
                                      doc_id, chunks, title)

        return json.dumps({
            "ok": True,
            "doc_id": doc_id,
            "title": title,
            "category": category,
            "chunks": len(chunks),
        }, ensure_ascii=False, indent=2)

    @mcp.tool()
    def remove_graph(owner: str, repo: str) -> str:
        """删除仓库图谱

        Args:
            owner: 仓库所有者
            repo: 仓库名称

        Returns:
            JSON 字符串，包含删除结果
        """
        success = db.delete_repo(owner, repo)
        if success:
            # 卸载图谱
            repo_info = db.get_repo(owner, repo)
            if repo_info and repo_info.id:
                graph_engine.unload_graph(repo_info.id)

            # 删除文件
            graph_dir = Path(config.data_dir) / "graphs" / owner / repo
            if graph_dir.exists():
                import shutil
                shutil.rmtree(graph_dir)

            return json.dumps({
                "ok": True,
                "message": f"Graph for {owner}/{repo} deleted",
            })
        else:
            return json.dumps({
                "ok": False,
                "error": f"Graph for {owner}/{repo} not found",
            })

    @mcp.tool()
    def remove_doc(file_path: str) -> str:
        """删除文档

        Args:
            file_path: 文档路径

        Returns:
            JSON 字符串，包含删除结果
        """
        success = db.delete_doc(file_path)
        if success:
            # 删除文件
            doc_path = Path(config.data_dir) / "docs" / file_path
            if doc_path.exists():
                doc_path.unlink()

            return json.dumps({
                "ok": True,
                "message": f"Document '{file_path}' deleted",
            })
        else:
            return json.dumps({
                "ok": False,
                "error": f"Document '{file_path}' not found",
            })

    @mcp.tool()
    def list_repos() -> str:
        """列出所有已注册的仓库

        Returns:
            JSON 字符串，包含仓库列表
        """
        repos = db.list_repos()
        result = []
        for r in repos:
            result.append({
                "owner": r.owner,
                "name": r.name,
                "node_count": r.node_count,
                "edge_count": r.edge_count,
                "community_count": r.community_count,
                "updated_at": r.updated_at,
            })
        return json.dumps({
            "repos": result,
            "total": len(result),
        }, ensure_ascii=False, indent=2)


def _generate_node_embeddings(db: Database, graph_engine: GraphEngine,
                             embedding_engine: EmbeddingEngine,
                             repo_id: int, nodes: list[dict]):
    """为节点生成 embedding"""
    texts = []
    node_ids = []

    for node in nodes:
        text = graph_engine.get_node_text(node)
        if text.strip():
            texts.append(text)
            # 获取数据库中的 node id
            row = db.conn.execute(
                "SELECT id FROM graph_nodes WHERE repo_id = ? AND node_id = ?",
                (repo_id, node["node_id"])).fetchone()
            if row:
                node_ids.append(row["id"])

    if not texts:
        return

    # 批量生成 embedding
    vectors = embedding_engine.embed_batch(texts)

    # 存储
    for node_id, vec in zip(node_ids, vectors):
        db.store_embedding(
            source_type="node",
            source_id=node_id,
            vector=vec,
            model=embedding_engine.model_name,
            repo_id=repo_id,
        )


def _generate_chunk_embeddings(db: Database, doc_engine: DocEngine,
                              embedding_engine: EmbeddingEngine,
                              doc_id: int, chunks: list[dict], title: str):
    """为文档分块生成 embedding"""
    texts = []
    chunk_ids = []

    for i, chunk in enumerate(chunks):
        text = doc_engine.get_chunk_text(chunk, title)
        if text.strip():
            texts.append(text)
            # 获取数据库中的 chunk id
            row = db.conn.execute(
                "SELECT id FROM doc_chunks WHERE doc_id = ? AND chunk_index = ?",
                (doc_id, i)).fetchone()
            if row:
                chunk_ids.append(row["id"])

    if not texts:
        return

    # 批量生成 embedding
    vectors = embedding_engine.embed_batch(texts)

    # 存储
    for chunk_id, vec in zip(chunk_ids, vectors):
        db.store_embedding(
            source_type="chunk",
            source_id=chunk_id,
            vector=vec,
            model=embedding_engine.model_name,
            doc_id=doc_id,
        )
