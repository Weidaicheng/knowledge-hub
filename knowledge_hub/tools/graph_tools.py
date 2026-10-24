"""图谱查询工具"""

import json
from typing import Optional

from ..engines.graph_engine import GraphEngine
from ..engines.hybrid_retriever import HybridRetriever
from ..storage.db import Database
from ..models import Config


def register_graph_tools(mcp, graph_engine: GraphEngine,
                        retriever: HybridRetriever, db: Database, config: Config):
    """注册图谱查询工具"""

    @mcp.tool()
    def query_code_graph(question: str, repos: list[str] = [],
                        top_k: int = 10) -> str:
        """仅查询代码知识图谱

        输入自然语言问题，在代码图谱中检索相关节点和上下文。

        Args:
            question: 自然语言问题
            repos: 限定仓库范围（格式 "owner/repo"），空列表表示全部
            top_k: 返回的最大结果数

        Returns:
            JSON 字符串，包含匹配的节点及其邻居
        """
        # 解析 repo 过滤
        repo_ids = None
        if repos:
            repo_ids = []
            for repo_name in repos:
                parts = repo_name.split("/")
                if len(parts) == 2:
                    repo = db.get_repo(parts[0], parts[1])
                    if repo and repo.id:
                        repo_ids.append(repo.id)

        results = retriever.search_graph(question, top_k=top_k, repo_ids=repo_ids)
        return json.dumps({
            "graph_results": results,
        }, ensure_ascii=False, indent=2)

    @mcp.tool()
    def graph_stats(repo: str = "") -> str:
        """查看图谱统计信息

        Args:
            repo: 指定仓库（格式 "owner/repo"），空字符串表示全部汇总

        Returns:
            JSON 字符串，包含节点数、边数、社区数等统计
        """
        if repo:
            parts = repo.split("/")
            if len(parts) == 2:
                r = db.get_repo(parts[0], parts[1])
                if r and r.id:
                    stats = graph_engine.stats(r.id)
                else:
                    return json.dumps({"error": f"Repo '{repo}' not found"})
            else:
                return json.dumps({"error": "Invalid repo format, use 'owner/repo'"})
        else:
            stats = graph_engine.stats()

        return json.dumps(stats, ensure_ascii=False, indent=2)

    @mcp.tool()
    def graph_neighbors(node_name: str, repo: str = "", depth: int = 1) -> str:
        """获取某节点的邻居（展开图结构）

        Args:
            node_name: 节点名称
            repo: 仓库（格式 "owner/repo"），空字符串表示在所有仓库中查找
            depth: 展开深度（默认 1 跳）

        Returns:
            JSON 字符串，包含节点及其邻居的完整子图
        """
        repo_id = None
        if repo:
            parts = repo.split("/")
            if len(parts) == 2:
                r = db.get_repo(parts[0], parts[1])
                if r and r.id:
                    repo_id = r.id

        result = graph_engine.get_neighbors(node_name, repo_id=repo_id, depth=depth)
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool()
    def graph_community(community_id: int, repo: str = "") -> str:
        """列出指定社区中的所有节点

        Args:
            community_id: 社区 ID
            repo: 仓库（格式 "owner/repo"），空字符串表示在所有仓库中查找

        Returns:
            JSON 字符串，包含社区中的节点列表
        """
        repo_id = None
        if repo:
            parts = repo.split("/")
            if len(parts) == 2:
                r = db.get_repo(parts[0], parts[1])
                if r and r.id:
                    repo_id = r.id

        result = graph_engine.get_community(community_id, repo_id=repo_id)
        return json.dumps(result, ensure_ascii=False, indent=2)

    @mcp.tool()
    def graph_god_nodes(repo: str = "", top_n: int = 10) -> str:
        """返回度数最高的核心节点（God Nodes）

        Args:
            repo: 仓库（格式 "owner/repo"），空字符串表示全部
            top_n: 返回数量

        Returns:
            JSON 字符串，包含核心节点列表
        """
        repo_id = None
        if repo:
            parts = repo.split("/")
            if len(parts) == 2:
                r = db.get_repo(parts[0], parts[1])
                if r and r.id:
                    repo_id = r.id

        results = graph_engine.god_nodes(repo_id=repo_id, top_n=top_n)
        return json.dumps(results, ensure_ascii=False, indent=2)
