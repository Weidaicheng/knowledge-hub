"""图谱引擎 — 加载、查询、遍历 graphify 图谱"""

import json
from collections import deque
from pathlib import Path
from typing import Optional

import networkx as nx
from networkx.readwrite import json_graph

from ..models import Repo, GraphNode


class GraphEngine:
    """图谱查询引擎

    支持多个仓库的独立图谱，提供节点查询、邻居遍历、社区查询等功能。
    """

    def __init__(self):
        # repo_id -> NetworkX Graph
        self._graphs: dict[int, nx.Graph] = {}
        # repo_id -> repo info
        self._repos: dict[int, Repo] = {}
        # repo_id -> {community_id: [node_ids]}
        self._communities: dict[int, dict[int, list[str]]] = {}

    def load_graph(self, repo: Repo, graph_path: str) -> bool:
        """加载指定仓库的图谱"""
        if not Path(graph_path).exists():
            return False

        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 兼容 graphify 输出格式
        if "edges" in data and "links" not in data:
            data["links"] = data.pop("edges")

        directed = data.get("directed", False)
        G = json_graph.node_link_graph(
            data, edges="links", directed=directed, multigraph=False)

        self._graphs[repo.id] = G
        self._repos[repo.id] = repo
        self._build_communities(repo.id)
        return True

    def unload_graph(self, repo_id: int):
        """卸载指定仓库的图谱"""
        self._graphs.pop(repo_id, None)
        self._repos.pop(repo_id, None)
        self._communities.pop(repo_id, None)

    def _build_communities(self, repo_id: int):
        """构建社区索引"""
        G = self._graphs.get(repo_id)
        if not G:
            return

        communities: dict[int, list[str]] = {}
        for nid, data in G.nodes(data=True):
            cid = data.get("community", 0)
            communities.setdefault(cid, []).append(nid)
        self._communities[repo_id] = communities

    @property
    def loaded_repo_ids(self) -> list[int]:
        """返回已加载图谱的仓库 ID 列表"""
        return list(self._graphs.keys())

    def is_loaded(self, repo_id: Optional[int] = None) -> bool:
        """检查图谱是否已加载"""
        if repo_id is not None:
            return repo_id in self._graphs
        return len(self._graphs) > 0

    # ==================== 统计 ====================

    def stats(self, repo_id: Optional[int] = None) -> dict:
        """返回图谱统计信息"""
        if repo_id is not None:
            G = self._graphs.get(repo_id)
            if not G:
                return {"error": f"Graph for repo {repo_id} not loaded"}
            return self._graph_stats(G, repo_id)

        # 汇总所有仓库
        total_nodes = 0
        total_edges = 0
        repo_stats = []
        for rid, G in self._graphs.items():
            repo = self._repos.get(rid)
            stats = self._graph_stats(G, rid)
            total_nodes += stats["nodes"]
            total_edges += stats["edges"]
            repo_stats.append({
                "repo_id": rid,
                "name": repo.full_name if repo else f"repo_{rid}",
                "nodes": stats["nodes"],
                "edges": stats["edges"],
                "communities": stats["communities"],
            })
        return {
            "repos": repo_stats,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        }

    def _graph_stats(self, G: nx.Graph, repo_id: int) -> dict:
        """单个图谱的统计"""
        communities = self._communities.get(repo_id, {})
        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "communities": len(communities),
        }

    # ==================== 节点查询 ====================

    def find_node(self, label: str, repo_id: Optional[int] = None) -> Optional[dict]:
        """按名称查找节点

        支持分层匹配：精确 ID → 精确 label → 前缀 → 子串
        """
        repos_to_search = [repo_id] if repo_id else self._graphs.keys()

        for rid in repos_to_search:
            G = self._graphs.get(rid)
            if not G:
                continue

            label_lower = label.lower()
            for tier in ("exact_id", "exact_label", "prefix", "substring"):
                matches = []
                for nid, data in G.nodes(data=True):
                    nl = data.get("label", nid)
                    if tier == "exact_id" and nid == label:
                        matches.append((nid, data))
                    elif tier == "exact_label" and nl.lower() == label_lower:
                        matches.append((nid, data))
                    elif tier == "prefix" and nl.lower().startswith(label_lower):
                        matches.append((nid, data))
                    elif tier == "substring" and label_lower in nl.lower():
                        matches.append((nid, data))

                if matches:
                    repo = self._repos.get(rid)
                    if len(matches) == 1:
                        nid, data = matches[0]
                        return {
                            "repo_id": rid,
                            "repo": repo.full_name if repo else f"repo_{rid}",
                            "id": nid,
                            "label": data.get("label", nid),
                            "source_file": data.get("source_file", ""),
                            "source_location": data.get("source_location", ""),
                            "file_type": data.get("file_type", ""),
                            "community": data.get("community"),
                            "community_name": data.get("community_name", ""),
                            "degree": G.degree(nid),
                        }
                    # 多个匹配，返回歧义
                    return {
                        "repo_id": rid,
                        "repo": repo.full_name if repo else f"repo_{rid}",
                        "ambiguous": True,
                        "matches": [
                            {"id": nid, "label": d.get("label", nid),
                             "degree": G.degree(nid)}
                            for nid, d in matches[:10]
                        ],
                    }
        return None

    def get_neighbors(self, label: str, repo_id: Optional[int] = None,
                     depth: int = 1, limit: int = 30) -> dict:
        """获取节点的邻居"""
        node = self.find_node(label, repo_id)
        if not node:
            return {"error": f"Node '{label}' not found"}
        if node.get("ambiguous"):
            return {"error": f"Node '{label}' is ambiguous", "matches": node.get("matches", [])}

        rid = node["repo_id"]
        G = self._graphs.get(rid)
        if not G:
            return {"error": "Graph not loaded"}

        nid = node["id"]
        neighbors = []

        # BFS 扩展
        visited = {nid}
        queue = deque([(nid, 0)])

        while queue and len(neighbors) < limit:
            current, d = queue.popleft()
            if d >= depth:
                continue

            # 出边
            for neighbor_id in G.neighbors(current):
                edge = G.edges.get((current, neighbor_id)) or \
                       G.edges.get((neighbor_id, current)) or {}
                nd = G.nodes.get(neighbor_id, {})
                neighbors.append({
                    "id": neighbor_id,
                    "label": nd.get("label", neighbor_id),
                    "relation": edge.get("relation", ""),
                    "confidence": edge.get("confidence", ""),
                    "direction": "outgoing" if G.has_edge(current, neighbor_id) else "incoming",
                    "source_file": nd.get("source_file", ""),
                })
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, d + 1))

            # 入边（有向图）
            if G.is_directed():
                for pred_id in G.predecessors(current):
                    if pred_id in visited:
                        continue
                    edge = G.edges.get((pred_id, current)) or {}
                    pd = G.nodes.get(pred_id, {})
                    neighbors.append({
                        "id": pred_id,
                        "label": pd.get("label", pred_id),
                        "relation": edge.get("relation", ""),
                        "confidence": edge.get("confidence", ""),
                        "direction": "incoming",
                        "source_file": pd.get("source_file", ""),
                    })
                    visited.add(pred_id)
                    queue.append((pred_id, d + 1))

        return {
            "node": node,
            "neighbors": neighbors[:limit],
            "total": len(neighbors),
        }

    def get_community(self, community_id: int, repo_id: Optional[int] = None,
                     limit: int = 50) -> dict:
        """获取社区中的节点"""
        repos_to_search = [repo_id] if repo_id else self._graphs.keys()

        for rid in repos_to_search:
            communities = self._communities.get(rid, {})
            if community_id not in communities:
                continue

            G = self._graphs.get(rid)
            if not G:
                continue

            members = communities[community_id]
            nodes = []
            for nid in members[:limit]:
                d = G.nodes.get(nid, {})
                nodes.append({
                    "id": nid,
                    "label": d.get("label", nid),
                    "source_file": d.get("source_file", ""),
                    "degree": G.degree(nid),
                })

            name = ""
            if members:
                name = G.nodes[members[0]].get("community_name", "")

            repo = self._repos.get(rid)
            return {
                "repo_id": rid,
                "repo": repo.full_name if repo else f"repo_{rid}",
                "community_id": community_id,
                "name": name,
                "nodes": nodes,
                "total": len(members),
            }

        return {"error": f"Community {community_id} not found"}

    def god_nodes(self, repo_id: Optional[int] = None, top_n: int = 10) -> list:
        """返回度数最高的核心节点"""
        repos_to_search = [repo_id] if repo_id else self._graphs.keys()
        all_gods = []

        for rid in repos_to_search:
            G = self._graphs.get(rid)
            if not G:
                continue

            degrees = sorted(G.degree, key=lambda x: x[1], reverse=True)
            repo = self._repos.get(rid)

            for nid, deg in degrees[:top_n]:
                d = G.nodes.get(nid, {})
                if d.get("file_type") in ("document", "paper"):
                    continue
                all_gods.append({
                    "repo_id": rid,
                    "repo": repo.full_name if repo else f"repo_{rid}",
                    "id": nid,
                    "label": d.get("label", nid),
                    "degree": deg,
                    "community_name": d.get("community_name", ""),
                })

        # 按度数排序
        all_gods.sort(key=lambda x: -x["degree"])
        return all_gods[:top_n]

    # ==================== 节点提取（用于索引） ====================

    def extract_nodes(self, repo_id: int) -> list[dict]:
        """从图谱中提取所有节点信息（用于数据库索引）"""
        G = self._graphs.get(repo_id)
        if not G:
            return []

        nodes = []
        for nid, data in G.nodes(data=True):
            nodes.append({
                "node_id": nid,
                "name": data.get("label", nid),
                "node_type": data.get("file_type"),
                "description": data.get("description", ""),
                "file_path": data.get("source_file", ""),
                "community": data.get("community"),
                "community_name": data.get("community_name", ""),
            })
        return nodes

    def get_node_text(self, node: dict) -> str:
        """获取节点的文本表示（用于 embedding）"""
        parts = [node.get("name", "")]
        if node.get("description"):
            parts.append(node["description"])
        if node.get("community_name"):
            parts.append(node["community_name"])
        return " ".join(parts)
