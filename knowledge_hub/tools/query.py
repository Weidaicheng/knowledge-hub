"""统一查询工具"""

import json
from typing import Optional

from ..engines.hybrid_retriever import HybridRetriever
from ..models import Config


def register_query_tools(mcp, retriever: HybridRetriever, config: Config):
    """注册统一查询工具"""

    @mcp.tool()
    def query(question: str, top_k: int = 10) -> str:
        """统一查询 — 同时检索代码图谱和文档

        输入自然语言问题，返回来自代码图谱和文档两个维度的结构化结果。
        AI 客户端应将这些信息整合成连贯的回答。

        Args:
            question: 自然语言问题
            top_k: 每类返回的最大结果数（默认 10）

        Returns:
            JSON 字符串，包含 graph_results 和 doc_results
        """
        result = retriever.query(question, top_k=top_k)
        return json.dumps({
            "graph_results": result.graph_results,
            "doc_results": result.doc_results,
            "hint": result.hint,
        }, ensure_ascii=False, indent=2)
