"""文档检索工具"""

import json
from typing import Optional

from ..engines.hybrid_retriever import HybridRetriever
from ..storage.db import Database
from ..models import Config


def register_doc_tools(mcp, retriever: HybridRetriever, db: Database, config: Config):
    """注册文档检索工具"""

    @mcp.tool()
    def search_docs(question: str, category: str = "", top_k: int = 10) -> str:
        """仅检索 Markdown 文档

        输入自然语言问题，在文档库中检索相关内容。

        Args:
            question: 自然语言问题
            category: 限定文档分类，空字符串表示全部
            top_k: 返回的最大结果数

        Returns:
            JSON 字符串，包含匹配的文档片段
        """
        results = retriever.search_docs(question, top_k=top_k,
                                       category=category or None)
        return json.dumps({
            "doc_results": results,
        }, ensure_ascii=False, indent=2)

    @mcp.tool()
    def list_docs(category: str = "") -> str:
        """列出所有文档及分类

        Args:
            category: 限定分类，空字符串表示全部

        Returns:
            JSON 字符串，包含文档列表
        """
        docs = db.list_docs(category=category or None)
        result = []
        for doc in docs:
            result.append({
                "id": doc.id,
                "title": doc.title,
                "file_path": doc.file_path,
                "category": doc.category,
                "chunk_count": doc.chunk_count,
                "created_at": doc.created_at,
                "updated_at": doc.updated_at,
            })
        return json.dumps({
            "documents": result,
            "total": len(result),
        }, ensure_ascii=False, indent=2)

    @mcp.tool()
    def get_doc(doc_id: int) -> str:
        """获取文档全文

        Args:
            doc_id: 文档 ID

        Returns:
            JSON 字符串，包含文档完整内容
        """
        doc = db.get_doc(doc_id)
        if not doc:
            return json.dumps({"error": f"Document {doc_id} not found"})

        return json.dumps({
            "id": doc.id,
            "title": doc.title,
            "file_path": doc.file_path,
            "category": doc.category,
            "content": doc.content,
            "chunk_count": doc.chunk_count,
        }, ensure_ascii=False, indent=2)
