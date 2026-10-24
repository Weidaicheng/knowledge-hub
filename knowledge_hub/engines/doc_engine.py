"""文档引擎 — Markdown 分块、索引、检索"""

import re
from typing import Optional

from ..models import Doc, DocChunk, Config


class DocEngine:
    """文档处理引擎

    负责：
    - Markdown 文档分块
    - 分块文本提取（用于 embedding）
    """

    def __init__(self, config: Config):
        self.config = config
        self.chunk_max_chars = config.docs_chunk_max_chars

    def chunk_markdown(self, content: str, title: str = "") -> list[dict]:
        """将 Markdown 文档按标题分块

        分块策略：
        1. 按二级标题（##）自然分块
        2. 每个 chunk 保留所属的文档标题和章节 heading
        3. 若某节过长（>chunk_max_chars），按段落进一步切分

        Returns:
            [{"content": str, "heading": str}, ...]
        """
        if not content.strip():
            return []

        chunks = []
        current_heading = title  # 初始 heading 为文档标题
        current_content = []
        current_chars = 0

        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i]

            # 检测标题行
            heading_match = re.match(r'^(#{1,3})\s+(.+)$', line)

            if heading_match:
                # 保存之前的 chunk
                if current_content:
                    chunk_text = "\n".join(current_content).strip()
                    if chunk_text:
                        # 如果内容过长，进一步切分
                        if len(chunk_text) > self.chunk_max_chars:
                            sub_chunks = self._split_long_content(chunk_text)
                            for sc in sub_chunks:
                                chunks.append({
                                    "content": sc,
                                    "heading": current_heading,
                                })
                        else:
                            chunks.append({
                                "content": chunk_text,
                                "heading": current_heading,
                            })

                # 开始新的 chunk
                level = len(heading_match.group(1))
                heading_text = heading_match.group(2).strip()

                # 只有一级标题时更新 current_heading
                if level <= 2:
                    current_heading = heading_text
                else:
                    current_heading = f"{current_heading} > {heading_text}"

                current_content = [line]
                current_chars = len(line)
            else:
                current_content.append(line)
                current_chars += len(line)

            i += 1

        # 保存最后一个 chunk
        if current_content:
            chunk_text = "\n".join(current_content).strip()
            if chunk_text:
                if len(chunk_text) > self.chunk_max_chars:
                    sub_chunks = self._split_long_content(chunk_text)
                    for sc in sub_chunks:
                        chunks.append({
                            "content": sc,
                            "heading": current_heading,
                        })
                else:
                    chunks.append({
                        "content": chunk_text,
                        "heading": current_heading,
                    })

        return chunks

    def _split_long_content(self, content: str) -> list[str]:
        """将过长的内容按段落切分"""
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        current_chunk = []
        current_chars = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if current_chars + len(para) > self.chunk_max_chars and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_chars = 0

            current_chunk.append(para)
            current_chars += len(para)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def get_chunk_text(self, chunk: dict, doc_title: str = "") -> str:
        """获取分块的文本表示（用于 embedding）

        组合 heading + content，便于语义理解
        """
        parts = []
        if doc_title:
            parts.append(doc_title)
        if chunk.get("heading"):
            parts.append(chunk["heading"])
        parts.append(chunk.get("content", ""))
        return "\n".join(parts)

    def expand_context(self, chunks: list[dict], all_chunks: list[dict],
                      context_window: int = 1) -> list[dict]:
        """扩展分块上下文

        对于匹配到的 chunk，取前后相邻的 chunk 作为上下文

        Args:
            chunks: 匹配到的分块列表（带 chunk_index）
            all_chunks: 文档的所有分块
            context_window: 上下文窗口大小

        Returns:
            扩展后的分块列表
        """
        if not chunks or not all_chunks:
            return chunks

        # 构建索引映射
        chunk_map = {c.get("chunk_index", i): c for i, c in enumerate(all_chunks)}
        expanded = []
        seen_indices = set()

        for chunk in chunks:
            idx = chunk.get("chunk_index", 0)

            # 添加上下文
            for offset in range(-context_window, context_window + 1):
                ctx_idx = idx + offset
                if ctx_idx in chunk_map and ctx_idx not in seen_indices:
                    ctx_chunk = chunk_map[ctx_idx].copy()
                    ctx_chunk["is_match"] = (offset == 0)
                    expanded.append(ctx_chunk)
                    seen_indices.add(ctx_idx)

        # 按索引排序
        expanded.sort(key=lambda x: x.get("chunk_index", 0))
        return expanded
