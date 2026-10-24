"""Embedding 引擎 — 支持本地模型和 OpenAI API"""

import os
from typing import Optional

import numpy as np

from ..models import Config


class EmbeddingEngine:
    """向量嵌入引擎

    支持两种后端：
    - local: 使用 sentence-transformers 本地模型
    - openai: 使用 OpenAI Embeddings API
    """

    def __init__(self, config: Config):
        self.config = config
        self._model = None
        self._backend = config.embedding_backend
        self._model_name = config.embedding_model
        self._openai_client = None
        self._openai_model = config.embedding_openai_model
        self._dimensions: Optional[int] = None
        self._enabled = config.embedding_enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def model_name(self) -> str:
        if self._backend == "openai":
            return self._openai_model
        return self._model_name

    @property
    def dimensions(self) -> int:
        if self._dimensions:
            return self._dimensions
        # 尝试获取维度
        if self._backend == "openai":
            # OpenAI 模型维度
            dims = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            }
            self._dimensions = dims.get(self._openai_model, 1536)
        else:
            # 本地模型需要加载后才能获取
            if self._model is not None:
                self._dimensions = self._model.get_sentence_embedding_dimension()
            else:
                # 默认值
                self._dimensions = 512
        return self._dimensions

    def _load_local_model(self):
        """加载本地 sentence-transformers 模型"""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dimensions = self._model.get_sentence_embedding_dimension()
        except ImportError:
            raise RuntimeError(
                "sentence-transformers 未安装。请运行: pip install sentence-transformers")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")

    def _get_openai_client(self):
        """获取 OpenAI 客户端"""
        if self._openai_client is not None:
            return self._openai_client

        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai 未安装。请运行: pip install openai")

        api_key = self.config.embedding_openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key 未配置。请设置环境变量 OPENAI_API_KEY 或在配置文件中指定")

        self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def embed(self, text: str) -> np.ndarray:
        """生成单个文本的向量"""
        if self._backend == "openai":
            return self._embed_openai(text)
        else:
            return self._embed_local(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """批量生成向量"""
        if not texts:
            return []

        if self._backend == "openai":
            return self._embed_batch_openai(texts)
        else:
            return self._embed_batch_local(texts)

    def _embed_local(self, text: str) -> np.ndarray:
        """本地模型生成向量"""
        self._load_local_model()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    def _embed_batch_local(self, texts: list[str]) -> list[np.ndarray]:
        """本地模型批量生成"""
        self._load_local_model()

        batch_size = self.config.embedding_batch_size
        all_vecs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            vecs = self._model.encode(batch, normalize_embeddings=True)
            all_vecs.extend([v.astype(np.float32) for v in vecs])

        return all_vecs

    def _embed_openai(self, text: str) -> np.ndarray:
        """OpenAI API 生成向量"""
        client = self._get_openai_client()
        response = client.embeddings.create(
            model=self._openai_model,
            input=text
        )
        vec = response.data[0].embedding
        self._dimensions = len(vec)
        return np.array(vec, dtype=np.float32)

    def _embed_batch_openai(self, texts: list[str]) -> list[np.ndarray]:
        """OpenAI API 批量生成"""
        client = self._get_openai_client()

        batch_size = self.config.embedding_batch_size
        all_vecs = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=self._openai_model,
                input=batch
            )
            for item in response.data:
                vec = np.array(item.embedding, dtype=np.float32)
                # 归一化
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                all_vecs.append(vec)

        if all_vecs:
            self._dimensions = len(all_vecs[0])

        return all_vecs

    def search(self, query: str, all_vectors: Optional[np.ndarray],
              source_ids: list[int], top_k: int = 10) -> list[dict]:
        """向量相似度搜索

        Args:
            query: 查询文本
            all_vectors: 所有候选向量矩阵 (n, dim)
            source_ids: 对应的源 ID 列表
            top_k: 返回数量

        Returns:
            [{"source_id": int, "score": float}, ...]
        """
        if all_vectors is None or len(source_ids) == 0:
            return []

        # 生成查询向量
        qvec = self.embed(query)

        # 计算余弦相似度（向量已归一化，直接点积）
        scores = all_vectors @ qvec

        # 取 top_k
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_idx:
            results.append({
                "source_id": source_ids[idx],
                "score": float(scores[idx]),
            })

        return results

    def warmup(self):
        """预热模型（提前加载）"""
        if self._backend == "local":
            self._load_local_model()
        else:
            self._get_openai_client()
