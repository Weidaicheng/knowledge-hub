"""Knowledge Hub MCP Server — 入口"""

import argparse
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .models import Config
from .storage.db import Database
from .engines.graph_engine import GraphEngine
from .engines.embedding_engine import EmbeddingEngine
from .engines.doc_engine import DocEngine
from .engines.hybrid_retriever import HybridRetriever
from .tools.query import register_query_tools
from .tools.graph_tools import register_graph_tools
from .tools.doc_tools import register_doc_tools
from .tools.admin_tools import register_admin_tools


def load_config(config_path: str = None) -> Config:
    """加载配置"""
    if config_path and Path(config_path).exists():
        return Config.from_yaml(config_path)

    # 尝试默认位置
    default_paths = [
        Path("config.yaml"),
        Path("./config.yaml"),
        Path(__file__).parent.parent / "config.yaml",
    ]
    for p in default_paths:
        if p.exists():
            return Config.from_yaml(str(p))

    return Config()


def create_server(config: Config) -> tuple[FastMCP, dict]:
    """创建 MCP Server 实例

    Returns:
        (mcp_server, components) 组件字典用于后续清理
    """
    # 初始化数据库
    db_path = Path(config.data_dir) / "knowledge_hub.db"
    db = Database(str(db_path))

    # 初始化引擎
    graph_engine = GraphEngine()
    embedding_engine = EmbeddingEngine(config)
    doc_engine = DocEngine(config)
    retriever = HybridRetriever(db, graph_engine, embedding_engine, doc_engine, config)

    # 加载已有图谱
    for repo in db.list_repos():
        if repo.graph_path and Path(repo.graph_path).exists():
            if graph_engine.load_graph(repo, repo.graph_path):
                print(f"  Loaded graph: {repo.full_name}")

    # 预热 embedding 模型
    if config.embedding_enabled:
        try:
            embedding_engine.warmup()
            print(f"  Embedding model ready: {embedding_engine.model_name}")
        except Exception as e:
            print(f"  Warning: Embedding model failed to load: {e}")
            print("  Falling back to FTS5-only mode")
            config.embedding_enabled = False

    # 创建 MCP Server
    mcp = FastMCP("knowledge-hub")

    # 注册工具
    register_query_tools(mcp, retriever, config)
    register_graph_tools(mcp, graph_engine, retriever, db, config)
    register_doc_tools(mcp, retriever, db, config)
    register_admin_tools(mcp, db, graph_engine, embedding_engine, doc_engine, config)

    components = {
        "db": db,
        "graph_engine": graph_engine,
        "embedding_engine": embedding_engine,
        "doc_engine": doc_engine,
        "retriever": retriever,
    }

    return mcp, components


def main():
    """主入口"""
    parser = argparse.ArgumentParser(
        description="Knowledge Hub MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        help="配置文件路径（默认 config.yaml）",
    )
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "sse"],
        help="传输协议（覆盖配置文件）",
    )
    parser.add_argument(
        "--host",
        help="SSE 模式监听地址（覆盖配置文件）",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="SSE 模式监听端口（覆盖配置文件）",
    )
    parser.add_argument(
        "--data-dir", "-d",
        help="数据目录（覆盖配置文件）",
    )

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 命令行参数覆盖
    if args.transport:
        config.transport = args.transport
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.data_dir:
        config.data_dir = args.data_dir

    # 打印启动信息
    print("=" * 50)
    print("Knowledge Hub MCP Server")
    print("=" * 50)
    print(f"  Data dir:   {config.data_dir}")
    print(f"  Transport:  {config.transport}")
    if config.transport == "sse":
        print(f"  Endpoint:   http://{config.host}:{config.port}/mcp")
    print(f"  Embedding:  {'enabled' if config.embedding_enabled else 'disabled'}")
    print("-" * 50)

    # 创建并启动服务器
    try:
        mcp, components = create_server(config)

        print("-" * 50)
        print("Server ready. Waiting for connections...")
        print("Press Ctrl+C to stop.")
        print("=" * 50)

        if config.transport == "sse":
            mcp.run(transport="sse", host=config.host, port=config.port)
        else:
            mcp.run(transport="stdio")

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # 清理资源
        if "db" in components:
            components["db"].close()


if __name__ == "__main__":
    main()
