"""
PostgreSQL 数据库初始化脚本
用于创建 LangGraph 检查点所需的表结构
"""
import os
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

# 构建 DSN
PG_DSN = os.getenv("PG_DSN")
if not PG_DSN:
    PG_HOST = os.getenv("PG_HOST", "localhost")
    PG_PORT = os.getenv("PG_PORT", "5432")
    PG_USER = os.getenv("PG_USER", "postgres")
    PG_PASSWORD = os.getenv("PG_PASSWORD", "")
    PG_DBNAME = os.getenv("PG_DBNAME", "langgraph_db")
    PG_DSN = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DBNAME}"

def setup_postgres_tables():
    """初始化 PostgreSQL 检查点表结构"""
    print(f"正在连接数据库: {PG_HOST}/{PG_DBNAME}")
    try:
        with PostgresSaver.from_conn_string(PG_DSN) as saver:
            saver.setup()
            print("✅ PostgreSQL 表结构初始化成功！")
            print("已创建的表: checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("\n请检查:")
        print("  1. PostgreSQL 服务是否运行")
        print("  2. 数据库是否存在")
        print("  3. 用户名/密码是否正确")
        print(f"  4. 是否已安装依赖: pip install langgraph-checkpoint-postgres")

if __name__ == "__main__":
    setup_postgres_tables()
