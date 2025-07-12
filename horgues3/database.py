import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

def get_database_url() -> str:
    """
    環境変数からデータベースURLを取得する
    
    Returns:
    --------
    str
        PostgreSQLの接続URL
    """
    # 環境変数から接続情報を取得
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'horgues3')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'postgres')
    
    return f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

def create_database_engine():
    """
    データベースエンジンを作成する
    
    Returns:
    --------
    sqlalchemy.engine.Engine
        SQLAlchemyエンジン
    """
    return create_engine(get_database_url())