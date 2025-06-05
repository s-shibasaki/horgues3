import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def fetch_odds_data(start_ymd: str, end_ymd: str, engine=None) -> Dict[str, pd.DataFrame]:
    """
    指定期間のオッズデータを効率的に取得する
    
    Args:
        start_ymd: 開始日 (YYYYMMDD形式)
        end_ymd: 終了日 (YYYYMMDD形式) 
        engine: SQLAlchemyエンジン (未指定の場合はデフォルト接続)
        
    Returns:
        各馬券種のオッズデータを含む辞書
    """
    if engine is None:
        engine = create_engine("postgresql://postgres:postgres@localhost/horgues3")
    
    logger.info(f"Fetching odds data from {start_ymd} to {end_ymd}")
    
    # 日付フィルタ条件
    date_filter = f"""
        kaisai_year || kaisai_month_day BETWEEN '{start_ymd}' AND '{end_ymd}'
    """
    
    odds_data = {}
    
    # 各馬券種のクエリを定義
    queries = {
        'tansho': f"""
        WITH latest_odds AS (
            SELECT 
                kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number,
                MAX(creation_date || announce_month_day_hour_minute) as max_timestamp
            FROM public.odds_1_tan_fuku_waku
            WHERE {date_filter}
            GROUP BY kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number
        )
        SELECT 
            h.kaisai_year || h.kaisai_month_day || h.track_code || h.kaisai_kai || h.kaisai_day || h.race_number as race_id,
            CASE 
                WHEN o.horse_number = '  ' THEN NULL
                ELSE CAST(o.horse_number AS INTEGER)
            END as horse_number,
            NULL as horse_number_2,
            NULL as horse_number_3,
            CASE 
                WHEN o.odds IN ('0000', '----', '****', '    ') THEN 0
                ELSE CAST(o.odds AS INTEGER)
            END as odds_raw,
            CASE 
                WHEN o.popularity IN ('  ', '--', '**') THEN 0
                ELSE CAST(o.popularity AS INTEGER)
            END as popularity
        FROM public.odds_1_tan_fuku_waku h
        INNER JOIN latest_odds l ON 
            h.kaisai_year = l.kaisai_year AND
            h.kaisai_month_day = l.kaisai_month_day AND
            h.track_code = l.track_code AND
            h.kaisai_kai = l.kaisai_kai AND
            h.kaisai_day = l.kaisai_day AND
            h.race_number = l.race_number AND
            h.creation_date || h.announce_month_day_hour_minute = l.max_timestamp
        INNER JOIN public.odds_1_tan_fuku_waku_tansho_odds o ON
            h.kaisai_year = o.kaisai_year AND
            h.kaisai_month_day = o.kaisai_month_day AND
            h.track_code = o.track_code AND
            h.kaisai_kai = o.kaisai_kai AND
            h.kaisai_day = o.kaisai_day AND
            h.race_number = o.race_number AND
            h.announce_month_day_hour_minute = o.announce_month_day_hour_minute
        WHERE o.horse_number != '  '
        ORDER BY race_id, horse_number
        """,
        
        'fukusho': f"""
        WITH latest_odds AS (
            SELECT 
                kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number,
                MAX(creation_date || announce_month_day_hour_minute) as max_timestamp
            FROM public.odds_1_tan_fuku_waku
            WHERE {date_filter}
            GROUP BY kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number
        )
        SELECT 
            h.kaisai_year || h.kaisai_month_day || h.track_code || h.kaisai_kai || h.kaisai_day || h.race_number as race_id,
            CASE 
                WHEN o.horse_number = '  ' THEN NULL
                ELSE CAST(o.horse_number AS INTEGER)
            END as horse_number,
            NULL as horse_number_2,
            NULL as horse_number_3,
            CASE 
                WHEN o.min_odds IN ('0000', '----', '****', '    ') THEN 0
                ELSE CAST(o.min_odds AS INTEGER)
            END as odds_raw,
            CASE 
                WHEN o.popularity IN ('  ', '--', '**') THEN 0
                ELSE CAST(o.popularity AS INTEGER)
            END as popularity
        FROM public.odds_1_tan_fuku_waku h
        INNER JOIN latest_odds l ON 
            h.kaisai_year = l.kaisai_year AND
            h.kaisai_month_day = l.kaisai_month_day AND
            h.track_code = l.track_code AND
            h.kaisai_kai = l.kaisai_kai AND
            h.kaisai_day = l.kaisai_day AND
            h.race_number = l.race_number AND
            h.creation_date || h.announce_month_day_hour_minute = l.max_timestamp
        INNER JOIN public.odds_1_tan_fuku_waku_fukusho_odds o ON
            h.kaisai_year = o.kaisai_year AND
            h.kaisai_month_day = o.kaisai_month_day AND
            h.track_code = o.track_code AND
            h.kaisai_kai = o.kaisai_kai AND
            h.kaisai_day = o.kaisai_day AND
            h.race_number = o.race_number AND
            h.announce_month_day_hour_minute = o.announce_month_day_hour_minute
        WHERE o.horse_number != '  '
        ORDER BY race_id, horse_number
        """,
        
        'umaren': f"""
        WITH latest_odds AS (
            SELECT 
                kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number,
                MAX(creation_date || announce_month_day_hour_minute) as max_timestamp
            FROM public.odds_2_umaren
            WHERE {date_filter}
            GROUP BY kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number
        )
        SELECT 
            h.kaisai_year || h.kaisai_month_day || h.track_code || h.kaisai_kai || h.kaisai_day || h.race_number as race_id,
            CASE 
                WHEN o.combination = '    ' THEN NULL
                ELSE CAST(SUBSTRING(o.combination, 1, 2) AS INTEGER)
            END as horse_number,
            CASE 
                WHEN o.combination = '    ' THEN NULL
                ELSE CAST(SUBSTRING(o.combination, 3, 2) AS INTEGER)
            END as horse_number_2,
            NULL as horse_number_3,
            CASE 
                WHEN o.odds IN ('000000', '------', '******', '      ') THEN 0
                ELSE CAST(o.odds AS INTEGER)
            END as odds_raw,
            CASE 
                WHEN o.popularity IN ('   ', '---', '***') THEN 0
                ELSE CAST(o.popularity AS INTEGER)
            END as popularity
        FROM public.odds_2_umaren h
        INNER JOIN latest_odds l ON 
            h.kaisai_year = l.kaisai_year AND
            h.kaisai_month_day = l.kaisai_month_day AND
            h.track_code = l.track_code AND
            h.kaisai_kai = l.kaisai_kai AND
            h.kaisai_day = l.kaisai_day AND
            h.race_number = l.race_number AND
            h.creation_date || h.announce_month_day_hour_minute = l.max_timestamp
        INNER JOIN public.odds_2_umaren_umaren_odds o ON
            h.kaisai_year = o.kaisai_year AND
            h.kaisai_month_day = o.kaisai_month_day AND
            h.track_code = o.track_code AND
            h.kaisai_kai = o.kaisai_kai AND
            h.kaisai_day = o.kaisai_day AND
            h.race_number = o.race_number AND
            h.announce_month_day_hour_minute = o.announce_month_day_hour_minute
        WHERE o.combination != '    '
        ORDER BY race_id, horse_number, horse_number_2
        """,
        
        'wide': f"""
        WITH latest_odds AS (
            SELECT 
                kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number,
                MAX(creation_date || announce_month_day_hour_minute) as max_timestamp
            FROM public.odds_3_wide
            WHERE {date_filter}
            GROUP BY kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number
        )
        SELECT 
            h.kaisai_year || h.kaisai_month_day || h.track_code || h.kaisai_kai || h.kaisai_day || h.race_number as race_id,
            CASE 
                WHEN o.combination = '    ' THEN NULL
                ELSE CAST(SUBSTRING(o.combination, 1, 2) AS INTEGER)
            END as horse_number,
            CASE 
                WHEN o.combination = '    ' THEN NULL
                ELSE CAST(SUBSTRING(o.combination, 3, 2) AS INTEGER)
            END as horse_number_2,
            NULL as horse_number_3,
            CASE 
                WHEN o.min_odds IN ('00000', '-----', '*****', '     ') THEN 0
                ELSE CAST(o.min_odds AS INTEGER)
            END as odds_raw,
            CASE 
                WHEN o.popularity IN ('   ', '---', '***') THEN 0
                ELSE CAST(o.popularity AS INTEGER)
            END as popularity
        FROM public.odds_3_wide h
        INNER JOIN latest_odds l ON 
            h.kaisai_year = l.kaisai_year AND
            h.kaisai_month_day = l.kaisai_month_day AND
            h.track_code = l.track_code AND
            h.kaisai_kai = l.kaisai_kai AND
            h.kaisai_day = l.kaisai_day AND
            h.race_number = l.race_number AND
            h.creation_date || h.announce_month_day_hour_minute = l.max_timestamp
        INNER JOIN public.odds_3_wide_wide_odds o ON
            h.kaisai_year = o.kaisai_year AND
            h.kaisai_month_day = o.kaisai_month_day AND
            h.track_code = o.track_code AND
            h.kaisai_kai = o.kaisai_kai AND
            h.kaisai_day = o.kaisai_day AND
            h.race_number = o.race_number AND
            h.announce_month_day_hour_minute = o.announce_month_day_hour_minute
        WHERE o.combination != '    '
        ORDER BY race_id, horse_number, horse_number_2
        """,
        
        'umatan': f"""
        WITH latest_odds AS (
            SELECT 
                kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number,
                MAX(creation_date || announce_month_day_hour_minute) as max_timestamp
            FROM public.odds_4_umatan
            WHERE {date_filter}
            GROUP BY kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number
        )
        SELECT 
            h.kaisai_year || h.kaisai_month_day || h.track_code || h.kaisai_kai || h.kaisai_day || h.race_number as race_id,
            CASE 
                WHEN SUBSTRING(o.combination, 1, 2) = '  ' THEN NULL
                ELSE CAST(SUBSTRING(o.combination, 1, 2) AS INTEGER)
            END as horse_number,
            CASE 
                WHEN SUBSTRING(o.combination, 3, 2) = '  ' THEN NULL
                ELSE CAST(SUBSTRING(o.combination, 3, 2) AS INTEGER)
            END as horse_number_2,
            NULL as horse_number_3,
            CASE 
                WHEN o.odds IN ('000000', '------', '******', '      ') THEN 0
                ELSE CAST(o.odds AS INTEGER)
            END as odds_raw,
            CASE 
                WHEN o.popularity IN ('   ', '---', '***') THEN 0
                ELSE CAST(o.popularity AS INTEGER)
            END as popularity
        FROM public.odds_4_umatan h
        INNER JOIN latest_odds l ON 
            h.kaisai_year = l.kaisai_year AND
            h.kaisai_month_day = l.kaisai_month_day AND
            h.track_code = l.track_code AND
            h.kaisai_kai = l.kaisai_kai AND
            h.kaisai_day = l.kaisai_day AND
            h.race_number = l.race_number AND
            h.creation_date || h.announce_month_day_hour_minute = l.max_timestamp
        INNER JOIN public.odds_4_umatan_umatan_odds o ON
            h.kaisai_year = o.kaisai_year AND
            h.kaisai_month_day = o.kaisai_month_day AND
            h.track_code = o.track_code AND
            h.kaisai_kai = o.kaisai_kai AND
            h.kaisai_day = o.kaisai_day AND
            h.race_number = o.race_number AND
            h.announce_month_day_hour_minute = o.announce_month_day_hour_minute
        WHERE LENGTH(TRIM(o.combination)) = 4 AND o.combination != '    '
        ORDER BY race_id, horse_number, horse_number_2
        """,
        
        'sanrenfuku': f"""
        WITH latest_odds AS (
            SELECT 
                kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number,
                MAX(creation_date || announce_month_day_hour_minute) as max_timestamp
            FROM public.odds_5_sanrenpuku
            WHERE {date_filter}
            GROUP BY kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number
        )
        SELECT 
            h.kaisai_year || h.kaisai_month_day || h.track_code || h.kaisai_kai || h.kaisai_day || h.race_number as race_id,
            CASE 
                WHEN SUBSTRING(o.combination, 1, 2) = '  ' THEN NULL
                ELSE CAST(SUBSTRING(o.combination, 1, 2) AS INTEGER)
            END as horse_number,
            CASE 
                WHEN SUBSTRING(o.combination, 3, 2) = '  ' THEN NULL
                ELSE CAST(SUBSTRING(o.combination, 3, 2) AS INTEGER)
            END as horse_number_2,
            CASE 
                WHEN SUBSTRING(o.combination, 5, 2) = '  ' THEN NULL
                ELSE CAST(SUBSTRING(o.combination, 5, 2) AS INTEGER)
            END as horse_number_3,
            CASE 
                WHEN o.odds IN ('000000', '------', '******', '      ') THEN 0
                ELSE CAST(o.odds AS INTEGER)
            END as odds_raw,
            CASE 
                WHEN o.popularity IN ('   ', '---', '***') THEN 0
                ELSE CAST(o.popularity AS INTEGER)
            END as popularity
        FROM public.odds_5_sanrenpuku h
        INNER JOIN latest_odds l ON 
            h.kaisai_year = l.kaisai_year AND
            h.kaisai_month_day = l.kaisai_month_day AND
            h.track_code = l.track_code AND
            h.kaisai_kai = l.kaisai_kai AND
            h.kaisai_day = l.kaisai_day AND
            h.race_number = l.race_number AND
            h.creation_date || h.announce_month_day_hour_minute = l.max_timestamp
        INNER JOIN public.odds_5_sanrenpuku_sanrenpuku_odds o ON
            h.kaisai_year = o.kaisai_year AND
            h.kaisai_month_day = o.kaisai_month_day AND
            h.track_code = o.track_code AND
            h.kaisai_kai = o.kaisai_kai AND
            h.kaisai_day = o.kaisai_day AND
            h.race_number = o.race_number AND
            h.announce_month_day_hour_minute = o.announce_month_day_hour_minute
        WHERE LENGTH(TRIM(o.combination)) = 6 AND o.combination != '      '
        ORDER BY race_id, horse_number, horse_number_2, horse_number_3
        """,
        
        'sanrentan': f"""
        WITH latest_odds AS (
            SELECT 
                kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number,
                MAX(creation_date || announce_month_day_hour_minute) as max_timestamp
            FROM public.odds_6_sanrentan
            WHERE {date_filter}
            GROUP BY kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number
        )
        SELECT 
            h.kaisai_year || h.kaisai_month_day || h.track_code || h.kaisai_kai || h.kaisai_day || h.race_number as race_id,
            CASE 
                WHEN SUBSTRING(o.combination, 1, 2) = '  ' THEN NULL
                ELSE CAST(SUBSTRING(o.combination, 1, 2) AS INTEGER)
            END as horse_number,
            CASE 
                WHEN SUBSTRING(o.combination, 3, 2) = '  ' THEN NULL
                ELSE CAST(SUBSTRING(o.combination, 3, 2) AS INTEGER)
            END as horse_number_2,
            CASE 
                WHEN SUBSTRING(o.combination, 5, 2) = '  ' THEN NULL
                ELSE CAST(SUBSTRING(o.combination, 5, 2) AS INTEGER)
            END as horse_number_3,
            CASE 
                WHEN o.odds IN ('0000000', '-------', '*******', '       ') THEN 0
                ELSE CAST(o.odds AS INTEGER)
            END as odds_raw,
            CASE 
                WHEN o.popularity IN ('    ', '----', '****') THEN 0
                ELSE CAST(o.popularity AS INTEGER)
            END as popularity
        FROM public.odds_6_sanrentan h
        INNER JOIN latest_odds l ON 
            h.kaisai_year = l.kaisai_year AND
            h.kaisai_month_day = l.kaisai_month_day AND
            h.track_code = l.track_code AND
            h.kaisai_kai = l.kaisai_kai AND
            h.kaisai_day = l.kaisai_day AND
            h.race_number = l.race_number AND
            h.creation_date || h.announce_month_day_hour_minute = l.max_timestamp
        INNER JOIN public.odds_6_sanrentan_sanrentan_odds o ON
            h.kaisai_year = o.kaisai_year AND
            h.kaisai_month_day = o.kaisai_month_day AND
            h.track_code = o.track_code AND
            h.kaisai_kai = o.kaisai_kai AND
            h.kaisai_day = o.kaisai_day AND
            h.race_number = o.race_number AND
            h.announce_month_day_hour_minute = o.announce_month_day_hour_minute
        WHERE LENGTH(TRIM(o.combination)) = 6 AND o.combination != '      '
        ORDER BY race_id, horse_number, horse_number_2, horse_number_3
        """
    }
    
    # 各馬券種のデータを順次取得
    for bet_type, query in queries.items():
        try:
            logger.info(f"Fetching {bet_type} odds data...")
            df = pd.read_sql_query(query, engine)
            
            if not df.empty:
                # データの後処理
                # オッズを10で割って浮動小数点に変換、0の場合はNaNに
                df['odds'] = df['odds_raw'].astype(np.float32) / 10.0
                df.loc[df['odds_raw'] == 0, 'odds'] = np.nan
                
                # 型変換
                df['horse_number'] = df['horse_number'].astype('Int64')  # pandas nullable integer
                df['horse_number_2'] = df['horse_number_2'].astype('Int64')
                df['horse_number_3'] = df['horse_number_3'].astype('Int64')
                df['popularity'] = df['popularity'].astype(np.int64)
                
                # 不要な列を削除
                df = df.drop('odds_raw', axis=1)
                
                odds_data[bet_type] = df
                logger.info(f"Fetched {len(df)} {bet_type} odds records")
            else:
                logger.warning(f"No {bet_type} odds data found")
                
        except Exception as e:
            logger.error(f"Error fetching {bet_type} odds data: {e}")
            continue
    
    total_records = sum(len(df) for df in odds_data.values())
    logger.info(f"Total odds data fetched: {total_records} records across {len(odds_data)} bet types")
    return odds_data


def get_odds_summary(odds_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    オッズデータの概要を取得する
    
    Args:
        odds_data: fetch_odds_dataの戻り値
        
    Returns:
        各馬券種の統計情報を含むDataFrame
    """
    summary_data = []
    
    for bet_type, df in odds_data.items():
        if not df.empty:
            summary = {
                'bet_type': bet_type,
                'total_records': len(df),
                'unique_races': df['race_id'].nunique(),
                'avg_odds': df['odds'].mean(),
                'min_odds': df['odds'].min(),
                'max_odds': df['odds'].max(),
                'nan_odds_count': df['odds'].isna().sum(),
                'zero_popularity_count': (df['popularity'] == 0).sum()
            }
            summary_data.append(summary)
    
    return pd.DataFrame(summary_data)