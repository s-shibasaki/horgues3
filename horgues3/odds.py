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


def restructure_odds_data(odds_data: Dict[str, pd.DataFrame], 
                         race_ids: List[str], 
                         max_horses: int = 18) -> Dict[str, np.ndarray]:
    """
    オッズデータを馬券確率と同じ (num_races, num_combinations) 形状に再構成する
    
    Args:
        odds_data: 各馬券種のオッズデータを含む辞書
        race_ids: 対象レースIDのリスト
        max_horses: 最大馬数 (デフォルト: 18)
        
    Returns:
        各馬券種のオッズを格納した辞書 (num_races, num_combinations)
    """
    from .betting import get_betting_combinations
    
    logger.info(f"Converting odds data for {len(race_ids)} races")
    
    num_races = len(race_ids)
    combinations = get_betting_combinations()
    
    # レースIDとインデックスのマッピング（一度だけ作成）
    race_id_to_idx = {race_id: idx for idx, race_id in enumerate(race_ids)}
    race_ids_set = set(race_ids)  # O(1)検索用
    
    # 結果を格納する辞書
    odds_arrays = {}
    
    # 各馬券種の組み合わせ数を計算
    combination_counts = {
        'tansho': max_horses,
        'fukusho': max_horses,
        'umaren': max_horses * (max_horses - 1) // 2,
        'umatan': max_horses * (max_horses - 1),
        'wide': max_horses * (max_horses - 1) // 2,
        'sanrenfuku': max_horses * (max_horses - 1) * (max_horses - 2) // 6,
        'sanrentan': max_horses * (max_horses - 1) * (max_horses - 2)
    }
    
    # 組み合わせインデックスマッピングを事前計算
    combo_mappings = {}
    for bet_type in ['umaren', 'umatan', 'wide', 'sanrenfuku', 'sanrentan']:
        if bet_type in combinations:
            combo_mappings[bet_type] = {combo: idx for idx, combo in enumerate(combinations[bet_type])}
    
    for bet_type in ['tansho', 'fukusho', 'umaren', 'umatan', 'wide', 'sanrenfuku', 'sanrentan']:
        if bet_type not in odds_data or odds_data[bet_type].empty:
            logger.warning(f"No odds data for {bet_type}")
            continue
            
        df = odds_data[bet_type].copy()
        num_combinations = combination_counts[bet_type]
        
        # 対象レースのみにフィルタリング
        df = df[df['race_id'].isin(race_ids_set)]
        if df.empty:
            logger.warning(f"No odds data for {bet_type} in specified races")
            continue
        
        # レースインデックスを事前計算
        df['race_idx'] = df['race_id'].map(race_id_to_idx)
        
        # 結果配列を初期化
        odds_array = np.full((num_races, num_combinations), np.nan, dtype=np.float32)
        
        if bet_type in ['tansho', 'fukusho']:
            # 単勝・複勝: ベクトル化処理
            valid_mask = (
                df['horse_number'].notna() & 
                (df['horse_number'] >= 1) & 
                (df['horse_number'] <= max_horses) &
                df['odds'].notna()
            )
            valid_df = df[valid_mask].copy()
            
            if not valid_df.empty:
                race_indices = valid_df['race_idx'].values
                combo_indices = (valid_df['horse_number'] - 1).values
                odds_values = valid_df['odds'].values
                
                odds_array[race_indices, combo_indices] = odds_values
                
        elif bet_type in ['umaren', 'wide']:
            # 馬連・ワイド: バッチ処理
            valid_mask = (
                df['horse_number'].notna() & 
                df['horse_number_2'].notna() &
                (df['horse_number'] >= 1) & (df['horse_number'] <= max_horses) &
                (df['horse_number_2'] >= 1) & (df['horse_number_2'] <= max_horses) &
                df['odds'].notna()
            )
            valid_df = df[valid_mask].copy()
            
            if not valid_df.empty:
                # 組み合わせを正規化（小さい番号を先に）
                h1 = np.minimum(valid_df['horse_number'] - 1, valid_df['horse_number_2'] - 1)
                h2 = np.maximum(valid_df['horse_number'] - 1, valid_df['horse_number_2'] - 1)
                
                # 組み合わせインデックスを計算
                combo_indices = []
                combo_to_idx = combo_mappings[bet_type]
                
                for i in range(len(valid_df)):
                    combo = (int(h1.iloc[i]), int(h2.iloc[i]))
                    if combo in combo_to_idx:
                        combo_indices.append(combo_to_idx[combo])
                    else:
                        combo_indices.append(-1)  # 無効な組み合わせ
                
                combo_indices = np.array(combo_indices)
                valid_combo_mask = combo_indices >= 0
                
                if valid_combo_mask.any():
                    race_indices = valid_df['race_idx'].values[valid_combo_mask]
                    combo_indices = combo_indices[valid_combo_mask]
                    odds_values = valid_df['odds'].values[valid_combo_mask]
                    
                    odds_array[race_indices, combo_indices] = odds_values
                    
        elif bet_type == 'umatan':
            # 馬単: バッチ処理
            valid_mask = (
                df['horse_number'].notna() & 
                df['horse_number_2'].notna() &
                (df['horse_number'] >= 1) & (df['horse_number'] <= max_horses) &
                (df['horse_number_2'] >= 1) & (df['horse_number_2'] <= max_horses) &
                df['odds'].notna()
            )
            valid_df = df[valid_mask].copy()
            
            if not valid_df.empty:
                combo_indices = []
                combo_to_idx = combo_mappings[bet_type]
                
                for _, row in valid_df.iterrows():
                    combo = (int(row['horse_number'] - 1), int(row['horse_number_2'] - 1))
                    if combo in combo_to_idx:
                        combo_indices.append(combo_to_idx[combo])
                    else:
                        combo_indices.append(-1)
                
                combo_indices = np.array(combo_indices)
                valid_combo_mask = combo_indices >= 0
                
                if valid_combo_mask.any():
                    race_indices = valid_df['race_idx'].values[valid_combo_mask]
                    combo_indices = combo_indices[valid_combo_mask]
                    odds_values = valid_df['odds'].values[valid_combo_mask]
                    
                    odds_array[race_indices, combo_indices] = odds_values
                    
        elif bet_type in ['sanrenfuku', 'sanrentan']:
            # 3連系: バッチ処理
            valid_mask = (
                df['horse_number'].notna() & 
                df['horse_number_2'].notna() &
                df['horse_number_3'].notna() &
                (df['horse_number'] >= 1) & (df['horse_number'] <= max_horses) &
                (df['horse_number_2'] >= 1) & (df['horse_number_2'] <= max_horses) &
                (df['horse_number_3'] >= 1) & (df['horse_number_3'] <= max_horses) &
                df['odds'].notna()
            )
            valid_df = df[valid_mask].copy()
            
            if not valid_df.empty:
                combo_indices = []
                combo_to_idx = combo_mappings[bet_type]
                
                if bet_type == 'sanrenfuku':
                    # 3連複: ソートして順不同にする
                    for _, row in valid_df.iterrows():
                        horses = sorted([int(row['horse_number'] - 1), 
                                       int(row['horse_number_2'] - 1), 
                                       int(row['horse_number_3'] - 1)])
                        combo = tuple(horses)
                        if combo in combo_to_idx:
                            combo_indices.append(combo_to_idx[combo])
                        else:
                            combo_indices.append(-1)
                else:
                    # 3連単: 順序そのまま
                    for _, row in valid_df.iterrows():
                        combo = (int(row['horse_number'] - 1), 
                               int(row['horse_number_2'] - 1), 
                               int(row['horse_number_3'] - 1))
                        if combo in combo_to_idx:
                            combo_indices.append(combo_to_idx[combo])
                        else:
                            combo_indices.append(-1)
                
                combo_indices = np.array(combo_indices)
                valid_combo_mask = combo_indices >= 0
                
                if valid_combo_mask.any():
                    race_indices = valid_df['race_idx'].values[valid_combo_mask]
                    combo_indices = combo_indices[valid_combo_mask]
                    odds_values = valid_df['odds'].values[valid_combo_mask]
                    
                    odds_array[race_indices, combo_indices] = odds_values
        
        odds_arrays[bet_type] = odds_array
        
        # 統計情報をログ出力
        valid_odds = ~np.isnan(odds_array)
        total_combinations = odds_array.size
        valid_combinations = valid_odds.sum()
        logger.info(f"{bet_type}: {valid_combinations}/{total_combinations} combinations have valid odds")
    
    logger.info(f"Converted odds data for {len(odds_arrays)} bet types")
    return odds_arrays


def calculate_implied_probabilities(odds_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    オッズから暗示確率を計算し、各レースで合計が1になるように正規化する
    
    Args:
        odds_arrays: restructure_odds_dataの戻り値
        
    Returns:
        各馬券種の暗示確率を格納した辞書 (num_races, num_combinations)
    """
    logger.info("Calculating implied probabilities from odds")
    
    implied_probs = {}
    
    for bet_type, odds_array in odds_arrays.items():
        # オッズから確率を計算 (1/オッズ)
        # 0やNaNのオッズは0確率とする
        with np.errstate(divide='ignore', invalid='ignore'):
            prob_array = 1.0 / odds_array
            prob_array = np.where(np.isfinite(prob_array), prob_array, 0.0)
        
        # 各レースで正規化（合計を1にする）
        normalized_probs = np.zeros_like(prob_array)
        
        for race_idx in range(prob_array.shape[0]):
            race_probs = prob_array[race_idx]
            prob_sum = race_probs.sum()
            
            if prob_sum > 0:
                # 確率の合計が0より大きい場合のみ正規化
                normalized_probs[race_idx] = race_probs / prob_sum
            else:
                # 全ての確率が0の場合はそのまま0を設定
                normalized_probs[race_idx] = race_probs
        
        implied_probs[bet_type] = normalized_probs.astype(np.float32)
        
        # 統計情報をログ出力
        non_zero_probs = (normalized_probs > 0).sum()
        total_combinations = normalized_probs.size
        avg_prob = normalized_probs[normalized_probs > 0].mean() if non_zero_probs > 0 else 0
        
        # 正規化後の各レースの確率合計をチェック
        race_sums = normalized_probs.sum(axis=1)
        valid_races = (race_sums > 0).sum()
        avg_sum = race_sums[race_sums > 0].mean() if valid_races > 0 else 0
        
        logger.info(f"{bet_type}: {non_zero_probs}/{total_combinations} combinations have non-zero probability")
        logger.info(f"{bet_type}: {valid_races} races with valid probabilities (avg sum: {avg_sum:.6f})")
    
    return implied_probs


def compare_probabilities_with_odds(predicted_probs: Dict[str, np.ndarray], 
                                  odds_arrays: Dict[str, np.ndarray], 
                                  race_ids: List[str],
                                  top_n: int = 5) -> str:
    """
    予測確率とオッズ（暗示確率）を比較する
    
    Args:
        predicted_probs: 予測された確率 (betting.pyのcalculate_betting_probabilitiesの戻り値)
        odds_arrays: オッズ配列 (restructure_odds_dataの戻り値)
        race_ids: レースIDのリスト
        top_n: 上位何位まで表示するか
        
    Returns:
        比較結果の文字列
    """
    from .betting import get_betting_combinations
    
    logger.info(f"Comparing predicted probabilities with odds for {len(race_ids)} races")
    
    combinations = get_betting_combinations()
    implied_probs = calculate_implied_probabilities(odds_arrays)
    
    results = []
    results.append("=== PROBABILITY vs ODDS COMPARISON ===\n")
    
    for race_idx, race_id in enumerate(race_ids):
        results.append(f"Race ID: {race_id}")
        results.append("-" * 50)
        
        for bet_type in ['tansho', 'fukusho', 'umaren', 'umatan', 'wide', 'sanrenfuku', 'sanrentan']:
            if bet_type not in predicted_probs or bet_type not in odds_arrays:
                continue
                
            pred_probs = predicted_probs[bet_type][race_idx]
            odds_vals = odds_arrays[bet_type][race_idx]
            impl_probs = implied_probs[bet_type][race_idx]
            combos = combinations[bet_type]
            
            # 有効な組み合わせ（予測確率>0かつオッズが存在）を抽出
            valid_mask = (pred_probs > 0) & ~np.isnan(odds_vals)
            if not valid_mask.any():
                continue
                
            valid_indices = np.where(valid_mask)[0]
            
            # 予測確率でソート（降順）
            sorted_pred_indices = valid_indices[np.argsort(pred_probs[valid_indices])[::-1]]
            
            results.append(f"\n{bet_type.upper()}:")
            results.append("Rank | Combination | Pred.Prob | Odds | Impl.Prob | Ratio")
            results.append("-" * 65)
            
            for rank, idx in enumerate(sorted_pred_indices[:top_n], 1):
                pred_prob = pred_probs[idx]
                odds_val = odds_vals[idx]
                impl_prob = impl_probs[idx]
                ratio = pred_prob / impl_prob if impl_prob > 0 else float('inf')
                
                # 組み合わせの表示形式を作成
                if bet_type in ['tansho', 'fukusho']:
                    combo_str = f"{combos[idx] + 1:2d}"
                elif bet_type in ['umaren', 'umatan', 'wide']:
                    h1, h2 = combos[idx]
                    combo_str = f"{h1+1:2d}-{h2+1:2d}"
                else:  # 3連系
                    h1, h2, h3 = combos[idx]
                    combo_str = f"{h1+1:2d}-{h2+1:2d}-{h3+1:2d}"
                
                results.append(f"{rank:4d} | {combo_str:11s} | {pred_prob:9.6f} | {odds_val:4.1f} | {impl_prob:9.6f} | {ratio:5.2f}")
        
        results.append("\n")
    
    return '\n'.join(results)