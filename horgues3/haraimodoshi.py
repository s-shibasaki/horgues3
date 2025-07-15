import pandas as pd
import numpy as np
from horgues3.database import create_database_engine
import logging
from typing import Dict, Optional
from itertools import combinations, permutations

def get_haraimodoshi_dataframes(start_date: str, end_date: str, num_horses: int = 18) -> Dict[str, pd.DataFrame]:
    """
    競馬払戻テーブルから払戻データを取得してDataFrameに変換する
    
    Parameters:
    -----------
    start_date : str
        開始日 (YYYYMMDD形式)
    end_date : str  
        終了日 (YYYYMMDD形式)
    num_horses : int, default=18
        馬数 (デフォルト18頭)
    
    Returns:
    --------
    Dict[str, pd.DataFrame]
        各賭式の払戻DataFrame辞書
    """
    logger = logging.getLogger(__name__)
    engine = create_database_engine()
    
    # 日付条件の作成
    start_year = start_date[:4]
    start_monthday = start_date[4:]
    end_year = end_date[:4]
    end_monthday = end_date[4:]
    date_condition = """
    AND (haraimodoshi_joho_kaisai_year > '{start_year}' OR (haraimodoshi_joho_kaisai_year = '{start_year}' AND haraimodoshi_joho_kaisai_monthday >= '{start_monthday}'))
    AND (haraimodoshi_joho_kaisai_year < '{end_year}' OR (haraimodoshi_joho_kaisai_year = '{end_year}' AND haraimodoshi_joho_kaisai_monthday <= '{end_monthday}'))
    """.format(start_year=start_year, start_monthday=start_monthday, end_year=end_year, end_monthday=end_monthday)
    
    results = {}
    
    # 1. 単勝払戻
    logger.info("単勝払戻を取得中...")
    tansho_query = f"""
    SELECT 
        haraimodoshi_joho_kaisai_year || haraimodoshi_joho_kaisai_monthday || haraimodoshi_joho_keibajo_code || 
        haraimodoshi_joho_kaisai_kai || haraimodoshi_joho_kaisai_nichime || haraimodoshi_joho_race_number as race_id,
        umaban,
        haraimodoshi_kin
    FROM public.haraimodoshi_joho_tansho_haraimodoshi 
    WHERE umaban != '  '
    AND haraimodoshi_kin NOT IN ('000000000', '*********', '---------', '         ')
    {date_condition}
    """
    
    tansho_df = pd.read_sql(tansho_query, engine)
    if not tansho_df.empty:
        tansho_df['haraimodoshi_kin'] = pd.to_numeric(tansho_df['haraimodoshi_kin'], errors='coerce').astype(np.float32)
        
        # 払戻金額のpivot
        tansho_haraimodoshi_pivot = tansho_df.pivot(index='race_id', columns='umaban', values='haraimodoshi_kin')
        tansho_haraimodoshi_pivot.columns = [f'{int(col):02d}' for col in tansho_haraimodoshi_pivot.columns if pd.notna(col)]
        all_horses = [f'{i:02d}' for i in range(1, num_horses + 1)]
        results['tansho'] = tansho_haraimodoshi_pivot.reindex(columns=all_horses).fillna(0.0).astype(np.float32)
    
    # 2. 複勝払戻
    logger.info("複勝払戻を取得中...")
    fukusho_query = f"""
    SELECT 
        haraimodoshi_joho_kaisai_year || haraimodoshi_joho_kaisai_monthday || haraimodoshi_joho_keibajo_code || 
        haraimodoshi_joho_kaisai_kai || haraimodoshi_joho_kaisai_nichime || haraimodoshi_joho_race_number as race_id,
        umaban,
        haraimodoshi_kin
    FROM public.haraimodoshi_joho_fukusho_haraimodoshi 
    WHERE umaban != '  '
    AND haraimodoshi_kin NOT IN ('000000000', '*********', '---------', '         ')
    {date_condition}
    """
    
    fukusho_df = pd.read_sql(fukusho_query, engine)
    if not fukusho_df.empty:
        fukusho_df['haraimodoshi_kin'] = pd.to_numeric(fukusho_df['haraimodoshi_kin'], errors='coerce').astype(np.float32)
        
        fukusho_haraimodoshi_pivot = fukusho_df.pivot(index='race_id', columns='umaban', values='haraimodoshi_kin')
        fukusho_haraimodoshi_pivot.columns = [f'{int(col):02d}' for col in fukusho_haraimodoshi_pivot.columns if pd.notna(col)]
        all_horses = [f'{i:02d}' for i in range(1, num_horses + 1)]
        results['fukusho'] = fukusho_haraimodoshi_pivot.reindex(columns=all_horses).fillna(0.0).astype(np.float32)
    
    # 3. 馬連払戻
    logger.info("馬連払戻を取得中...")
    umaren_query = f"""
    SELECT 
        haraimodoshi_joho_kaisai_year || haraimodoshi_joho_kaisai_monthday || haraimodoshi_joho_keibajo_code || 
        haraimodoshi_joho_kaisai_kai || haraimodoshi_joho_kaisai_nichime || haraimodoshi_joho_race_number as race_id,
        kumiban,
        haraimodoshi_kin
    FROM public.haraimodoshi_joho_umaren_haraimodoshi 
    WHERE kumiban != '    '
    AND haraimodoshi_kin NOT IN ('000000000', '*********', '---------', '         ')
    {date_condition}
    """
    
    umaren_df = pd.read_sql(umaren_query, engine)
    if not umaren_df.empty:
        umaren_df['haraimodoshi_kin'] = pd.to_numeric(umaren_df['haraimodoshi_kin'], errors='coerce').astype(np.float32)
        
        # カラム名を '01-02' 形式に変更
        umaren_df['kumiban_formatted'] = umaren_df['kumiban'].apply(
            lambda x: f"{x[:2]}-{x[2:]}" if len(x) == 4 else x
        )
        
        umaren_haraimodoshi_pivot = umaren_df.pivot(index='race_id', columns='kumiban_formatted', values='haraimodoshi_kin')
        all_combinations = [f'{i:02d}-{j:02d}' for i, j in combinations(range(1, num_horses + 1), 2)]
        results['umaren'] = umaren_haraimodoshi_pivot.reindex(columns=all_combinations).fillna(0.0).astype(np.float32)
    
    # 4. ワイド払戻
    logger.info("ワイド払戻を取得中...")
    wide_query = f"""
    SELECT 
        haraimodoshi_joho_kaisai_year || haraimodoshi_joho_kaisai_monthday || haraimodoshi_joho_keibajo_code || 
        haraimodoshi_joho_kaisai_kai || haraimodoshi_joho_kaisai_nichime || haraimodoshi_joho_race_number as race_id,
        kumiban,
        haraimodoshi_kin
    FROM public.haraimodoshi_joho_wide_haraimodoshi 
    WHERE kumiban != '    '
    AND haraimodoshi_kin NOT IN ('000000000', '*********', '---------', '         ')
    {date_condition}
    """
    
    wide_df = pd.read_sql(wide_query, engine)
    if not wide_df.empty:
        wide_df['haraimodoshi_kin'] = pd.to_numeric(wide_df['haraimodoshi_kin'], errors='coerce').astype(np.float32)
        
        wide_df['kumiban_formatted'] = wide_df['kumiban'].apply(
            lambda x: f"{x[:2]}-{x[2:]}" if len(x) == 4 else x
        )
        
        wide_haraimodoshi_pivot = wide_df.pivot(index='race_id', columns='kumiban_formatted', values='haraimodoshi_kin')
        all_combinations = [f'{i:02d}-{j:02d}' for i, j in combinations(range(1, num_horses + 1), 2)]
        results['wide'] = wide_haraimodoshi_pivot.reindex(columns=all_combinations).fillna(0.0).astype(np.float32)
    
    # 5. 馬単払戻
    logger.info("馬単払戻を取得中...")
    umatan_query = f"""
    SELECT 
        haraimodoshi_joho_kaisai_year || haraimodoshi_joho_kaisai_monthday || haraimodoshi_joho_keibajo_code || 
        haraimodoshi_joho_kaisai_kai || haraimodoshi_joho_kaisai_nichime || haraimodoshi_joho_race_number as race_id,
        kumiban,
        haraimodoshi_kin
    FROM public.haraimodoshi_joho_umatan_haraimodoshi 
    WHERE kumiban != '    '
    AND haraimodoshi_kin NOT IN ('000000000', '*********', '---------', '         ')
    {date_condition}
    """
    
    umatan_df = pd.read_sql(umatan_query, engine)
    if not umatan_df.empty:
        umatan_df['haraimodoshi_kin'] = pd.to_numeric(umatan_df['haraimodoshi_kin'], errors='coerce').astype(np.float32)
        
        umatan_df['kumiban_formatted'] = umatan_df['kumiban'].apply(
            lambda x: f"{x[:2]}-{x[2:]}" if len(x) == 4 else x
        )
        
        umatan_haraimodoshi_pivot = umatan_df.pivot(index='race_id', columns='kumiban_formatted', values='haraimodoshi_kin')
        all_permutations = [f'{i:02d}-{j:02d}' for i, j in permutations(range(1, num_horses + 1), 2)]
        results['umatan'] = umatan_haraimodoshi_pivot.reindex(columns=all_permutations).fillna(0.0).astype(np.float32)
    
    # 6. 三連複払戻
    logger.info("三連複払戻を取得中...")
    sanrenpuku_query = f"""
    SELECT 
        haraimodoshi_joho_kaisai_year || haraimodoshi_joho_kaisai_monthday || haraimodoshi_joho_keibajo_code || 
        haraimodoshi_joho_kaisai_kai || haraimodoshi_joho_kaisai_nichime || haraimodoshi_joho_race_number as race_id,
        kumiban,
        haraimodoshi_kin
    FROM public.haraimodoshi_joho_sanrenpuku_haraimodoshi 
    WHERE kumiban != '      '
    AND haraimodoshi_kin NOT IN ('000000000', '*********', '---------', '         ')
    {date_condition}
    """
    
    sanrenpuku_df = pd.read_sql(sanrenpuku_query, engine)
    if not sanrenpuku_df.empty:
        sanrenpuku_df['haraimodoshi_kin'] = pd.to_numeric(sanrenpuku_df['haraimodoshi_kin'], errors='coerce').astype(np.float32)
        
        sanrenpuku_df['kumiban_formatted'] = sanrenpuku_df['kumiban'].apply(
            lambda x: f"{x[:2]}-{x[2:4]}-{x[4:]}" if len(x) == 6 else x
        )
        
        sanrenpuku_haraimodoshi_pivot = sanrenpuku_df.pivot(index='race_id', columns='kumiban_formatted', values='haraimodoshi_kin')
        all_combinations_3 = [f'{i:02d}-{j:02d}-{k:02d}' for i, j, k in combinations(range(1, num_horses + 1), 3)]
        results['sanrenpuku'] = sanrenpuku_haraimodoshi_pivot.reindex(columns=all_combinations_3).fillna(0.0).astype(np.float32)
    
    # 7. 三連単払戻
    logger.info("三連単払戻を取得中...")
    sanrentan_query = f"""
    SELECT 
        haraimodoshi_joho_kaisai_year || haraimodoshi_joho_kaisai_monthday || haraimodoshi_joho_keibajo_code || 
        haraimodoshi_joho_kaisai_kai || haraimodoshi_joho_kaisai_nichime || haraimodoshi_joho_race_number as race_id,
        kumiban,
        haraimodoshi_kin
    FROM public.haraimodoshi_joho_sanrentan_haraimodoshi 
    WHERE kumiban != '      '
    AND haraimodoshi_kin NOT IN ('000000000', '*********', '---------', '         ')
    {date_condition}
    """
    
    sanrentan_df = pd.read_sql(sanrentan_query, engine)
    if not sanrentan_df.empty:
        sanrentan_df['haraimodoshi_kin'] = pd.to_numeric(sanrentan_df['haraimodoshi_kin'], errors='coerce').astype(np.float32)
        
        sanrentan_df['kumiban_formatted'] = sanrentan_df['kumiban'].apply(
            lambda x: f"{x[:2]}-{x[2:4]}-{x[4:]}" if len(x) == 6 else x
        )
        
        sanrentan_haraimodoshi_pivot = sanrentan_df.pivot(index='race_id', columns='kumiban_formatted', values='haraimodoshi_kin')
        all_permutations_3 = [f'{i:02d}-{j:02d}-{k:02d}' for i, j, k in permutations(range(1, num_horses + 1), 3)]
        results['sanrentan'] = sanrentan_haraimodoshi_pivot.reindex(columns=all_permutations_3).fillna(0.0).astype(np.float32)
    
    logger.info(f"データ取得完了: {len(results)}種類の払戻データ")
    for data_type, df in results.items():
        logger.info(f"{data_type}: {df.shape[0]}レース, {df.shape[1]}カラム")
    
    return results

# 使用例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # データ取得
    haraimodoshi_data = get_haraimodoshi_dataframes(
        start_date='20240101',
        end_date='20240131',
        num_horses=18
    )
    
    # 結果確認
    for data_type, df in haraimodoshi_data.items():
        print(f"\n{data_type}:")
        print(f"Shape: {df.shape}")
        print(df.head())