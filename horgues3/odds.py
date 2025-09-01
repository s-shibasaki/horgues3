import pandas as pd
import numpy as np
from horgues3.database import create_database_engine
import logging
from typing import Dict, Optional
from itertools import combinations, permutations

def get_odds_dataframes(start_date: str, end_date: str, num_horses: int = 18) -> Dict[str, pd.DataFrame]:
    """
    競馬オッズテーブルからオッズデータを取得してDataFrameに変換する
    
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
        各賭式のオッズDataFrame辞書
    """
    logger = logging.getLogger(__name__)
    engine = create_database_engine()
    
    # 日付条件の作成
    start_year = start_date[:4]
    start_monthday = start_date[4:]
    end_year = end_date[:4]
    end_monthday = end_date[4:]
    date_condition = f"""
    AND (odds_{{table_num}}_kaisai_year > '{start_year}' OR (odds_{{table_num}}_kaisai_year = '{start_year}' AND odds_{{table_num}}_kaisai_monthday >= '{start_monthday}'))
    AND (odds_{{table_num}}_kaisai_year < '{end_year}' OR (odds_{{table_num}}_kaisai_year = '{end_year}' AND odds_{{table_num}}_kaisai_monthday <= '{end_monthday}'))
    """
    
    results = {}
    
    # 1. 単勝オッズ
    logger.info("単勝オッズを取得中...")
    tansho_query = f"""
    SELECT 
        odds_1_kaisai_year || odds_1_kaisai_monthday || odds_1_keibajo_code || 
        odds_1_kaisai_kai || odds_1_kaisai_nichime || odds_1_race_number as race_id,
        umaban,
        odds
    FROM public.odds_1_tansho_odds 
    WHERE umaban != '  '
    AND odds NOT IN ('0000', '****', '----', '    ')
    {date_condition.format(table_num='1')}
    """
    
    tansho_df = pd.read_sql(tansho_query, engine)
    if not tansho_df.empty:
        tansho_df['odds'] = (pd.to_numeric(tansho_df['odds'], errors='coerce') / 10.0).astype(np.float32)
        tansho_pivot = tansho_df.pivot(index='race_id', columns='umaban', values='odds')
        # カラム名を2桁の馬番に統一
        tansho_pivot.columns = [f'{int(col):02d}' for col in tansho_pivot.columns]
        # 全馬番のカラムを作成
        all_horses = [f'{i:02d}' for i in range(1, num_horses + 1)]
        results['tansho'] = tansho_pivot.reindex(columns=all_horses).fillna(0.0).astype(np.float32)
    
    # 2. 複勝オッズ
    logger.info("複勝オッズを取得中...")
    fukusho_query = f"""
    SELECT 
        odds_1_kaisai_year || odds_1_kaisai_monthday || odds_1_keibajo_code || 
        odds_1_kaisai_kai || odds_1_kaisai_nichime || odds_1_race_number as race_id,
        umaban,
        min_odds,
        max_odds
    FROM public.odds_1_fukusho_odds 
    WHERE umaban != '  '
    AND min_odds NOT IN ('0000', '****', '----', '    ')
    AND max_odds NOT IN ('0000', '****', '----', '    ')
    {date_condition.format(table_num='1')}
    """
    
    fukusho_df = pd.read_sql(fukusho_query, engine)
    if not fukusho_df.empty:
        fukusho_df['min_odds'] = (pd.to_numeric(fukusho_df['min_odds'], errors='coerce') / 10.0).astype(np.float32)
        fukusho_df['max_odds'] = (pd.to_numeric(fukusho_df['max_odds'], errors='coerce') / 10.0).astype(np.float32)
        fukusho_df['avg_odds'] = ((fukusho_df['min_odds'] + fukusho_df['max_odds']) / 2.0).astype(np.float32)
        fukusho_pivot = fukusho_df.pivot(index='race_id', columns='umaban', values='min_odds')
        fukusho_pivot.columns = [f'{int(col):02d}' for col in fukusho_pivot.columns]
        all_horses = [f'{i:02d}' for i in range(1, num_horses + 1)]
        results['fukusho'] = fukusho_pivot.reindex(columns=all_horses).fillna(0.0).astype(np.float32)
    
    # 3. 馬連オッズ
    logger.info("馬連オッズを取得中...")
    umaren_query = f"""
    SELECT 
        odds_2_kaisai_year || odds_2_kaisai_monthday || odds_2_keibajo_code || 
        odds_2_kaisai_kai || odds_2_kaisai_nichime || odds_2_race_number as race_id,
        kumiban,
        odds
    FROM public.odds_2_umaren_odds 
    WHERE kumiban != '    '
    AND odds NOT IN ('000000', '******', '------', '      ')
    {date_condition.format(table_num='2')}
    """
    
    umaren_df = pd.read_sql(umaren_query, engine)
    if not umaren_df.empty:
        umaren_df['odds'] = (pd.to_numeric(umaren_df['odds'], errors='coerce') / 10.0).astype(np.float32)
        # pivotしてからカラム名を変更
        umaren_pivot = umaren_df.pivot(index='race_id', columns='kumiban', values='odds')
        # カラム名を '01-02' 形式に変更
        umaren_pivot.columns = [f"{col[:2]}-{col[2:]}" if len(col) == 4 else col 
                               for col in umaren_pivot.columns]
        # 全組み合わせのカラムを作成
        all_combinations = [f'{i:02d}-{j:02d}' for i, j in combinations(range(1, num_horses + 1), 2)]
        results['umaren'] = umaren_pivot.reindex(columns=all_combinations).fillna(0.0).astype(np.float32)
    
    # 4. ワイドオッズ
    logger.info("ワイドオッズを取得中...")
    wide_query = f"""
    SELECT 
        odds_3_kaisai_year || odds_3_kaisai_monthday || odds_3_keibajo_code || 
        odds_3_kaisai_kai || odds_3_kaisai_nichime || odds_3_race_number as race_id,
        kumiban,
        min_odds,
        max_odds
    FROM public.odds_3_wide_odds 
    WHERE kumiban != '    '
    AND min_odds NOT IN ('00000', '*****', '-----', '     ')
    AND max_odds NOT IN ('00000', '*****', '-----', '     ')
    {date_condition.format(table_num='3')}
    """
    
    wide_df = pd.read_sql(wide_query, engine)
    if not wide_df.empty:
        wide_df['min_odds'] = (pd.to_numeric(wide_df['min_odds'], errors='coerce') / 10.0).astype(np.float32)
        wide_df['max_odds'] = (pd.to_numeric(wide_df['max_odds'], errors='coerce') / 10.0).astype(np.float32)
        wide_df['avg_odds'] = ((wide_df['min_odds'] + wide_df['max_odds']) / 2.0).astype(np.float32)
        # pivotしてからカラム名を変更
        wide_pivot = wide_df.pivot(index='race_id', columns='kumiban', values='min_odds')
        # カラム名を '01-02' 形式に変更
        wide_pivot.columns = [f"{col[:2]}-{col[2:]}" if len(col) == 4 else col 
                             for col in wide_pivot.columns]
        all_combinations = [f'{i:02d}-{j:02d}' for i, j in combinations(range(1, num_horses + 1), 2)]
        results['wide'] = wide_pivot.reindex(columns=all_combinations).fillna(0.0).astype(np.float32)
    
    # 5. 馬単オッズ
    logger.info("馬単オッズを取得中...")
    umatan_query = f"""
    SELECT 
        odds_4_kaisai_year || odds_4_kaisai_monthday || odds_4_keibajo_code || 
        odds_4_kaisai_kai || odds_4_kaisai_nichime || odds_4_race_number as race_id,
        kumiban,
        odds
    FROM public.odds_4_umatan_odds 
    WHERE kumiban != '    '
    AND odds NOT IN ('000000', '******', '------', '      ')
    {date_condition.format(table_num='4')}
    """
    
    umatan_df = pd.read_sql(umatan_query, engine)
    if not umatan_df.empty:
        umatan_df['odds'] = (pd.to_numeric(umatan_df['odds'], errors='coerce') / 10.0).astype(np.float32)
        # pivotしてからカラム名を変更
        umatan_pivot = umatan_df.pivot(index='race_id', columns='kumiban', values='odds')
        # カラム名を '01-02' 形式に変更
        umatan_pivot.columns = [f"{col[:2]}-{col[2:]}" if len(col) == 4 else col 
                               for col in umatan_pivot.columns]
        # 順列の全組み合わせを作成
        all_permutations = [f'{i:02d}-{j:02d}' for i, j in permutations(range(1, num_horses + 1), 2)]
        results['umatan'] = umatan_pivot.reindex(columns=all_permutations).fillna(0.0).astype(np.float32)
    
    # 6. 三連複オッズ
    logger.info("三連複オッズを取得中...")
    sanrenpuku_query = f"""
    SELECT 
        odds_5_kaisai_year || odds_5_kaisai_monthday || odds_5_keibajo_code || 
        odds_5_kaisai_kai || odds_5_kaisai_nichime || odds_5_race_number as race_id,
        kumiban,
        odds
    FROM public.odds_5_sanrenpuku_odds 
    WHERE kumiban != '      '
    AND odds NOT IN ('000000', '******', '------', '      ')
    {date_condition.format(table_num='5')}
    """
    
    sanrenpuku_df = pd.read_sql(sanrenpuku_query, engine)
    if not sanrenpuku_df.empty:
        sanrenpuku_df['odds'] = (pd.to_numeric(sanrenpuku_df['odds'], errors='coerce') / 10.0).astype(np.float32)
        # pivotしてからカラム名を変更
        sanrenpuku_pivot = sanrenpuku_df.pivot(index='race_id', columns='kumiban', values='odds')
        # カラム名を '01-02-03' 形式に変更
        sanrenpuku_pivot.columns = [f"{col[:2]}-{col[2:4]}-{col[4:]}" if len(col) == 6 else col 
                                   for col in sanrenpuku_pivot.columns]
        # 3つの組み合わせを作成
        all_combinations_3 = [f'{i:02d}-{j:02d}-{k:02d}' for i, j, k in combinations(range(1, num_horses + 1), 3)]
        results['sanrenpuku'] = sanrenpuku_pivot.reindex(columns=all_combinations_3).fillna(0.0).astype(np.float32)
    
    # 7. 三連単オッズ
    logger.info("三連単オッズを取得中...")
    sanrentan_query = f"""
    SELECT 
        odds_6_kaisai_year || odds_6_kaisai_monthday || odds_6_keibajo_code || 
        odds_6_kaisai_kai || odds_6_kaisai_nichime || odds_6_race_number as race_id,
        kumiban,
        odds
    FROM public.odds_6_sanrentan_odds 
    WHERE kumiban != '      '
    AND odds NOT IN ('0000000', '*******', '-------', '       ')
    {date_condition.format(table_num='6')}
    """
    
    sanrentan_df = pd.read_sql(sanrentan_query, engine)
    if not sanrentan_df.empty:
        sanrentan_df['odds'] = (pd.to_numeric(sanrentan_df['odds'], errors='coerce') / 10.0).astype(np.float32)
        # pivotしてからカラム名を変更
        sanrentan_pivot = sanrentan_df.pivot(index='race_id', columns='kumiban', values='odds')
        # カラム名を '01-02-03' 形式に変更
        sanrentan_pivot.columns = [f"{col[:2]}-{col[2:4]}-{col[4:]}" if len(col) == 6 else col 
                                  for col in sanrentan_pivot.columns]
        # 3つの順列を作成
        all_permutations_3 = [f'{i:02d}-{j:02d}-{k:02d}' for i, j, k in permutations(range(1, num_horses + 1), 3)]
        results['sanrentan'] = sanrentan_pivot.reindex(columns=all_permutations_3).fillna(0.0).astype(np.float32)
    
    logger.info(f"データ取得完了: {len(results)}種類のオッズデータ")
    for bet_type, df in results.items():
        logger.info(f"{bet_type}: {df.shape[0]}レース, {df.shape[1]}カラム")
    
    return results

# 使用例
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # データ取得
    odds_data = get_odds_dataframes(
        start_date='20240101',
        end_date='20240131',
        num_horses=18
    )
    
    # 結果確認
    for bet_type, df in odds_data.items():
        print(f"\n{bet_type}オッズ:")
        print(f"Shape: {df.shape}")
        print(df.head())
