import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import logging
from sqlalchemy import create_engine
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class HorguesDataset(Dataset):

    def __init__(self, start_ymd: str, end_ymd: str, max_horses: int = 18, max_hist_len: int = 18, max_prev_days: int = 1000, hours_before_race: int = 2):
        """
        Args:
            start_ymd (str): 開始日 (YYYYMMDD形式)
            end_ymd (str): 終了日 (YYYYMMDD形式)
            max_horses (int): 最大馬数
            max_hist_len (int): 履歴の最大長
            max_prev_days (int): 取得する過去の日数
            hours_before_race (int): レース発走時刻の何時間前までの過去データを使用するか
        """
        super().__init__()

        self.start_date = datetime.strptime(start_ymd, "%Y%m%d").date()
        self.end_date = datetime.strptime(end_ymd, "%Y%m%d").date()

        self.max_horses = max_horses
        self.max_hist_len = max_hist_len
        self.max_prev_days = max_prev_days
        self.hours_before_race = hours_before_race

    def fetch(self):
        """ データベースからデータを取得する"""
        logger.info("Fetching data from database...")

        extended_start_date = self.start_date - timedelta(days=self.max_prev_days)

        start_year = extended_start_date.strftime("%Y")
        start_month_day = extended_start_date.strftime("%m%d")
        end_year = self.end_date.strftime("%Y")
        end_month_day = self.end_date.strftime("%m%d")
        
        engine = create_engine("postgresql://postgres:postgres@localhost/horgues3")

        query = f"""
        WITH pedigree_pivot AS (
            SELECT 
                blood_registration_number,
                MAX(CASE WHEN pedigree_index = 0 THEN breeding_registration_number END) as pedigree_0,
                MAX(CASE WHEN pedigree_index = 1 THEN breeding_registration_number END) as pedigree_1,
                MAX(CASE WHEN pedigree_index = 2 THEN breeding_registration_number END) as pedigree_2,
                MAX(CASE WHEN pedigree_index = 3 THEN breeding_registration_number END) as pedigree_3,
                MAX(CASE WHEN pedigree_index = 4 THEN breeding_registration_number END) as pedigree_4,
                MAX(CASE WHEN pedigree_index = 5 THEN breeding_registration_number END) as pedigree_5,
                MAX(CASE WHEN pedigree_index = 6 THEN breeding_registration_number END) as pedigree_6,
                MAX(CASE WHEN pedigree_index = 7 THEN breeding_registration_number END) as pedigree_7,
                MAX(CASE WHEN pedigree_index = 8 THEN breeding_registration_number END) as pedigree_8,
                MAX(CASE WHEN pedigree_index = 9 THEN breeding_registration_number END) as pedigree_9,
                MAX(CASE WHEN pedigree_index = 10 THEN breeding_registration_number END) as pedigree_10,
                MAX(CASE WHEN pedigree_index = 11 THEN breeding_registration_number END) as pedigree_11,
                MAX(CASE WHEN pedigree_index = 12 THEN breeding_registration_number END) as pedigree_12,
                MAX(CASE WHEN pedigree_index = 13 THEN breeding_registration_number END) as pedigree_13
            FROM public.horse_master_pedigree
            WHERE pedigree_index BETWEEN 0 AND 13
            GROUP BY blood_registration_number
        )
        SELECT 
            hri.kaisai_year || hri.kaisai_month_day || hri.track_code || hri.kaisai_kai || hri.kaisai_day || hri.race_number as race_id_raw,
            hri.horse_number as horse_number_raw,

            -- メタデータ
            hri.kaisai_year || hri.kaisai_month_day as kaisai_ymd_raw,
            rd.start_time as start_hm_raw,

            -- 数値特徴量
            hri.horse_weight as horse_weight_raw,
            hri.weight_change_sign || hri.weight_change as weight_change_raw,
            rd.distance as distance_raw,
            hri.race_number as race_number_raw,
            rd.registration_count as registration_count_raw,
            -- hri.horse_number as horse_number_raw,  # メタデータセクションで取得済み
            hri.frame_number as frame_number_raw,
            hri.horse_age as horse_age_raw,
            hri.burden_weight as burden_weight_raw,

            -- カテゴリ特徴量
            hri.blood_registration_number as horse_id_raw,
            hri.jockey_code as jockey_id_raw,
            rd.track_code_detail as track_detail_raw,
            rd.weather_code as weather_raw,
            rd.turf_condition_code as turf_cond_raw,
            rd.dirt_condition_code as dirt_cond_raw,
            rd.grade_code as grade_raw,
            rd.course_kubun as course_raw,
            hri.sex_code as sex_raw,
            hri.trainer_area_code as trainer_area_raw,
            hri.trainer_code as trainer_id_raw,
            hri.blinker_use_kubun as blinker_use_raw,
            jm.sex_code as jockey_sex_raw,
            tm.sex_code as trainer_sex_raw,

            -- 血統情報
            pp.pedigree_0 as pedigree_0_raw,
            pp.pedigree_1 as pedigree_1_raw,
            pp.pedigree_2 as pedigree_2_raw,
            pp.pedigree_3 as pedigree_3_raw,
            pp.pedigree_4 as pedigree_4_raw,
            pp.pedigree_5 as pedigree_5_raw,
            pp.pedigree_6 as pedigree_6_raw,
            pp.pedigree_7 as pedigree_7_raw,
            pp.pedigree_8 as pedigree_8_raw,
            pp.pedigree_9 as pedigree_9_raw,
            pp.pedigree_10 as pedigree_10_raw,
            pp.pedigree_11 as pedigree_11_raw,
            pp.pedigree_12 as pedigree_12_raw,
            pp.pedigree_13 as pedigree_13_raw,

            -- ターゲット
            hri.final_order as ranking_raw

        FROM
            public.horse_race_info hri
        INNER JOIN 
            public.race_detail rd ON
            hri.kaisai_year = rd.kaisai_year AND
            hri.kaisai_month_day = rd.kaisai_month_day AND
            hri.track_code = rd.track_code AND
            hri.kaisai_kai = rd.kaisai_kai AND
            hri.kaisai_day = rd.kaisai_day AND
            hri.race_number = rd.race_number
        LEFT JOIN
            public.jockey_master jm ON
            hri.jockey_code = jm.jockey_code
        LEFT JOIN
            public.trainer_master tm ON
            hri.trainer_code = tm.trainer_code
        LEFT JOIN
            pedigree_pivot pp ON
            hri.blood_registration_number = pp.blood_registration_number
        WHERE 
            hri.horse_number BETWEEN '01' AND '{self.max_horses:02}' AND
            hri.final_order BETWEEN '01' AND '{self.max_horses:02}' AND
            (hri.kaisai_year > '{start_year}' OR (hri.kaisai_year = '{start_year}' AND hri.kaisai_month_day >= '{start_month_day}')) AND
            (hri.kaisai_year < '{end_year}' OR (hri.kaisai_year = '{end_year}' AND hri.kaisai_month_day <= '{end_month_day}'))
        ORDER BY hri.kaisai_year, hri.kaisai_month_day, hri.track_code, hri.kaisai_kai, hri.kaisai_day, hri.horse_number;
        """

        self.fetched_data = pd.read_sql_query(query, engine)
        logger.info(f"Fetched {len(self.fetched_data)} records from the database.")
        return self

    def prepare(self):
        """データの前処理を行う"""
        logger.info("Preparing data...")

        data = self.fetched_data.copy()

        # ==========================================
        # メタデータ
        # ==========================================

        # race_id: race_id_rawをそのまま使用
        data['race_id'] = data['race_id_raw']

        # horse_number_int: horse_number_rawをint64に変換
        data['horse_number_int'] = data['horse_number_raw'].astype(np.int64)
        horse_number_mask = data['horse_number_int'].between(1, self.max_horses)  # 1からmax_horsesの範囲
        data.loc[~horse_number_mask, 'horse_number_int'] = 0  # 無効な値は0に設定

        # kaisai_ymd_date: kaisai_ymd_rawをdate型に変換
        data['kaisai_ymd_date'] = pd.to_datetime(data['kaisai_ymd_raw'], format='%Y%m%d', errors='coerce').dt.date
        kaisai_ymd_mask = data['kaisai_ymd_raw'] != '00000000'
        data.loc[~kaisai_ymd_mask, 'kaisai_ymd_date'] = pd.NaT

        # start_hm_time: start_hm_rawをtime型に変換 (HHMM形式を想定)
        data['start_hm_time'] = pd.to_datetime(data['start_hm_raw'], format='%H%M', errors='coerce').dt.time
        start_hm_mask = data['start_hm_raw'] != '0000'
        data.loc[~start_hm_mask, 'start_hm_time'] = pd.NaT

        # kaisai_start_datetime: kaisai_ymd_dateとstart_hm_timeを結合してdatetime型に変換
        data['kaisai_start_datetime'] = pd.to_datetime(data['kaisai_ymd_date'], errors='coerce')
        start_hm_mask = data['start_hm_time'].notna()
        data.loc[start_hm_mask, 'kaisai_start_datetime'] += pd.to_timedelta(data.loc[start_hm_mask, 'start_hm_time'].astype(str))


        # ==========================================
        # 数値特徴量
        # ==========================================

        # horse_weight_numeric: 馬体重
        data['horse_weight_numeric'] = pd.to_numeric(data['horse_weight_raw'], errors='coerce').astype(np.float32)
        horse_weight_mask = data['horse_weight_numeric'].between(2, 998)  # 馬体重は2kgから998kgの範囲
        data.loc[~horse_weight_mask, 'horse_weight_numeric'] = np.nan

        # weight_change_numeric: 増減差
        data['weight_change_numeric'] = pd.to_numeric(data['weight_change_raw'], errors='coerce').astype(np.float32)
        weight_change_mask = data['weight_change_numeric'].between(-998, 998)  # 増減差は-998から998の範囲
        data.loc[~weight_change_mask, 'weight_change_numeric'] = np.nan

        # distance_numeric: 距離
        data['distance_numeric'] = pd.to_numeric(data['distance_raw'], errors='coerce').astype(np.float32)
        data.loc[data['distance_numeric'] == 0, 'distance_numeric'] = np.nan  # 距離が0のレコードは無効とする

        # race_number_numeric: レース番号
        data['race_number_numeric'] = pd.to_numeric(data['race_number_raw'], errors='coerce').astype(np.float32)
        data.loc[data['race_number_numeric'] == 0, 'race_number_numeric'] = np.nan  # レース番号が0のレコードは無効

        # registration_count_numeric: 登録頭数
        data['registration_count_numeric'] = pd.to_numeric(data['registration_count_raw'], errors='coerce').astype(np.float32)
        registration_count_mask = data['registration_count_numeric'].between(1, self.max_horses)  # 1からmax_horsesの範囲
        data.loc[~registration_count_mask, 'registration_count_numeric'] = np.nan  # 無効な値はNaNに設定

        # horse_number_numeric: 馬番号
        data['horse_number_numeric'] = pd.to_numeric(data['horse_number_raw'], errors='coerce').astype(np.float32)
        horse_number_numeric_mask = data['horse_number_numeric'].between(1, self.max_horses)  # 1からmax_horsesの範囲
        data.loc[~horse_number_numeric_mask, 'horse_number_numeric'] = np.nan  # 無効な値はNaNに設定

        # frame_number_numeric: 枠番
        data['frame_number_numeric'] = pd.to_numeric(data['frame_number_raw'], errors='coerce').astype(np.float32)
        data.loc[data['frame_number_numeric'] == 0, 'frame_number_numeric'] = np.nan  # 枠番が0のレコードは無効とする

        # horse_age_numeric: 馬齢
        data['horse_age_numeric'] = pd.to_numeric(data['horse_age_raw'], errors='coerce').astype(np.float32)
        horse_age_mask = data['horse_age_numeric'].between(1, 30)  # 馬齢は1歳から30歳の範囲
        data.loc[~horse_age_mask, 'horse_age_numeric'] = np.nan

        # burden_weight_numeric: 負担重量 
        data['burden_weight_numeric'] = pd.to_numeric(data['burden_weight_raw'], errors='coerce').astype(np.float32) * 0.1  # 負担重量は0.1倍
        burden_weight_mask = data['burden_weight_numeric'].between(0.1, 99.9)  # 負担重量は0.1倍から99.9倍の範囲
        data.loc[~burden_weight_mask, 'burden_weight_numeric'] = np.nan  # 無効な値はNaNに設定

        # ==========================================
        # カテゴリ特徴量
        # ==========================================

        # horse_id_valid: 血統登録番号
        data['horse_id_valid'] = data['horse_id_raw'].fillna("<NULL>")
        data.loc[data['horse_id_valid'] == "0000000000", 'horse_id_valid'] = "<NULL>"  # 0000000000は無効とする

        # jockey_id_valid: 騎手コード
        data['jockey_id_valid'] = data['jockey_id_raw'].fillna("<NULL>")
        data.loc[data['jockey_id_valid'] == "00000", 'jockey_id_valid'] = "<NULL>"  # 00000は無効とする

        # track_detail_valid: トラックコード詳細
        data['track_detail_valid'] = data['track_detail_raw'].fillna("<NULL>")
        data.loc[data['track_detail_valid'] == "00", 'track_detail_valid'] = "<NULL>"  # 00は無効とする

        # track_type_valid: 芝 or ダート
        turf_mask = data['track_detail_valid'].between('10', '22') | data['track_detail_valid'].equals('51') | data['track_detail_valid'].between('53', '59')
        dirt_mask = data['track_detail_valid'].between('23', '29') | data['track_detail_valid'].equals('52')
        data['track_type_valid'] = "<NULL>"
        data.loc[turf_mask, 'track_type_valid'] = 'turf'
        data.loc[dirt_mask, 'track_type_valid'] = 'dirt'

        # weather_valid: 天候コード
        data['weather_valid'] = data['weather_raw'].fillna("<NULL>")
        data.loc[data['weather_valid'] == "0", 'weather_valid'] = "<NULL>"  # 0は無効とする

        # track_cond_valid: 馬場状態
        data['track_cond_valid'] = "<NULL>"
        data.loc[data['track_type_valid'] == 'turf', 'track_cond_valid'] = data['turf_cond_raw']
        data.loc[data['track_type_valid'] == 'dirt', 'track_cond_valid'] = data['dirt_cond_raw']
        data.loc[data['track_cond_valid'] == '0', 'track_cond_valid'] = "<NULL>"  # 0は無効とする

        # grade_valid: グレードコード
        data['grade_valid'] = data['grade_raw'].fillna("<NULL>")
        data.loc[data['grade_valid'] == "0", 'grade_valid'] = "<NULL>"

        # course_valid: コース区分
        data['course_valid'] = data['course_raw'].fillna("<NULL>")
        data.loc[data['course_valid'] == "  ", 'course_valid'] = "<NULL>"

        # sex_valid: 性別コード
        data['sex_valid'] = data['sex_raw'].fillna("<NULL>")
        data.loc[data['sex_valid'] == "0", 'sex_valid'] = "<NULL>"

        # trainer_area_valid: 調教師エリアコード
        data['trainer_area_valid'] = data['trainer_area_raw'].fillna("<NULL>")
        data.loc[data['trainer_area_valid'] == "0", 'trainer_area_valid'] = "<NULL>"  # 0は無効とする

        # trainer_id_valid: 調教師コード
        data['trainer_id_valid'] = data['trainer_id_raw'].fillna("<NULL>")
        data.loc[data['trainer_id_valid'] == "00000", 'trainer_id_valid'] = "<NULL>"  # 00000は無効とする

        # blinker_use_valid: ブリンカー使用区分
        data['blinker_use_valid'] = data['blinker_use_raw'].fillna("<NULL>")  # 0は未使用を意味する有効値

        # jockey_sex_valid: 騎手性別区分
        data['jockey_sex_valid'] = data['jockey_sex_raw'].fillna('<NULL>')
        data.loc[data['jockey_sex_valid'] == "0", 'jockey_sex_valid'] = "<NULL>"

        # trainer_sex_valid: 調教師性別区分
        data['trainer_sex_valid'] = data['trainer_sex_raw'].fillna('<NULL>')
        data.loc[data['trainer_sex_valid'] == "0", 'trainer_sex_valid'] = "<NULL>"
 
        # pedigree_0_valid から pedigree_13_valid: 血統情報
        for i in range(14):
            col_name = f'pedigree_{i}_valid'
            raw_col_name = f'pedigree_{i}_raw'
            data[col_name] = data[raw_col_name].fillna("<NULL>")
            data.loc[data[col_name] == "0000000000", col_name] = "<NULL>"  # 0000000000は無効とする

        # ==========================================
        # ターゲット
        # ==========================================

        # ranking_int: 最終着順
        data['ranking_int'] = data['ranking_raw'].astype(np.int64)
        ranking_mask = data['ranking_int'].between(1, self.max_horses)  # 1からnum_horsesの範囲
        data.loc[~ranking_mask, 'ranking_int'] = 0  # 無効な値は0に設定

        self.prepared_data = data
        logger.info(f"Prepared data with {len(self.prepared_data)} records.")

        return self

    def build(self):
        """データ構造を構築する"""
        logger.info("Building data structure...")

        data = self.prepared_data
        target_data = data[data['kaisai_ymd_date'].between(self.start_date, self.end_date)]

        # レースIDでグループ化
        race_groups = target_data.groupby('race_id')

        # レース一覧を取得 (レースID順にソート)
        race_ids = sorted(race_groups.groups.keys())
        num_races = len(race_ids)

        logger.info(f"Building data for {num_races} races...")

        # データ構造の初期化
        self.built_data = {
            "x_num": {
                "horse_weight": np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
                "weight_change": np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
                "distance": np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
                "race_number": np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
                "registration_count": np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
                "horse_number": np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
                "frame_number": np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
                "horse_age": np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
                "burden_weight": np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
            },
            "x_cat": {
                "horse_id": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "jockey_id": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                'track_detail': np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                'track_type': np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "weather": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "track_cond": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "grade": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "course": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "sex": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "trainer_area": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "trainer_id": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "blinker_use": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "jockey_sex": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "trainer_sex": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_0": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_1": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_2": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_3": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_4": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_5": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_6": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_7": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_8": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_9": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_10": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_11": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_12": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "pedigree_13": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
            },
            "sequence_data": {
                "horse_weight_history": {
                    "x_num": {
                        'days_before': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "horse_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "weight_change": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                    },
                    "x_cat": {},
                    "mask": np.zeros((num_races, self.max_horses, self.max_hist_len), dtype=bool),
                }
            },
            "mask": np.zeros((num_races, self.max_horses), dtype=bool),
            "rankings": np.full((num_races, self.max_horses), 0, dtype=np.int64),  # 0は無効値
            'race_id': np.array(race_ids, dtype=object),
        }

        # シーケンスデータ構築に使用するグループ（ソート済みに変更）
        horse_groups = {}
        for name, group in data.groupby('horse_id_valid'):
            # 各馬のデータを事前にソート (降順)
            horse_groups[name] = group.sort_values('kaisai_start_datetime', ascending=False)

        # 進捗報告の間隔を設定
        progress_interval = min(1000, max(1, -(-num_races // 10)))  # 10%ごとに進捗報告

        # 各レースのデータを構築
        for race_idx, race_id in enumerate(race_ids):

            # 進捗ログ出力
            if (race_idx + 1) % progress_interval == 0 or (race_idx + 1) == num_races:
                progress_pct = (race_idx + 1) / num_races * 100
                logger.info(f"Processing race {race_idx + 1}/{num_races} ({progress_pct:.1f}%): {race_id}")

            race_data = race_groups.get_group(race_id)
            
            # シーケンスデータ構築のためタイムスタンプを取得（最初の行から一度だけ取得）
            current_datetime = race_data.iloc[0]['kaisai_start_datetime']
            cutoff_datetime = current_datetime - timedelta(hours=self.hours_before_race)

            for _, row in race_data.iterrows():
                horse_idx = row['horse_number_int'] - 1  # 0-indexedに変換

                # マスク
                self.built_data['mask'][race_idx, horse_idx] = True

                # 数値特徴量
                self.built_data['x_num']['horse_weight'][race_idx, horse_idx] = row['horse_weight_numeric']
                self.built_data['x_num']['weight_change'][race_idx, horse_idx] = row['weight_change_numeric']
                self.built_data['x_num']['distance'][race_idx, horse_idx] = row['distance_numeric']
                self.built_data['x_num']['race_number'][race_idx, horse_idx] = row['race_number_numeric']
                self.built_data['x_num']['registration_count'][race_idx, horse_idx] = row['registration_count_numeric']
                self.built_data['x_num']['horse_number'][race_idx, horse_idx] = row['horse_number_numeric']
                self.built_data['x_num']['frame_number'][race_idx, horse_idx] = row['frame_number_numeric']
                self.built_data['x_num']['horse_age'][race_idx, horse_idx] = row['horse_age_numeric']
                self.built_data['x_num']['burden_weight'][race_idx, horse_idx] = row['burden_weight_numeric']

                # カテゴリ特徴量
                self.built_data['x_cat']['horse_id'][race_idx, horse_idx] = row['horse_id_valid']
                self.built_data['x_cat']['jockey_id'][race_idx, horse_idx] = row['jockey_id_valid']
                self.built_data['x_cat']['track_detail'][race_idx, horse_idx] = row['track_detail_valid']
                self.built_data['x_cat']['track_type'][race_idx, horse_idx] = row['track_type_valid']
                self.built_data['x_cat']['weather'][race_idx, horse_idx] = row['weather_valid']
                self.built_data['x_cat']['track_cond'][race_idx, horse_idx] = row['track_cond_valid']
                self.built_data['x_cat']['grade'][race_idx, horse_idx] = row['grade_valid']
                self.built_data['x_cat']['course'][race_idx, horse_idx] = row['course_valid']
                self.built_data['x_cat']['sex'][race_idx, horse_idx] = row['sex_valid']
                self.built_data['x_cat']['trainer_area'][race_idx, horse_idx] = row['trainer_area_valid']
                self.built_data['x_cat']['trainer_id'][race_idx, horse_idx] = row['trainer_id_valid']
                self.built_data['x_cat']['blinker_use'][race_idx, horse_idx] = row['blinker_use_valid']
                self.built_data['x_cat']['jockey_sex'][race_idx, horse_idx] = row['jockey_sex_valid']
                self.built_data['x_cat']['trainer_sex'][race_idx, horse_idx] = row['trainer_sex_valid']
                self.built_data['x_cat']['pedigree_0'][race_idx, horse_idx] = row['pedigree_0_valid']
                self.built_data['x_cat']['pedigree_1'][race_idx, horse_idx] = row['pedigree_1_valid']
                self.built_data['x_cat']['pedigree_2'][race_idx, horse_idx] = row['pedigree_2_valid']
                self.built_data['x_cat']['pedigree_3'][race_idx, horse_idx] = row['pedigree_3_valid']
                self.built_data['x_cat']['pedigree_4'][race_idx, horse_idx] = row['pedigree_4_valid']
                self.built_data['x_cat']['pedigree_5'][race_idx, horse_idx] = row['pedigree_5_valid']
                self.built_data['x_cat']['pedigree_6'][race_idx, horse_idx] = row['pedigree_6_valid']
                self.built_data['x_cat']['pedigree_7'][race_idx, horse_idx] = row['pedigree_7_valid']
                self.built_data['x_cat']['pedigree_8'][race_idx, horse_idx] = row['pedigree_8_valid']
                self.built_data['x_cat']['pedigree_9'][race_idx, horse_idx] = row['pedigree_9_valid']
                self.built_data['x_cat']['pedigree_10'][race_idx, horse_idx] = row['pedigree_10_valid']
                self.built_data['x_cat']['pedigree_11'][race_idx, horse_idx] = row['pedigree_11_valid']
                self.built_data['x_cat']['pedigree_12'][race_idx, horse_idx] = row['pedigree_12_valid']
                self.built_data['x_cat']['pedigree_13'][race_idx, horse_idx] = row['pedigree_13_valid']

                # ターゲット
                self.built_data['rankings'][race_idx, horse_idx] = row['ranking_int']  # ranking_intを使用

                # シーケンスデータ構築時に使用するキー
                horse_key = row['horse_id_valid']  

                # 馬体重履歴データの構築
                if horse_key != "<NULL>" and horse_key in horse_groups:
                    history = horse_groups[horse_key]

                    # 過去のデータのみフィルタ（既にソート済みなので効率的）
                    valid_history = history[history['kaisai_start_datetime'] < cutoff_datetime].head(self.max_hist_len)  # 既にソート済みなのでheadで十分
                    
                    valid_length = len(valid_history)
                    if valid_length > 0:
                        days_before = np.array([(current_datetime - hist_datetime).total_seconds() / (24 * 3600) for hist_datetime in valid_history['kaisai_start_datetime']], dtype=np.float32)
                        self.built_data['sequence_data']['horse_weight_history']['x_num']['days_before'][race_idx, horse_idx, :valid_length] = days_before
                        self.built_data['sequence_data']['horse_weight_history']['x_num']['horse_weight'][race_idx, horse_idx, :valid_length] = valid_history['horse_weight_numeric'].values
                        self.built_data['sequence_data']['horse_weight_history']['x_num']['weight_change'][race_idx, horse_idx, :valid_length] = valid_history['weight_change_numeric'].values
                        self.built_data['sequence_data']['horse_weight_history']['mask'][race_idx, horse_idx, :valid_length] = True

        logger.info("Data structure construction completed successfully.")
        return self

    def fit(self):
        # エイリアス設定（血統情報を統一）
        self.params = {
            'numerical': {}, 
            'categorical': {}, 
            'aliases': {
                # 血統情報のエイリアス（全世代統一）
                'pedigree_0': 'pedigree',
                'pedigree_1': 'pedigree',
                'pedigree_2': 'pedigree',
                'pedigree_3': 'pedigree',
                'pedigree_4': 'pedigree',
                'pedigree_5': 'pedigree',
                'pedigree_6': 'pedigree',
                'pedigree_7': 'pedigree',
                'pedigree_8': 'pedigree',
                'pedigree_9': 'pedigree',
                'pedigree_10': 'pedigree',
                'pedigree_11': 'pedigree',
                'pedigree_12': 'pedigree',
                'pedigree_13': 'pedigree',
            }
        }

        # 特徴量の収集
        feature_values = {'numerical': defaultdict(lambda: np.full(0, np.nan, dtype=np.float32)), 'categorical': defaultdict(lambda: np.full(0, "<NULL>", dtype=object))}
        for name, data in self.built_data['x_num'].items():
            alias = self.params['aliases'].get(name, name)
            feature_values['numerical'][alias] = np.concatenate([feature_values['numerical'][alias], data.reshape(-1)])
        for name, data in self.built_data['x_cat'].items():
            alias = self.params['aliases'].get(name, name)
            feature_values['categorical'][alias] = np.concatenate([feature_values['categorical'][alias], data.reshape(-1)])
        for seq_data in self.built_data['sequence_data'].values():
            for name, data in seq_data['x_num'].items():
                alias = self.params['aliases'].get(name, name)
                feature_values['numerical'][alias] = np.concatenate([feature_values['numerical'][alias], data.reshape(-1)])
            for name, data in seq_data['x_cat'].items():
                alias = self.params['aliases'].get(name, name)
                feature_values['categorical'][alias] = np.concatenate([feature_values['categorical'][alias], data.reshape(-1)])

        # 数値特徴量のパラメータ取得
        for name, data in feature_values['numerical'].items():
            mean = np.nanmean(data)
            std = np.nanstd(data)
            self.params['numerical'][name] = {'mean': mean, 'std': std}
            logger.info(f"Numerical feature '{name}' has mean: {mean}, std: {std}")

        # カテゴリ特徴量のエンコーダ作成
        for name, data in feature_values['categorical'].items():
            unique_values = np.unique(data)
            valid_values = unique_values[unique_values != "<NULL>"]
            encoder = {"<NULL>": 0}
            for i, value in enumerate(sorted(valid_values), start=1):
                encoder[value] = i
            decoder = {v: k for k, v in encoder.items()}
            self.params['categorical'][name] = {
                'encoder': encoder,
                'decoder': decoder,
                'num_classes': len(encoder)
            }
            logger.info(f"Categorical feature '{name}' has {len(encoder)} classes")

        return self

    def transform(self):
        """データの変換を行う（標準化・エンコーディング）"""
        logger.info("Transforming data...")
        
        # built_dataをコピーしてtransformed_dataを作成
        self.transformed_data = {
            "x_num": {},
            "x_cat": {},
            "sequence_data": {},
            "mask": self.built_data['mask'].copy(),
            "rankings": self.built_data['rankings'].copy(),
            "race_id": self.built_data['race_id'].copy(),
        }
        
        def _transform_numerical_features(data: Dict[str, np.ndarray], feature_params: Dict[str, Dict[str, float]]) -> Dict[str, np.ndarray]:
            """数値特徴量の標準化を行う内部メソッド"""
            transformed = {}
            for name, values in data.items():
                alias = self.params.get('aliases', {}).get(name, name)
                mean = feature_params[alias]['mean']
                std = feature_params[alias]['std']
                # 標準化（stdが0の場合は0で埋める）
                if std > 0:
                    transformed[name] = ((values - mean) / std).astype(np.float32)
                else:
                    transformed[name] = np.zeros_like(values, dtype=np.float32)
            return transformed
        
        def _transform_categorical_features(data: Dict[str, np.ndarray], feature_params: Dict[str, Dict[str, Any]]) -> Dict[str, np.ndarray]:
            """カテゴリ特徴量のエンコーディングを行う内部メソッド"""
            transformed = {}
            for name, values in data.items():
                alias = self.params.get('aliases', {}).get(name, name)
                encoder = feature_params[alias]['encoder']
                # エンコーディング（未知の値は<NULL>として扱う）
                encoded_values = np.array([encoder.get(str(val), encoder["<NULL>"]) for val in values.flatten()])
                transformed[name] = encoded_values.reshape(values.shape).astype(np.int64)
            return transformed
        
        # 通常の特徴量の変換
        self.transformed_data['x_num'] = _transform_numerical_features(
            self.built_data['x_num'], 
            self.params['numerical']
        )
        self.transformed_data['x_cat'] = _transform_categorical_features(
            self.built_data['x_cat'], 
            self.params['categorical']
        )
        
        # シーケンスデータの変換
        for seq_name, seq_data in self.built_data['sequence_data'].items():
            self.transformed_data['sequence_data'][seq_name] = {
                'x_num': _transform_numerical_features(
                    seq_data['x_num'], 
                    self.params['numerical']
                ),
                'x_cat': _transform_categorical_features(
                    seq_data['x_cat'], 
                    self.params['categorical']
                ),
                'mask': seq_data['mask'].copy()
            }
        
        logger.info("Data transformation completed successfully.")
        return self

    def get_params(self) -> Dict[str, Any]:
        return self.params
    
    def set_params(self, params):
        self.params = params
        return self

    def __len__(self) -> int:
        """データセットの長さを返す"""
        return len(self.transformed_data['race_id'])
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """指定されたインデックスのデータを返す"""
        data = self.transformed_data
        return {
            "x_num": {feat_name: feat_data[idx] for feat_name, feat_data in data['x_num'].items()},
            "x_cat": {feat_name: feat_data[idx] for feat_name, feat_data in data['x_cat'].items()},
            "sequence_data": {
                seq_name: {
                    "x_num": {feat_name: feat_data[idx] for feat_name, feat_data in seq_data["x_num"].items()},
                    "x_cat": {feat_name: feat_data[idx] for feat_name, feat_data in seq_data["x_cat"].items()},
                    "mask": seq_data["mask"][idx],
                } for seq_name, seq_data in data['sequence_data'].items()
            },
            "mask": data['mask'][idx],
            "rankings": data['rankings'][idx],
            "race_id": data['race_id'][idx],
        }

