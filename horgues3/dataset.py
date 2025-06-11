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

    def __init__(self, start_ymd: str, end_ymd: str, max_horses: int = 18, max_hist_len: int = 18, max_prev_days: int = 365, hours_before_race: int = 2):
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

        self.feature_aliases = {
            # 血統情報のエイリアス
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
            # ラップタイムのエイリアス
            **{f'lap_time_{i}': 'lap_time' for i in range(25)},
            # 調教タイムのエイリアス
            'furlong_10_total_time': 'furlong_total_time',
            'furlong_9_total_time': 'furlong_total_time',
            'furlong_8_total_time': 'furlong_total_time',
            'furlong_7_total_time': 'furlong_total_time',
            'furlong_6_total_time': 'furlong_total_time',
            'furlong_5_total_time': 'furlong_total_time',
            'furlong_4_total_time': 'furlong_total_time',
            'furlong_3_total_time': 'furlong_total_time',
            'furlong_2_total_time': 'furlong_total_time',
            # 調教ラップタイムのエイリアス
            'lap_time_2000_1800': 'training_lap_time',
            'lap_time_1800_1600': 'training_lap_time',
            'lap_time_1600_1400': 'training_lap_time',
            'lap_time_1400_1200': 'training_lap_time',
            'lap_time_1200_1000': 'training_lap_time',
            'lap_time_1000_800': 'training_lap_time',
            'lap_time_800_600': 'training_lap_time',
            'lap_time_600_400': 'training_lap_time',
            'lap_time_400_200': 'training_lap_time',
            'lap_time_200_0': 'training_lap_time',
            # 相手馬情報
            'rival_horse_id': 'horse_id',
        }

    def fetch(self):
        """ データベースからデータを取得する"""
        logger.info("Fetching data from database...")

        extended_start_date = self.start_date - timedelta(days=self.max_prev_days)

        start_year = extended_start_date.strftime("%Y")
        start_month_day = extended_start_date.strftime("%m%d")
        end_year = self.end_date.strftime("%Y")
        end_month_day = self.end_date.strftime("%m%d")
        
        engine = create_engine("postgresql://postgres:postgres@localhost/horgues3")

        # ==========================================
        # レースデータ
        # ==========================================

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
        ),
        lap_time_pivot AS (
            SELECT 
                kaisai_year || kaisai_month_day || track_code || kaisai_kai || kaisai_day || race_number as race_id,
                MAX(CASE WHEN lap_time_index = 0 THEN lap_time END) as lap_time_0,
                MAX(CASE WHEN lap_time_index = 1 THEN lap_time END) as lap_time_1,
                MAX(CASE WHEN lap_time_index = 2 THEN lap_time END) as lap_time_2,
                MAX(CASE WHEN lap_time_index = 3 THEN lap_time END) as lap_time_3,
                MAX(CASE WHEN lap_time_index = 4 THEN lap_time END) as lap_time_4,
                MAX(CASE WHEN lap_time_index = 5 THEN lap_time END) as lap_time_5,
                MAX(CASE WHEN lap_time_index = 6 THEN lap_time END) as lap_time_6,
                MAX(CASE WHEN lap_time_index = 7 THEN lap_time END) as lap_time_7,
                MAX(CASE WHEN lap_time_index = 8 THEN lap_time END) as lap_time_8,
                MAX(CASE WHEN lap_time_index = 9 THEN lap_time END) as lap_time_9,
                MAX(CASE WHEN lap_time_index = 10 THEN lap_time END) as lap_time_10,
                MAX(CASE WHEN lap_time_index = 11 THEN lap_time END) as lap_time_11,
                MAX(CASE WHEN lap_time_index = 12 THEN lap_time END) as lap_time_12,
                MAX(CASE WHEN lap_time_index = 13 THEN lap_time END) as lap_time_13,
                MAX(CASE WHEN lap_time_index = 14 THEN lap_time END) as lap_time_14,
                MAX(CASE WHEN lap_time_index = 15 THEN lap_time END) as lap_time_15,
                MAX(CASE WHEN lap_time_index = 16 THEN lap_time END) as lap_time_16,
                MAX(CASE WHEN lap_time_index = 17 THEN lap_time END) as lap_time_17,
                MAX(CASE WHEN lap_time_index = 18 THEN lap_time END) as lap_time_18,
                MAX(CASE WHEN lap_time_index = 19 THEN lap_time END) as lap_time_19,
                MAX(CASE WHEN lap_time_index = 20 THEN lap_time END) as lap_time_20,
                MAX(CASE WHEN lap_time_index = 21 THEN lap_time END) as lap_time_21,
                MAX(CASE WHEN lap_time_index = 22 THEN lap_time END) as lap_time_22,
                MAX(CASE WHEN lap_time_index = 23 THEN lap_time END) as lap_time_23,
                MAX(CASE WHEN lap_time_index = 24 THEN lap_time END) as lap_time_24
            FROM public.race_detail_lap_time
            WHERE lap_time_index BETWEEN 0 AND 24
            GROUP BY kaisai_year, kaisai_month_day, track_code, kaisai_kai, kaisai_day, race_number
        )
        SELECT 
            hri.kaisai_year || hri.kaisai_month_day || hri.track_code || hri.kaisai_kai || hri.kaisai_day || hri.race_number as race_id_raw,
            hri.horse_number as horse_number_raw,

            -- メタデータ
            hri.kaisai_year || hri.kaisai_month_day as kaisai_ymd_raw,
            rd.start_time as start_hm_raw,
            hri.data_kubun as data_kubun_raw,

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
            ltp.lap_time_0 as lap_time_0_raw,  -- ラップタイム
            ltp.lap_time_1 as lap_time_1_raw,
            ltp.lap_time_2 as lap_time_2_raw,
            ltp.lap_time_3 as lap_time_3_raw,
            ltp.lap_time_4 as lap_time_4_raw,
            ltp.lap_time_5 as lap_time_5_raw,
            ltp.lap_time_6 as lap_time_6_raw,
            ltp.lap_time_7 as lap_time_7_raw,
            ltp.lap_time_8 as lap_time_8_raw,
            ltp.lap_time_9 as lap_time_9_raw,
            ltp.lap_time_10 as lap_time_10_raw,
            ltp.lap_time_11 as lap_time_11_raw,
            ltp.lap_time_12 as lap_time_12_raw,
            ltp.lap_time_13 as lap_time_13_raw,
            ltp.lap_time_14 as lap_time_14_raw,
            ltp.lap_time_15 as lap_time_15_raw,
            ltp.lap_time_16 as lap_time_16_raw,
            ltp.lap_time_17 as lap_time_17_raw,
            ltp.lap_time_18 as lap_time_18_raw,
            ltp.lap_time_19 as lap_time_19_raw,
            ltp.lap_time_20 as lap_time_20_raw,
            ltp.lap_time_21 as lap_time_21_raw,
            ltp.lap_time_22 as lap_time_22_raw,
            ltp.lap_time_23 as lap_time_23_raw,
            ltp.lap_time_24 as lap_time_24_raw,
            hri.finish_time as finish_time_raw,
            hri.last_3furlong_time as last_3furlong_raw,
            hri.last_4furlong_time as last_4furlong_raw,

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
            pp.pedigree_0 as pedigree_0_raw,  -- 3代血統情報
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
            hri.track_code as track_raw,
            hri.running_style_judgment as running_style_raw,
            --hrir.rival_blood_registration_number as rival_horse_id_raw,

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
        LEFT JOIN
            lap_time_pivot ltp ON
            hri.kaisai_year || hri.kaisai_month_day || hri.track_code || hri.kaisai_kai || hri.kaisai_day || hri.race_number = ltp.race_id
        --INNER JOIN
        --    public.horse_race_info_rival hrir ON
        --    rd.kaisai_year = hrir.kaisai_year AND
        --    rd.kaisai_month_day = hrir.kaisai_month_day AND
        --    rd.track_code = hrir.track_code AND
        --    rd.kaisai_kai = hrir.kaisai_kai AND
        --    rd.kaisai_day = hrir.kaisai_day AND
        --    rd.race_number = hrir.race_number AND
        --    hri.horse_number = hrir.horse_number AND
        --    hrir.rival_index = 0

        WHERE 
            (hri.kaisai_year > '{start_year}' OR (hri.kaisai_year = '{start_year}' AND hri.kaisai_month_day >= '{start_month_day}')) AND
            (hri.kaisai_year < '{end_year}' OR (hri.kaisai_year = '{end_year}' AND hri.kaisai_month_day <= '{end_month_day}'))
        ORDER BY hri.kaisai_year, hri.kaisai_month_day, hri.track_code, hri.kaisai_kai, hri.kaisai_day, hri.horse_number;
        """

        self.fetched_data = pd.read_sql_query(query, engine)
        logger.info(f"Fetched {len(self.fetched_data)} records from the database.")

        # 追加のテーブル
        self.additional_fetched_data = {}

        # ==========================================
        # 調教データ（ウッドチップと坂路をUNIONで統合）
        # ==========================================

        training_query = f"""
        SELECT
            blood_registration_number as horse_id_raw,
            training_date as training_date_raw,
            training_time as training_time_raw,
            training_center_kubun as training_center_raw,
            course as course_raw,
            track_direction as track_direction_raw,
            furlong_10_total_time as furlong_10_total_time_raw,
            lap_time_2000_1800 as lap_time_2000_1800_raw,
            furlong_9_total_time as furlong_9_total_time_raw,
            lap_time_1800_1600 as lap_time_1800_1600_raw,
            furlong_8_total_time as furlong_8_total_time_raw,
            lap_time_1600_1400 as lap_time_1600_1400_raw,
            furlong_7_total_time as furlong_7_total_time_raw,
            lap_time_1400_1200 as lap_time_1400_1200_raw,
            furlong_6_total_time as furlong_6_total_time_raw,
            lap_time_1200_1000 as lap_time_1200_1000_raw,
            furlong_5_total_time as furlong_5_total_time_raw,
            lap_time_1000_800 as lap_time_1000_800_raw,
            furlong_4_total_time as furlong_4_total_time_raw,
            lap_time_800_600 as lap_time_800_600_raw,
            furlong_3_total_time as furlong_3_total_time_raw,
            lap_time_600_400 as lap_time_600_400_raw,
            furlong_2_total_time as furlong_2_total_time_raw,
            lap_time_400_200 as lap_time_400_200_raw,
            lap_time_200_0 as lap_time_200_0_raw,
            'woodchip' as training_type_raw
        FROM public.woodchip_training
        WHERE 
            (training_date > '{start_year}{start_month_day}' OR (training_date = '{start_year}{start_month_day}')) AND
            (training_date < '{end_year}{end_month_day}' OR (training_date = '{end_year}{end_month_day}'))
        
        UNION ALL
        
        SELECT 
            blood_registration_number as horse_id_raw,
            training_date as training_date_raw,
            training_time as training_time_raw,
            training_center_kubun as training_center_raw,
            NULL as course_raw,
            NULL as track_direction_raw,
            NULL as furlong_10_total_time_raw,
            NULL as lap_time_2000_1800_raw,
            NULL as furlong_9_total_time_raw,
            NULL as lap_time_1800_1600_raw,
            NULL as furlong_8_total_time_raw,
            NULL as lap_time_1600_1400_raw,
            NULL as furlong_7_total_time_raw,
            NULL as lap_time_1400_1200_raw,
            NULL as furlong_6_total_time_raw,
            NULL as lap_time_1200_1000_raw,
            NULL as furlong_5_total_time_raw,
            NULL as lap_time_1000_800_raw,
            four_furlong_total_time as furlong_4_total_time_raw,
            lap_time_800_600 as lap_time_800_600_raw,
            three_furlong_total_time as furlong_3_total_time_raw,
            lap_time_600_400 as lap_time_600_400_raw,
            two_furlong_total_time as furlong_2_total_time_raw,
            lap_time_400_200 as lap_time_400_200_raw,
            lap_time_200_0 as lap_time_200_0_raw,
            'slope' as training_type_raw
        FROM public.slope_training
        WHERE 
            (training_date > '{start_year}{start_month_day}' OR (training_date = '{start_year}{start_month_day}')) AND
            (training_date < '{end_year}{end_month_day}' OR (training_date = '{end_year}{end_month_day}'))
        
        ORDER BY training_date_raw, training_time_raw;
        """

        self.additional_fetched_data['training'] = pd.read_sql_query(training_query, engine)
        logger.info(f"Fetched {len(self.additional_fetched_data['training'])} training records (woodchip + slope).")

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

        # data_kubun: data_kubun_rawをそのまま使用
        data['data_kubun'] = data['data_kubun_raw']

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

        # lap_time_{i}_numeric: ラップタイム
        for i in range(25):
            raw_col_name = f'lap_time_{i}_raw'
            col_name = f'lap_time_{i}_numeric'
            data[col_name] = pd.to_numeric(data[raw_col_name], errors='coerce').astype(np.float32) * 0.1
            data.loc[data[col_name] == 0, col_name] = np.nan  # 0秒はNaNに設定
        # ラップタイムを右揃えにする
        columns = [f'lap_time_{i}_numeric' for i in range(25)]
        data[columns] = data[columns].apply(lambda row: pd.Series(
            np.concatenate([np.full(np.sum(np.isnan(row)), np.nan), row[~np.isnan(row)]])
        ), axis=1)

        # finish_time_numeric: 走破タイム
        data['finish_time_numeric'] = pd.to_numeric(data['finish_time_raw'][0], errors='coerce').astype(np.float32) * 60 + pd.to_numeric(data['finish_time_raw'][1:], errors='coerce').astype(np.float32) * 0.1
        data.loc[data['finish_time_numeric'] == 0, 'finish_time_numeric'] = np.nan

        # speed_numeric: 走破速度
        data['speed_numeric'] = data['distance_numeric'] / data['finish_time_numeric']

        # last_3furlong_numeric: 後3ハロンタイム
        data['last_3furlong_numeric'] = pd.to_numeric(data['last_3furlong_raw'], errors='coerce').astype(np.float32) * 0.1
        # 0をnanにする
        data.loc[data['last_3furlong_numeric'] == 0, 'last_3furlong_numeric'] = np.nan
        # 障害はnanにする
        shogai_mask = data['track_detail_raw'].between('51', '59')
        data.loc[shogai_mask, 'last_3furlong_numeric'] = np.nan
        # 後4ハロンタイムが設定されていれば0.75倍
        four_furlong = pd.to_numeric(data['last_4furlong_raw'], errors='coerce').astype(np.float32) * 0.1
        four_furlong_mask = four_furlong.notna() & (four_furlong != 0)
        data.loc[four_furlong_mask, 'last_3furlong_numeric'] = four_furlong * 0.75

        # 1. 相対的な馬体重（平均からの差）
        race_avg_weight = data.groupby('race_id')['horse_weight_numeric'].transform('mean')
        data['relative_horse_weight_numeric'] = data['horse_weight_numeric'] - race_avg_weight
        
        # 2. 相対的な負担重量
        race_avg_burden = data.groupby('race_id')['burden_weight_numeric'].transform('mean')
        data['relative_burden_weight_numeric'] = data['burden_weight_numeric'] - race_avg_burden
        
        # 4. 距離適性（前走との距離差）
        data['distance_change_numeric'] = data.groupby('horse_id_raw')['distance_numeric'].diff()
        data.loc[data['horse_id_raw'] == '0000000000', 'distance_change_numeric'] = np.nan
        
        # 5. 休養期間（日数）
        data['rest_days_numeric'] = data.groupby('horse_id_raw')['kaisai_start_datetime'].diff().dt.days
        data.loc[data['horse_id_raw'] == '0000000000', 'rest_days_numeric'] = np.nan

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
        turf_mask = data['track_detail_valid'].between('10', '22') | (data['track_detail_valid'] == '51') | data['track_detail_valid'].between('53', '59')
        dirt_mask = data['track_detail_valid'].between('23', '29') | (data['track_detail_valid'] == '52')
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

        # track_valid: 競馬場コード
        data['track_valid'] = data['track_raw'].fillna("<NULL>")
        data.loc[data['track_valid'] == "00", 'track_valid'] = "<NULL>"

        # running_style_valid: 脚質判定
        data['running_style_valid'] = data['running_style_raw'].fillna('<NULL>')
        data.loc[data['running_style_valid'] == '0', 'running_style_valid'] = '<NULL>'

        # rival_horse_id_valid: 相手馬情報
        # data['rival_horse_id_valid'] = data['rival_horse_id_raw'].fillna("<NULL>")
        # data.loc[data['rival_horse_id_valid'] == "0000000000", 'rival_horse_id_valid'] = "<NULL>"  # 0000000000は無効とする

        # ==========================================
        # ターゲット
        # ==========================================

        # ranking_int: 最終着順
        data['ranking_int'] = data['ranking_raw'].astype(np.int64)
        ranking_mask = data['ranking_int'].between(1, self.max_horses)  # 1からnum_horsesの範囲
        data.loc[~ranking_mask, 'ranking_int'] = 0  # 無効な値は0に設定

        self.prepared_data = data
        logger.info(f"Prepared data with {len(self.prepared_data)} records.")

        # 追加のデータ
        self.additional_prepared_data = {}

        # ==========================================
        # 調教データ（統合された調教データ）
        # ==========================================

        training_data = self.additional_fetched_data['training'].copy()

        # horse_id_valid: 血統登録番号
        training_data['horse_id_valid'] = training_data['horse_id_raw'].fillna("<NULL>")
        training_data.loc[training_data['horse_id_valid'] == "0000000000", 'horse_id_valid'] = "<NULL>"
        
        # training_datetime: 調教日時を結合
        training_data['training_datetime'] = pd.to_datetime(training_data['training_date_raw'] + ' ' + training_data['training_time_raw'], format='%Y%m%d %H%M', errors='coerce')

        # カテゴリ特徴量
        training_data['training_center_valid'] = training_data['training_center_raw'].fillna("<NULL>")  # 0も使用
        training_data['course_valid'] = training_data['course_raw'].fillna("<NULL>")  # 0も使用
        training_data['track_direction_valid'] = training_data['track_direction_raw'].fillna("<NULL>")  # 0も使用
        training_data['training_type_valid'] = training_data['training_type_raw']

        # 数値特徴量（時間データを秒に変換）
        time_columns = [
            'furlong_10_total_time', 'furlong_9_total_time', 'furlong_8_total_time', 'furlong_7_total_time',
            'furlong_6_total_time', 'furlong_5_total_time', 'furlong_4_total_time', 'furlong_3_total_time', 'furlong_2_total_time'
        ]
        lap_time_columns = [
            'lap_time_2000_1800', 'lap_time_1800_1600', 'lap_time_1600_1400', 'lap_time_1400_1200',
            'lap_time_1200_1000', 'lap_time_1000_800', 'lap_time_800_600', 'lap_time_600_400',
            'lap_time_400_200', 'lap_time_200_0'
        ]
        
        for col in time_columns:
            raw_col = f'{col}_raw'
            numeric_col = f'{col}_numeric'
            training_data[numeric_col] = pd.to_numeric(training_data[raw_col], errors='coerce').astype(np.float32) * 0.1
            training_data.loc[training_data[numeric_col] == 0, numeric_col] = np.nan
        
        for col in lap_time_columns:
            raw_col = f'{col}_raw'
            numeric_col = f'{col}_numeric'
            training_data[numeric_col] = pd.to_numeric(training_data[raw_col], errors='coerce').astype(np.float32) * 0.1
            training_data.loc[training_data[numeric_col] == 0, numeric_col] = np.nan
        
        self.additional_prepared_data['training'] = training_data
        logger.info(f"Prepared {len(self.additional_prepared_data['training'])} training records.")

        return self

    def build(self):
        """データ構造を構築する"""
        logger.info("Building data structure...")

        data = self.prepared_data.copy()

        # 予測対象データの絞込
        target_mask = data['kaisai_ymd_date'].between(self.start_date, self.end_date)  # 期間
        target_mask &= data['data_kubun'].isin(['2', '7'])  # 地方競馬や海外競馬は予測しない
        target_mask &= (data['horse_number_int'] >= 1) & (data['horse_number_int'] <= self.max_horses)
        target_mask &= (data['ranking_int'] >= 1) & (data['ranking_int'] <= self.max_horses)
        target_data = data[target_mask].copy()

        # レース情報を事前計算
        race_info = target_data.groupby('race_id').first()
        race_ids = sorted(race_info.index)
        num_races = len(race_ids)

        # 高速化のための事前処理
        # レースIDと馬番をマルチインデックスに変換
        target_data['race_idx'] = target_data['race_id'].map({race_id: idx for idx, race_id in enumerate(race_ids)})
        target_data['horse_idx'] = target_data['horse_number_int'] - 1

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
                'relative_horse_weight': np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
                'relative_burden_weight': np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
                'distance_change': np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
                'rest_days': np.full((num_races, self.max_horses), np.nan, dtype=np.float32),
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
                "track": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
            },
            "sequence_data": {
                "horse_history": {
                    "x_num": {
                        'days_before': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "horse_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "weight_change": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "distance": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "race_number": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "registration_count": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "horse_number": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "frame_number": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "horse_age": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "burden_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "finish_time": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "speed": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "last_3furlong": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "relative_horse_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "relative_burden_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "distance_change": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "rest_days": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                    },
                    "x_cat": {
                        'horse_id': np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "jockey_id": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        'track_detail': np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        'track_type': np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "weather": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "track_cond": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "grade": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "course": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "sex": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "trainer_area": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "trainer_id": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "blinker_use": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "jockey_sex": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "trainer_sex": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_0": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_1": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_2": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_3": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_4": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_5": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_6": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_7": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_8": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_9": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_10": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_11": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_12": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_13": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "track": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "running_style": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        # "rival_horse_id": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                    },
                    "mask": np.zeros((num_races, self.max_horses, self.max_hist_len), dtype=bool),
                },
                'jockey_history': {
                    "x_num": {
                        'days_before': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "horse_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "weight_change": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "distance": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "race_number": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "registration_count": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "horse_number": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "frame_number": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "horse_age": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "burden_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "finish_time": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "speed": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "last_3furlong": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "relative_horse_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "relative_burden_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "distance_change": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "rest_days": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                    },
                    "x_cat": {
                        "horse_id": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        'jockey_id': np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        'track_detail': np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        'track_type': np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "weather": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "track_cond": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "grade": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "course": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "sex": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "trainer_area": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "trainer_id": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "blinker_use": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "jockey_sex": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "trainer_sex": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_0": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_1": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_2": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_3": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_4": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_5": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_6": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_7": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_8": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_9": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_10": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_11": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_12": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_13": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "track": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "running_style": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        # "rival_horse_id": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                    },
                    "mask": np.zeros((num_races, self.max_horses, self.max_hist_len), dtype=bool),
                },
                'trainer_history': {
                    "x_num": {
                        'days_before': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "horse_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "weight_change": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "distance": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "race_number": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "registration_count": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "horse_number": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "frame_number": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "horse_age": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "burden_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "finish_time": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "speed": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "last_3furlong": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "relative_horse_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "relative_burden_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "distance_change": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "rest_days": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                    },
                    "x_cat": {
                        "horse_id": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "jockey_id": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        'track_detail': np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        'track_type': np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "weather": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "track_cond": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "grade": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "course": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "sex": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "trainer_area": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "trainer_id": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "blinker_use": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "jockey_sex": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "trainer_sex": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_0": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_1": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_2": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_3": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_4": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_5": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_6": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_7": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_8": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_9": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_10": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_11": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_12": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "pedigree_13": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "track": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        "running_style": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        # "rival_horse_id": np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                    },
                    "mask": np.zeros((num_races, self.max_horses, self.max_hist_len), dtype=bool),
                },
                  "course_lap_time_history": {
                    "x_num": {
                        'days_before': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        **{f"lap_time_{i}": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32) for i in range(25)}
                    },
                    "x_cat": {},
                    "mask": np.zeros((num_races, self.max_horses, self.max_hist_len), dtype=bool),
                },
                "training_history": {
                    "x_num": {
                        'days_before': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'furlong_10_total_time': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'furlong_9_total_time': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'furlong_8_total_time': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'furlong_7_total_time': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'furlong_6_total_time': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'furlong_5_total_time': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'furlong_4_total_time': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'furlong_3_total_time': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'furlong_2_total_time': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'lap_time_2000_1800': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'lap_time_1800_1600': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'lap_time_1600_1400': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'lap_time_1400_1200': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'lap_time_1200_1000': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'lap_time_1000_800': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'lap_time_800_600': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'lap_time_600_400': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'lap_time_400_200': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'lap_time_200_0': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                    },
                    "x_cat": {
                        'training_center': np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        'training_type': np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        'course': np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                        'track_direction': np.full((num_races, self.max_horses, self.max_hist_len), "<NULL>", dtype=object),
                    },
                    "mask": np.zeros((num_races, self.max_horses, self.max_hist_len), dtype=bool),
                }
            },
            "mask": np.zeros((num_races, self.max_horses), dtype=bool),
            "rankings": np.full((num_races, self.max_horses), 0, dtype=np.int64),  # 0は無効値
            'race_id': np.array(race_ids, dtype=object),
        }

        # ==========================================
        # 非シーケンスデータ部分
        # ==========================================
        
        # Numpyのインデックシング用
        race_indices = target_data['race_idx'].values
        horse_indices = target_data['horse_idx'].values

        # 数値特徴量の設定
        self.built_data['x_num']['horse_weight'][race_indices, horse_indices] = target_data['horse_weight_numeric'].values
        self.built_data['x_num']['weight_change'][race_indices, horse_indices] = target_data['weight_change_numeric'].values
        self.built_data['x_num']['distance'][race_indices, horse_indices] = target_data['distance_numeric'].values
        self.built_data['x_num']['race_number'][race_indices, horse_indices] = target_data['race_number_numeric'].values
        self.built_data['x_num']['registration_count'][race_indices, horse_indices] = target_data['registration_count_numeric'].values
        self.built_data['x_num']['horse_number'][race_indices, horse_indices] = target_data['horse_number_numeric'].values
        self.built_data['x_num']['frame_number'][race_indices, horse_indices] = target_data['frame_number_numeric'].values
        self.built_data['x_num']['horse_age'][race_indices, horse_indices] = target_data['horse_age_numeric'].values
        self.built_data['x_num']['burden_weight'][race_indices, horse_indices] = target_data['burden_weight_numeric'].values
        self.built_data['x_num']['relative_horse_weight'][race_indices, horse_indices] = target_data['relative_horse_weight_numeric'].values
        self.built_data['x_num']['relative_burden_weight'][race_indices, horse_indices] = target_data['relative_burden_weight_numeric'].values
        self.built_data['x_num']['distance_change'][race_indices, horse_indices] = target_data['distance_change_numeric'].values
        self.built_data['x_num']['rest_days'][race_indices, horse_indices] = target_data['rest_days_numeric'].values
        
        # カテゴリ特徴量の設定
        self.built_data['x_cat']['horse_id'][race_indices, horse_indices] = target_data['horse_id_valid'].values
        self.built_data['x_cat']['jockey_id'][race_indices, horse_indices] = target_data['jockey_id_valid'].values
        self.built_data['x_cat']['track_detail'][race_indices, horse_indices] = target_data['track_detail_valid'].values
        self.built_data['x_cat']['track_type'][race_indices, horse_indices] = target_data['track_type_valid'].values
        self.built_data['x_cat']['weather'][race_indices, horse_indices] = target_data['weather_valid'].values
        self.built_data['x_cat']['track_cond'][race_indices, horse_indices] = target_data['track_cond_valid'].values
        self.built_data['x_cat']['grade'][race_indices, horse_indices] = target_data['grade_valid'].values
        self.built_data['x_cat']['course'][race_indices, horse_indices] = target_data['course_valid'].values
        self.built_data['x_cat']['sex'][race_indices, horse_indices] = target_data['sex_valid'].values
        self.built_data['x_cat']['trainer_area'][race_indices, horse_indices] = target_data['trainer_area_valid'].values
        self.built_data['x_cat']['trainer_id'][race_indices, horse_indices] = target_data['trainer_id_valid'].values
        self.built_data['x_cat']['blinker_use'][race_indices, horse_indices] = target_data['blinker_use_valid'].values
        self.built_data['x_cat']['jockey_sex'][race_indices, horse_indices] = target_data['jockey_sex_valid'].values
        self.built_data['x_cat']['trainer_sex'][race_indices, horse_indices] = target_data['trainer_sex_valid'].values
        self.built_data['x_cat']['pedigree_0'][race_indices, horse_indices] = target_data['pedigree_0_valid'].values
        self.built_data['x_cat']['pedigree_1'][race_indices, horse_indices] = target_data['pedigree_1_valid'].values
        self.built_data['x_cat']['pedigree_2'][race_indices, horse_indices] = target_data['pedigree_2_valid'].values
        self.built_data['x_cat']['pedigree_3'][race_indices, horse_indices] = target_data['pedigree_3_valid'].values
        self.built_data['x_cat']['pedigree_4'][race_indices, horse_indices] = target_data['pedigree_4_valid'].values
        self.built_data['x_cat']['pedigree_5'][race_indices, horse_indices] = target_data['pedigree_5_valid'].values
        self.built_data['x_cat']['pedigree_6'][race_indices, horse_indices] = target_data['pedigree_6_valid'].values
        self.built_data['x_cat']['pedigree_7'][race_indices, horse_indices] = target_data['pedigree_7_valid'].values
        self.built_data['x_cat']['pedigree_8'][race_indices, horse_indices] = target_data['pedigree_8_valid'].values
        self.built_data['x_cat']['pedigree_9'][race_indices, horse_indices] = target_data['pedigree_9_valid'].values
        self.built_data['x_cat']['pedigree_10'][race_indices, horse_indices] = target_data['pedigree_10_valid'].values
        self.built_data['x_cat']['pedigree_11'][race_indices, horse_indices] = target_data['pedigree_11_valid'].values
        self.built_data['x_cat']['pedigree_12'][race_indices, horse_indices] = target_data['pedigree_12_valid'].values
        self.built_data['x_cat']['pedigree_13'][race_indices, horse_indices] = target_data['pedigree_13_valid'].values
        self.built_data['x_cat']['track'][race_indices, horse_indices] = target_data['track_valid'].values

        # ターゲットの設定
        self.built_data['rankings'][race_indices, horse_indices] = target_data['ranking_int'].values

        # マスクの設定
        self.built_data['mask'][race_indices, horse_indices] = True

        # ==========================================
        # シーケンスデータ部分
        # ==========================================

        # グループ化を事前に実行し、辞書として保存
        horse_groups = {horse_id: group.sort_values('kaisai_start_datetime', ascending=False) 
                    for horse_id, group in data.groupby('horse_id_valid') if horse_id != "<NULL>"}
        
        jockey_groups = {jockey_id: group.sort_values('kaisai_start_datetime', ascending=False)
                        for jockey_id, group in data.groupby('jockey_id_valid') if jockey_id != "<NULL>"}
        
        trainer_groups = {trainer_id: group.sort_values('kaisai_start_datetime', ascending=False)
                         for trainer_id, group in data.groupby('trainer_id_valid') if trainer_id != "<NULL>"}
        
        course_groups = {f"{track}_{track_detail}_{course}": group.sort_values('kaisai_start_datetime', ascending=False)
                        for (track, track_detail, course), group in data.groupby(['track_valid', 'track_detail_valid', 'course_valid'])
                        if track != "<NULL>" and track_detail != "<NULL>"}
        
        training_horse_groups = {horse_id: group.sort_values('training_datetime', ascending=False)
                            for horse_id, group in self.additional_prepared_data['training'].groupby('horse_id_valid')  
                            if horse_id != "<NULL>"}

        # 進捗報告の間隔
        progress_interval = 1000

        # レース情報をnumpy配列として事前取得
        race_datetimes = race_info['kaisai_start_datetime'].values

        for race_idx, (race_id, race_datetime) in enumerate(zip(race_ids, race_datetimes)):
            # 進捗ログ出力
            if (race_idx + 1) % progress_interval == 0 or (race_idx + 1) == num_races:
                progress_pct = (race_idx + 1) / num_races * 100
                logger.info(f"Processing race {race_idx + 1}/{num_races} ({progress_pct:.1f}%): {race_id}")

            # 現在のレースデータを取得
            race_data = target_data[target_data['race_idx'] == race_idx]

            cutoff_start = race_datetime - np.timedelta64(self.max_prev_days, 'D')
            cutoff_end = race_datetime - np.timedelta64(self.hours_before_race, 'h')

            # horse_history
            for _, row in race_data.iterrows():
                horse_idx = row['horse_idx']
                horse_key = row['horse_id_valid']
                if horse_key in horse_groups:
                    history = horse_groups[horse_key]
                    mask = (history['kaisai_start_datetime'] >= cutoff_start) & (history['kaisai_start_datetime'] < cutoff_end)
                    valid_history = history[mask].head(self.max_hist_len)
                    if len(valid_history) > 0:
                        days_before = ((race_datetime - valid_history['kaisai_start_datetime']).dt.total_seconds() / 86400).astype(np.float32)
                        length = len(valid_history)
                        self.built_data['sequence_data']['horse_history']['x_num']['days_before'][race_idx, horse_idx, :length] = days_before
                        self.built_data['sequence_data']['horse_history']['x_num']['horse_weight'][race_idx, horse_idx, :length] = valid_history['horse_weight_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['weight_change'][race_idx, horse_idx, :length] = valid_history['weight_change_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['distance'][race_idx, horse_idx, :length] = valid_history['distance_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['race_number'][race_idx, horse_idx, :length] = valid_history['race_number_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['registration_count'][race_idx, horse_idx, :length] = valid_history['registration_count_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['horse_number'][race_idx, horse_idx, :length] = valid_history['horse_number_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['frame_number'][race_idx, horse_idx, :length] = valid_history['frame_number_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['horse_age'][race_idx, horse_idx, :length] = valid_history['horse_age_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['burden_weight'][race_idx, horse_idx, :length] = valid_history['burden_weight_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['finish_time'][race_idx, horse_idx, :length] = valid_history['finish_time_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['speed'][race_idx, horse_idx, :length] = valid_history['speed_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['last_3furlong'][race_idx, horse_idx, :length] = valid_history['last_3furlong_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['relative_horse_weight'][race_idx, horse_idx, :length] = valid_history['relative_horse_weight_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['relative_burden_weight'][race_idx, horse_idx, :length] = valid_history['relative_burden_weight_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['distance_change'][race_idx, horse_idx, :length] = valid_history['distance_change_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['rest_days'][race_idx, horse_idx, :length] = valid_history['rest_days_numeric'].values
                        
                        self.built_data['sequence_data']['horse_history']['x_cat']['horse_id'][race_idx, horse_idx, :length] = valid_history['horse_id_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['jockey_id'][race_idx, horse_idx, :length] = valid_history['jockey_id_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['track_detail'][race_idx, horse_idx, :length] = valid_history['track_detail_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['track_type'][race_idx, horse_idx, :length] = valid_history['track_type_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['weather'][race_idx, horse_idx, :length] = valid_history['weather_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['track_cond'][race_idx, horse_idx, :length] = valid_history['track_cond_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['grade'][race_idx, horse_idx, :length] = valid_history['grade_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['course'][race_idx, horse_idx, :length] = valid_history['course_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['sex'][race_idx, horse_idx, :length] = valid_history['sex_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['trainer_area'][race_idx, horse_idx, :length] = valid_history['trainer_area_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['trainer_id'][race_idx, horse_idx, :length] = valid_history['trainer_id_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['blinker_use'][race_idx, horse_idx, :length] = valid_history['blinker_use_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['jockey_sex'][race_idx, horse_idx, :length] = valid_history['jockey_sex_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['trainer_sex'][race_idx, horse_idx, :length] = valid_history['trainer_sex_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_0'][race_idx, horse_idx, :length] = valid_history['pedigree_0_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_1'][race_idx, horse_idx, :length] = valid_history['pedigree_1_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_2'][race_idx, horse_idx, :length] = valid_history['pedigree_2_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_3'][race_idx, horse_idx, :length] = valid_history['pedigree_3_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_4'][race_idx, horse_idx, :length] = valid_history['pedigree_4_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_5'][race_idx, horse_idx, :length] = valid_history['pedigree_5_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_6'][race_idx, horse_idx, :length] = valid_history['pedigree_6_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_7'][race_idx, horse_idx, :length] = valid_history['pedigree_7_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_8'][race_idx, horse_idx, :length] = valid_history['pedigree_8_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_9'][race_idx, horse_idx, :length] = valid_history['pedigree_9_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_10'][race_idx, horse_idx, :length] = valid_history['pedigree_10_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_11'][race_idx, horse_idx, :length] = valid_history['pedigree_11_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_12'][race_idx, horse_idx, :length] = valid_history['pedigree_12_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['pedigree_13'][race_idx, horse_idx, :length] = valid_history['pedigree_13_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['track'][race_idx, horse_idx, :length] = valid_history['track_valid'].values
                        self.built_data['sequence_data']['horse_history']['x_cat']['running_style'][race_idx, horse_idx, :length] = valid_history['running_style_valid'].values
                        # self.built_data['sequence_data']['horse_history']['x_cat']['rival_horse_id'][race_idx, horse_idx, :length] = valid_history['rival_horse_id_valid'].values
                        
                        self.built_data['sequence_data']['horse_history']['mask'][race_idx, horse_idx, :length] = True

            # jockey_history
            for _, row in race_data.iterrows():
                horse_idx = row['horse_idx']
                jockey_key = row['jockey_id_valid']
                if jockey_key in jockey_groups:
                    history = jockey_groups[jockey_key]
                    mask = (history['kaisai_start_datetime'] >= cutoff_start) & (history['kaisai_start_datetime'] < cutoff_end)
                    valid_history = history[mask].head(self.max_hist_len)
                    if len(valid_history) > 0:
                        days_before = ((race_datetime - valid_history['kaisai_start_datetime']).dt.total_seconds() / 86400).astype(np.float32)
                        length = len(valid_history)
                        self.built_data['sequence_data']['jockey_history']['x_num']['days_before'][race_idx, horse_idx, :length] = days_before
                        self.built_data['sequence_data']['jockey_history']['x_num']['horse_weight'][race_idx, horse_idx, :length] = valid_history['horse_weight_numeric'].values
                        self.built_data['sequence_data']['jockey_history']['x_num']['weight_change'][race_idx, horse_idx, :length] = valid_history['weight_change_numeric'].values
                        self.built_data['sequence_data']['jockey_history']['x_num']['distance'][race_idx, horse_idx, :length] = valid_history['distance_numeric'].values
                        self.built_data['sequence_data']['jockey_history']['x_num']['race_number'][race_idx, horse_idx, :length] = valid_history['race_number_numeric'].values
                        self.built_data['sequence_data']['jockey_history']['x_num']['registration_count'][race_idx, horse_idx, :length] = valid_history['registration_count_numeric'].values
                        self.built_data['sequence_data']['jockey_history']['x_num']['horse_number'][race_idx, horse_idx, :length] = valid_history['horse_number_numeric'].values
                        self.built_data['sequence_data']['jockey_history']['x_num']['frame_number'][race_idx, horse_idx, :length] = valid_history['frame_number_numeric'].values
                        self.built_data['sequence_data']['jockey_history']['x_num']['horse_age'][race_idx, horse_idx, :length] = valid_history['horse_age_numeric'].values
                        self.built_data['sequence_data']['jockey_history']['x_num']['burden_weight'][race_idx, horse_idx, :length] = valid_history['burden_weight_numeric'].values
                        self.built_data['sequence_data']['jockey_history']['x_num']['finish_time'][race_idx, horse_idx, :length] = valid_history['finish_time_numeric'].values
                        self.built_data['sequence_data']['jockey_history']['x_num']['speed'][race_idx, horse_idx, :length] = valid_history['speed_numeric'].values
                        self.built_data['sequence_data']['jockey_history']['x_num']['last_3furlong'][race_idx, horse_idx, :length] = valid_history['last_3furlong_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['relative_horse_weight'][race_idx, horse_idx, :length] = valid_history['relative_horse_weight_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['relative_burden_weight'][race_idx, horse_idx, :length] = valid_history['relative_burden_weight_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['distance_change'][race_idx, horse_idx, :length] = valid_history['distance_change_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['rest_days'][race_idx, horse_idx, :length] = valid_history['rest_days_numeric'].values
                        
                        self.built_data['sequence_data']['jockey_history']['x_cat']['horse_id'][race_idx, horse_idx, :length] = valid_history['horse_id_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['jockey_id'][race_idx, horse_idx, :length] = valid_history['jockey_id_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['track_detail'][race_idx, horse_idx, :length] = valid_history['track_detail_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['track_type'][race_idx, horse_idx, :length] = valid_history['track_type_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['weather'][race_idx, horse_idx, :length] = valid_history['weather_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['track_cond'][race_idx, horse_idx, :length] = valid_history['track_cond_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['grade'][race_idx, horse_idx, :length] = valid_history['grade_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['course'][race_idx, horse_idx, :length] = valid_history['course_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['sex'][race_idx, horse_idx, :length] = valid_history['sex_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['trainer_area'][race_idx, horse_idx, :length] = valid_history['trainer_area_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['trainer_id'][race_idx, horse_idx, :length] = valid_history['trainer_id_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['blinker_use'][race_idx, horse_idx, :length] = valid_history['blinker_use_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['jockey_sex'][race_idx, horse_idx, :length] = valid_history['jockey_sex_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['trainer_sex'][race_idx, horse_idx, :length] = valid_history['trainer_sex_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_0'][race_idx, horse_idx, :length] = valid_history['pedigree_0_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_1'][race_idx, horse_idx, :length] = valid_history['pedigree_1_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_2'][race_idx, horse_idx, :length] = valid_history['pedigree_2_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_3'][race_idx, horse_idx, :length] = valid_history['pedigree_3_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_4'][race_idx, horse_idx, :length] = valid_history['pedigree_4_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_5'][race_idx, horse_idx, :length] = valid_history['pedigree_5_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_6'][race_idx, horse_idx, :length] = valid_history['pedigree_6_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_7'][race_idx, horse_idx, :length] = valid_history['pedigree_7_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_8'][race_idx, horse_idx, :length] = valid_history['pedigree_8_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_9'][race_idx, horse_idx, :length] = valid_history['pedigree_9_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_10'][race_idx, horse_idx, :length] = valid_history['pedigree_10_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_11'][race_idx, horse_idx, :length] = valid_history['pedigree_11_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_12'][race_idx, horse_idx, :length] = valid_history['pedigree_12_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['pedigree_13'][race_idx, horse_idx, :length] = valid_history['pedigree_13_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['track'][race_idx, horse_idx, :length] = valid_history['track_valid'].values
                        self.built_data['sequence_data']['jockey_history']['x_cat']['running_style'][race_idx, horse_idx, :length] = valid_history['running_style_valid'].values
                        # self.built_data['sequence_data']['jockey_history']['x_cat']['rival_horse_id'][race_idx, horse_idx, :length] = valid_history['rival_horse_id_valid'].values
                        
                        self.built_data['sequence_data']['jockey_history']['mask'][race_idx, horse_idx, :length] = True

            # trainer_history
            for _, row in race_data.iterrows():
                horse_idx = row['horse_idx']
                trainer_key = row['trainer_id_valid']
                if trainer_key in trainer_groups:
                    history = trainer_groups[trainer_key]
                    mask = (history['kaisai_start_datetime'] >= cutoff_start) & (history['kaisai_start_datetime'] < cutoff_end)
                    valid_history = history[mask].head(self.max_hist_len)
                    if len(valid_history) > 0:
                        days_before = ((race_datetime - valid_history['kaisai_start_datetime']).dt.total_seconds() / 86400).astype(np.float32)
                        length = len(valid_history)
                        self.built_data['sequence_data']['trainer_history']['x_num']['days_before'][race_idx, horse_idx, :length] = days_before
                        self.built_data['sequence_data']['trainer_history']['x_num']['horse_weight'][race_idx, horse_idx, :length] = valid_history['horse_weight_numeric'].values
                        self.built_data['sequence_data']['trainer_history']['x_num']['weight_change'][race_idx, horse_idx, :length] = valid_history['weight_change_numeric'].values
                        self.built_data['sequence_data']['trainer_history']['x_num']['distance'][race_idx, horse_idx, :length] = valid_history['distance_numeric'].values
                        self.built_data['sequence_data']['trainer_history']['x_num']['race_number'][race_idx, horse_idx, :length] = valid_history['race_number_numeric'].values
                        self.built_data['sequence_data']['trainer_history']['x_num']['registration_count'][race_idx, horse_idx, :length] = valid_history['registration_count_numeric'].values
                        self.built_data['sequence_data']['trainer_history']['x_num']['horse_number'][race_idx, horse_idx, :length] = valid_history['horse_number_numeric'].values
                        self.built_data['sequence_data']['trainer_history']['x_num']['frame_number'][race_idx, horse_idx, :length] = valid_history['frame_number_numeric'].values
                        self.built_data['sequence_data']['trainer_history']['x_num']['horse_age'][race_idx, horse_idx, :length] = valid_history['horse_age_numeric'].values
                        self.built_data['sequence_data']['trainer_history']['x_num']['burden_weight'][race_idx, horse_idx, :length] = valid_history['burden_weight_numeric'].values
                        self.built_data['sequence_data']['trainer_history']['x_num']['finish_time'][race_idx, horse_idx, :length] = valid_history['finish_time_numeric'].values
                        self.built_data['sequence_data']['trainer_history']['x_num']['speed'][race_idx, horse_idx, :length] = valid_history['speed_numeric'].values
                        self.built_data['sequence_data']['trainer_history']['x_num']['last_3furlong'][race_idx, horse_idx, :length] = valid_history['last_3furlong_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['relative_horse_weight'][race_idx, horse_idx, :length] = valid_history['relative_horse_weight_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['relative_burden_weight'][race_idx, horse_idx, :length] = valid_history['relative_burden_weight_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['distance_change'][race_idx, horse_idx, :length] = valid_history['distance_change_numeric'].values
                        self.built_data['sequence_data']['horse_history']['x_num']['rest_days'][race_idx, horse_idx, :length] = valid_history['rest_days_numeric'].values
                        
                        self.built_data['sequence_data']['trainer_history']['x_cat']['horse_id'][race_idx, horse_idx, :length] = valid_history['horse_id_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['jockey_id'][race_idx, horse_idx, :length] = valid_history['jockey_id_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['track_detail'][race_idx, horse_idx, :length] = valid_history['track_detail_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['track_type'][race_idx, horse_idx, :length] = valid_history['track_type_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['weather'][race_idx, horse_idx, :length] = valid_history['weather_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['track_cond'][race_idx, horse_idx, :length] = valid_history['track_cond_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['grade'][race_idx, horse_idx, :length] = valid_history['grade_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['course'][race_idx, horse_idx, :length] = valid_history['course_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['sex'][race_idx, horse_idx, :length] = valid_history['sex_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['trainer_area'][race_idx, horse_idx, :length] = valid_history['trainer_area_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['trainer_id'][race_idx, horse_idx, :length] = valid_history['trainer_id_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['blinker_use'][race_idx, horse_idx, :length] = valid_history['blinker_use_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['jockey_sex'][race_idx, horse_idx, :length] = valid_history['jockey_sex_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['trainer_sex'][race_idx, horse_idx, :length] = valid_history['trainer_sex_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_0'][race_idx, horse_idx, :length] = valid_history['pedigree_0_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_1'][race_idx, horse_idx, :length] = valid_history['pedigree_1_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_2'][race_idx, horse_idx, :length] = valid_history['pedigree_2_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_3'][race_idx, horse_idx, :length] = valid_history['pedigree_3_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_4'][race_idx, horse_idx, :length] = valid_history['pedigree_4_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_5'][race_idx, horse_idx, :length] = valid_history['pedigree_5_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_6'][race_idx, horse_idx, :length] = valid_history['pedigree_6_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_7'][race_idx, horse_idx, :length] = valid_history['pedigree_7_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_8'][race_idx, horse_idx, :length] = valid_history['pedigree_8_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_9'][race_idx, horse_idx, :length] = valid_history['pedigree_9_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_10'][race_idx, horse_idx, :length] = valid_history['pedigree_10_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_11'][race_idx, horse_idx, :length] = valid_history['pedigree_11_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_12'][race_idx, horse_idx, :length] = valid_history['pedigree_12_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['pedigree_13'][race_idx, horse_idx, :length] = valid_history['pedigree_13_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['track'][race_idx, horse_idx, :length] = valid_history['track_valid'].values
                        self.built_data['sequence_data']['trainer_history']['x_cat']['running_style'][race_idx, horse_idx, :length] = valid_history['running_style_valid'].values
                        # self.built_data['sequence_data']['trainer_history']['x_cat']['rival_horse_id'][race_idx, horse_idx, :length] = valid_history['rival_horse_id_valid'].values

                        self.built_data['sequence_data']['trainer_history']['mask'][race_idx, horse_idx, :length] = True

            # course_lap_time_history
            for _, row in race_data.iterrows():
                horse_idx = row['horse_idx']
                course_key = f"{row['track_valid']}_{row['track_detail_valid']}_{row['course_valid']}"
                if course_key in course_groups:
                    history = course_groups[course_key]
                    mask = (history['kaisai_start_datetime'] >= cutoff_start) & (history['kaisai_start_datetime'] < cutoff_end)
                    valid_history = history[mask].head(self.max_hist_len)
                    if len(valid_history) > 0:
                        days_before = ((race_datetime - valid_history['kaisai_start_datetime']).dt.total_seconds() / 86400).astype(np.float32)
                        length = len(valid_history)
                        self.built_data['sequence_data']['course_lap_time_history']['x_num']['days_before'][race_idx, horse_idx, :length] = days_before
                        for i in range(25):
                            self.built_data['sequence_data']['course_lap_time_history']['x_num'][f'lap_time_{i}'][race_idx, horse_idx, :length] = valid_history[f'lap_time_{i}_numeric'].values
                        self.built_data['sequence_data']['course_lap_time_history']['mask'][race_idx, horse_idx, :length] = True

            # training_history
            for _, row in race_data.iterrows():
                horse_idx = row['horse_idx']
                training_horse_key = row['horse_id_valid']
                if training_horse_key in training_horse_groups:
                    history = training_horse_groups[training_horse_key]
                    mask = (history['training_datetime'] >= cutoff_start) & (history['training_datetime'] < cutoff_end)
                    valid_history = history[mask].head(self.max_hist_len)
                    if len(valid_history) > 0:
                        days_before = ((race_datetime - valid_history['training_datetime']).dt.total_seconds() / 86400).astype(np.float32)
                        length = len(valid_history)
                        self.built_data['sequence_data']['training_history']['x_num']['days_before'][race_idx, horse_idx, :length] = days_before
                        self.built_data['sequence_data']['training_history']['x_num']['furlong_10_total_time'][race_idx, horse_idx, :length] = valid_history[f'furlong_10_total_time_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['furlong_9_total_time'][race_idx, horse_idx, :length] = valid_history[f'furlong_9_total_time_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['furlong_8_total_time'][race_idx, horse_idx, :length] = valid_history[f'furlong_8_total_time_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['furlong_7_total_time'][race_idx, horse_idx, :length] = valid_history[f'furlong_7_total_time_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['furlong_6_total_time'][race_idx, horse_idx, :length] = valid_history[f'furlong_6_total_time_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['furlong_5_total_time'][race_idx, horse_idx, :length] = valid_history[f'furlong_5_total_time_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['furlong_4_total_time'][race_idx, horse_idx, :length] = valid_history[f'furlong_4_total_time_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['furlong_3_total_time'][race_idx, horse_idx, :length] = valid_history[f'furlong_3_total_time_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['furlong_2_total_time'][race_idx, horse_idx, :length] = valid_history[f'furlong_2_total_time_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['lap_time_2000_1800'][race_idx, horse_idx, :length] = valid_history[f'lap_time_2000_1800_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['lap_time_1800_1600'][race_idx, horse_idx, :length] = valid_history[f'lap_time_1800_1600_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['lap_time_1600_1400'][race_idx, horse_idx, :length] = valid_history[f'lap_time_1600_1400_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['lap_time_1400_1200'][race_idx, horse_idx, :length] = valid_history[f'lap_time_1400_1200_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['lap_time_1200_1000'][race_idx, horse_idx, :length] = valid_history[f'lap_time_1200_1000_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['lap_time_1000_800'][race_idx, horse_idx, :length] = valid_history[f'lap_time_1000_800_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['lap_time_800_600'][race_idx, horse_idx, :length] = valid_history[f'lap_time_800_600_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['lap_time_600_400'][race_idx, horse_idx, :length] = valid_history[f'lap_time_600_400_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['lap_time_400_200'][race_idx, horse_idx, :length] = valid_history[f'lap_time_400_200_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_num']['lap_time_200_0'][race_idx, horse_idx, :length] = valid_history[f'lap_time_200_0_numeric'].values
                        self.built_data['sequence_data']['training_history']['x_cat']['training_center'][race_idx, horse_idx, :length] = valid_history['training_center_valid'].values
                        self.built_data['sequence_data']['training_history']['x_cat']['training_type'][race_idx, horse_idx, :length] = valid_history['training_type_valid'].values
                        self.built_data['sequence_data']['training_history']['x_cat']['course'][race_idx, horse_idx, :length] = valid_history['course_valid'].values
                        self.built_data['sequence_data']['training_history']['x_cat']['track_direction'][race_idx, horse_idx, :length] = valid_history['track_direction_valid'].values
                        self.built_data['sequence_data']['training_history']['mask'][race_idx, horse_idx, :length] = True
            
        logger.info("Data structure construction completed successfully.")
        return self

    def fit(self):
        self.params = {
            'numerical': {}, 
            'categorical': {}, 
        }

        # 特徴量の収集
        feature_values = {'numerical': defaultdict(lambda: np.full(0, np.nan, dtype=np.float32)), 'categorical': defaultdict(lambda: np.full(0, "<NULL>", dtype=object))}
        for name, data in self.built_data['x_num'].items():
            alias = self.feature_aliases.get(name, name)
            feature_values['numerical'][alias] = np.concatenate([feature_values['numerical'][alias], data.reshape(-1)])
        for name, data in self.built_data['x_cat'].items():
            alias = self.feature_aliases.get(name, name)
            feature_values['categorical'][alias] = np.concatenate([feature_values['categorical'][alias], data.reshape(-1)])
        for seq_data in self.built_data['sequence_data'].values():
            for name, data in seq_data['x_num'].items():
                alias = self.feature_aliases.get(name, name)
                feature_values['numerical'][alias] = np.concatenate([feature_values['numerical'][alias], data.reshape(-1)])
            for name, data in seq_data['x_cat'].items():
                alias = self.feature_aliases.get(name, name)
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
                alias = self.feature_aliases.get(name, name)
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
                alias = self.feature_aliases.get(name, name)
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

