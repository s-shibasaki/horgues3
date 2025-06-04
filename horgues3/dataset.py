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
        SELECT 
            hri.kaisai_year || hri.kaisai_month_day || hri.track_code || hri.kaisai_kai || hri.kaisai_day || hri.race_number as race_id,
            hri.horse_number,

            -- メタデータ
            hri.kaisai_year || hri.kaisai_month_day as kaisai_ymd,
            rd.start_time,

            -- 数値特徴量
            hri.horse_weight,
            hri.weight_change_sign,
            hri.weight_change,
            rd.distance,

            -- カテゴリ特徴量
            hri.blood_registration_number,
            hri.jockey_code,
            rd.weather_code,
            CASE 
                WHEN 
                    rd.track_code_detail BETWEEN '10' AND '22' OR
                    rd.track_code_detail = '51' OR
                    rd.track_code BETWEEN '53' AND '59'
                THEN
                    rd.turf_condition_code 
                WHEN
                    rd.track_code_detail BETWEEN '23' AND '29' OR
                    rd.track_code_detail = '52'
                THEN
                    rd.dirt_condition_code
                ELSE
                    '0'
            END track_condition_code,

            -- ターゲット
            hri.final_order
        FROM public.horse_race_info hri
        INNER JOIN public.race_detail rd ON (
            hri.kaisai_year = rd.kaisai_year AND
            hri.kaisai_month_day = rd.kaisai_month_day AND
            hri.track_code = rd.track_code AND
            hri.kaisai_kai = rd.kaisai_kai AND
            hri.kaisai_day = rd.kaisai_day AND
            hri.race_number = rd.race_number
        )
        WHERE 1=1 
            AND hri.horse_number BETWEEN '01' AND '{self.max_horses:02}'
            AND hri.final_order BETWEEN '01' AND '{self.max_horses:02}'
            AND (
                hri.kaisai_year > '{start_year}'
                OR (hri.kaisai_year = '{start_year}' AND hri.kaisai_month_day >= '{start_month_day}')
            )
            AND (
                hri.kaisai_year < '{end_year}'
                OR (hri.kaisai_year = '{end_year}' AND hri.kaisai_month_day <= '{end_month_day}')
            )
        ORDER BY race_id, hri.horse_number;
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

        # kaisai_ymd: 開催年月日をdatetime型に変換
        data['kaisai_ymd_parsed'] = pd.to_datetime(data['kaisai_ymd'], format='%Y%m%d', errors='coerce').dt.date
        kaisai_ymd_mask = data['kaisai_ymd'] != '00000000'
        data.loc[~kaisai_ymd_mask, 'kaisai_ymd_parsed'] = pd.NaT
        
        # start_time: 発走時刻をtime型に変換 (HHMM形式を想定)
        data['start_time_parsed'] = pd.to_datetime(data['start_time'], format='%H%M', errors='coerce').dt.time
        start_time_mask = data['start_time'] != '0000'
        data.loc[~start_time_mask, 'start_time_parsed'] = pd.NaT

        # ==========================================
        # 数値特徴量
        # ==========================================

        # horse_weight: 馬体重
        data['horse_weight_numeric'] = pd.to_numeric(data['horse_weight'], errors='coerce').astype(np.float32)
        horse_weight_mask = data['horse_weight_numeric'].between(2, 998)  # 馬体重は2kgから998kgの範囲
        data.loc[~horse_weight_mask, 'horse_weight_numeric'] = np.nan

        # weight_change: 増減差
        data['weight_change_numeric'] = pd.to_numeric(data['weight_change_sign'] + data['weight_change'], errors='coerce').astype(np.float32)
        weight_change_mask = data['weight_change_numeric'].between(-998, 998)  # 増減差は-998から998の範囲
        data.loc[~weight_change_mask, 'weight_change_numeric'] = np.nan

        # distance: 距離
        data['distance_numeric'] = pd.to_numeric(data['distance'], errors='coerce').astype(np.float32)
        data.loc[data['distance_numeric'] == 0, 'distance_numeric'] = np.nan  # 距離が0のレコードは無効とする

        # ==========================================
        # カテゴリ特徴量
        # ==========================================

        # blood_registration_number: 血統登録番号
        data['blood_registration_number_valid'] = data['blood_registration_number'].fillna("<NULL>").astype(str)
        data.loc[data['blood_registration_number_valid'] == "0000000000", 'blood_registration_number_valid'] = "<NULL>"  # 0000000000は無効とする

        # jockey_code: 騎手コード
        data['jockey_code_valid'] = data['jockey_code'].fillna("<NULL>").astype(str)
        data.loc[data['jockey_code_valid'] == "00000", 'jockey_code_valid'] = "<NULL>"  # 00000は無効とする

        # weather_code: 天候コード
        data['weather_code_valid'] = data['weather_code'].fillna("<NULL>").astype(str)
        data.loc[data['weather_code_valid'] == "0", 'weather_code_valid'] = "<NULL>"  # 0は無効とする

        # track_condition_code: 馬場状態コード
        data['track_condition_code_valid'] = data['track_condition_code'].fillna("<NULL>").astype(str)
        data.loc[data['track_condition_code_valid'] == "0", 'track_condition_code_valid'] = "<NULL>"  # 0は無効とする

        # ==========================================
        # ターゲット
        # ==========================================

        # final_order: 最終着順
        data['final_order_numeric'] = pd.to_numeric(data['final_order'], errors='coerce').astype(np.int64) - 1  # 0-indexedに変換
        final_order_mask = data['final_order_numeric'].between(0, self.max_horses - 1)  # 0からnum_horses-1の範囲
        data.loc[~final_order_mask, 'final_order_numeric'] = -1  # 無効な値は-1に設定

        self.prepared_data = data
        logger.info(f"Prepared data with {len(self.prepared_data)} records.")

        return self

    def build(self):
        """データ構造を構築する"""
        logger.info("Building data structure...")

        data = self.prepared_data
        target_data = data[data['kaisai_ymd_parsed'].between(self.start_date, self.end_date)]

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
            },
            "x_cat": {
                "blood_registration_number": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "jockey_code": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "weather_code": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
                "track_condition_code": np.full((num_races, self.max_horses), "<NULL>", dtype=object),
            },
            "sequence_data": {
                "horse_weight_history": {
                    "x_num": {
                        "horse_weight": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        "weight_change": np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                        'days_before': np.full((num_races, self.max_horses, self.max_hist_len), np.nan, dtype=np.float32),
                    },
                    "x_cat": {},
                    "mask": np.zeros((num_races, self.max_horses, self.max_hist_len), dtype=bool),
                }
            },
            "mask": np.zeros((num_races, self.max_horses), dtype=bool),
            "rankings": np.full((num_races, self.max_horses), -1, dtype=np.int64),
            'race_id': np.array(race_ids, dtype=object),
        }

        # シーケンスデータ構築に使用するグループ（ソート済みに変更）
        horse_groups = {}
        for name, group in data.groupby('blood_registration_number_valid'):
            # 各馬のデータを事前にソート（kaisai_ymd_parsed, start_time_parsed の降順）
            horse_groups[name] = group.sort_values(['kaisai_ymd_parsed', 'start_time_parsed'], ascending=False)

        # 進捗報告の間隔を設定
        progress_interval = min(1000, max(1, num_races // 10))  # 10%ごとに進捗報告

        # 各レースのデータを構築
        for race_idx, race_id in enumerate(race_ids):

            # 進捗ログ出力
            if (race_idx + 1) % progress_interval == 0 or (race_idx + 1) == num_races:
                progress_pct = (race_idx + 1) / num_races * 100
                logger.info(f"Processing race {race_idx + 1}/{num_races} ({progress_pct:.1f}%): {race_id}")

            race_data = race_groups.get_group(race_id)
            
            # シーケンスデータ構築のため開催日と発走時刻を取得（最初の行から一度だけ取得）
            first_row = race_data.iloc[0]
            current_date = first_row['kaisai_ymd_parsed']
            current_time = first_row['start_time_parsed']

            for _, row in race_data.iterrows():
                horse_num = int(row['horse_number']) - 1  # 0-indexedに変換

                # マスク
                self.built_data['mask'][race_idx, horse_num] = True

                # 数値特徴量
                self.built_data['x_num']['horse_weight'][race_idx, horse_num] = row['horse_weight_numeric']
                self.built_data['x_num']['weight_change'][race_idx, horse_num] = row['weight_change_numeric']
                self.built_data['x_num']['distance'][race_idx, horse_num] = row['distance_numeric']

                # カテゴリ特徴量
                self.built_data['x_cat']['blood_registration_number'][race_idx, horse_num] = row['blood_registration_number_valid']
                self.built_data['x_cat']['jockey_code'][race_idx, horse_num] = row['jockey_code_valid']
                self.built_data['x_cat']['weather_code'][race_idx, horse_num] = row['weather_code_valid']
                self.built_data['x_cat']['track_condition_code'][race_idx, horse_num] = row['track_condition_code_valid']

                # ターゲット
                self.built_data['rankings'][race_idx, horse_num] = row['final_order_numeric']

                # シーケンスデータ構築時に使用するキー
                horse_id = row['blood_registration_number_valid']  

                # 馬体重履歴データの構築
                if horse_id != "<NULL>" and horse_id in horse_groups:
                    history = horse_groups[horse_id]

                    # 過去のデータのみフィルタ（既にソート済みなので効率的）
                    valid_history = history[
                        (history['kaisai_ymd_parsed'] < current_date) | 
                        ((history['kaisai_ymd_parsed'] == current_date) & (history['start_time_parsed'] < current_time))
                    ].head(self.max_hist_len)  # 既にソート済みなのでheadで十分
                    
                    valid_length = len(valid_history)
                    if valid_length > 0:
                        days_before = np.array([(current_date - date).days for date in valid_history['kaisai_ymd_parsed']], dtype=np.float32)
                        self.built_data['sequence_data']['horse_weight_history']['x_num']['horse_weight'][race_idx, horse_num, :valid_length] = valid_history['horse_weight_numeric'].values
                        self.built_data['sequence_data']['horse_weight_history']['x_num']['weight_change'][race_idx, horse_num, :valid_length] = valid_history['weight_change_numeric'].values
                        self.built_data['sequence_data']['horse_weight_history']['x_num']['days_before'][race_idx, horse_num, :valid_length] = days_before
                        self.built_data['sequence_data']['horse_weight_history']['mask'][race_idx, horse_num, :valid_length] = True

        logger.info("Data structure construction completed successfully.")
        return self

    def fit(self):
        self.params = {'numerical': {}, 'categorical': {}, 'aliases': {}}  # エイリアスはここで設定する

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

