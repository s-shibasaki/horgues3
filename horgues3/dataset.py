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

logger = logging.getLogger(__name__)


class CustomLabelEncoder:
    """カスタムラベルエンコーダー（0をパディング値として予約）"""
    
    def __init__(self):
        self.classes_ = None
        self.class_to_index = {}
    
    def fit(self, y):
        """有効な値のみでフィット（NaN, None を無視）"""
        # 有効な値のみを抽出
        valid_values = []
        for value in y:
            if pd.isna(value) or value is None:
                continue
            valid_values.append(str(value))
        
        # ユニークな値を取得してソート
        unique_values = sorted(list(set(valid_values)))
        
        # クラス一覧を保存（0は予約済みなので1から開始）
        self.classes_ = unique_values
        self.class_to_index = {cls: idx + 1 for idx, cls in enumerate(unique_values)}
        
        return self
    
    def transform(self, y):
        """変換（NaN, None, 未知の値は0にマップ）"""
        if self.classes_ is None:
            raise ValueError("Encoder has not been fitted yet.")
        
        result = []
        for value in y:
            if pd.isna(value) or value is None:
                result.append(0)  # パディング値
            else:
                str_value = str(value)
                result.append(self.class_to_index.get(str_value, 0))  # 未知の値も0
        
        return np.array(result)
    
    def get_vocab_size(self):
        """語彙サイズを取得（パディング用の0を含む）"""
        if self.classes_ is None:
            return 1  # パディング値のみ
        return len(self.classes_) + 1  # 実際のクラス数 + パディング値


class HorguesDataset(Dataset):
    """レースデータセット - データ取得・前処理・データセット構築を統合"""
    
    def __init__(self, max_horses: int = 18):
        self.max_horses = max_horses
        
    def fetch(self, start_ymd: str, end_ymd: str):
        """レースデータを取得"""
        logger.info(f"Fetching data from {start_ymd} to {end_ymd}...")
        
        # ymd形式をyyyy, mmddに分割
        start_year = int(start_ymd[:4])
        start_month_day = start_ymd[4:]
        end_year = int(end_ymd[:4])
        end_month_day = end_ymd[4:]
        
        engine = create_engine("postgresql://postgres:postgres@localhost/horgues3")
        
        # 基本レースデータを取得するSQL
        query = f"""
        SELECT 
            hri.kaisai_year || hri.kaisai_month_day || hri.track_code || hri.kaisai_kai || hri.kaisai_day || hri.race_number as race_id,
            hri.horse_number,
            hri.horse_weight,
            hri.weight_change_sign,
            hri.weight_change,
            hri.final_order,
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
            rd.distance
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
        
        self.data = pd.read_sql_query(query, engine)
        logger.info(f"Retrieved {len(self.data)} records.")

        return self

    def process(self):
        """レースデータの後処理"""
        logger.info("Post-processing race data...")

        # 数値変換
        self.data['horse_number_numeric'] = pd.to_numeric(self.data['horse_number'], errors='coerce')
        self.data['final_order_numeric'] = pd.to_numeric(self.data['final_order'], errors='coerce')
        
        # 馬体重の処理（2～998の範囲外をNaN）
        self.data['horse_weight_numeric'] = pd.to_numeric(self.data['horse_weight'], errors='coerce')
        valid_weight_mask = (self.data['horse_weight_numeric'] >= 2) & (self.data['horse_weight_numeric'] <= 998)
        self.data.loc[~valid_weight_mask, 'horse_weight_numeric'] = np.nan
        
        # 増減差の処理（-998～998の範囲外をNaN）
        weight_change_str = self.data['weight_change_sign'].fillna('') + self.data['weight_change'].fillna('')
        self.data['weight_change_numeric'] = pd.to_numeric(weight_change_str, errors='coerce')
        valid_change_mask = (self.data['weight_change_numeric'] >= -998) & (self.data['weight_change_numeric'] <= 998)
        self.data.loc[~valid_change_mask, 'weight_change_numeric'] = np.nan
        
        # 距離の数値変換
        self.data['distance_numeric'] = pd.to_numeric(self.data['distance'], errors='coerce')

        return self
        
    
    def fit(self) -> 'HorguesDataset':
        """前処理器をフィット"""
        logger.info("Fitting preprocessors...")
        
        # 数値特徴量の前処理器
        numerical_features = [
            'horse_weight_numeric', 
            'weight_change_numeric',
            'distance_numeric'
        ]
        
        self.numerical_scalers = {}
        for feature in numerical_features:
            if feature in self.data.columns:
                scaler = StandardScaler()
                valid_data = self.data[feature].dropna().values.reshape(-1, 1)
                if len(valid_data) > 0:
                    scaler.fit(valid_data)
                    self.numerical_scalers[feature] = scaler
        
        # カテゴリ特徴量の前処理器
        categorical_features = [
            'weather_code',
            'track_condition_code', 
        ]
        
        self.categorical_encoders = {}
        for feature in categorical_features:
            if feature in self.data.columns:
                encoder = CustomLabelEncoder()
                encoder.fit(self.data[feature])
                self.categorical_encoders[feature] = encoder
        
        logger.info("Preprocessors fitted successfully.")
        return self
    
    def get_numerical_features(self) -> List[str]:
        """数値特徴量のリストを取得（正規化後の名前）"""
        return [f'{feature}_normalized' for feature in self.numerical_scalers.keys()]
    
    def get_categorical_features(self) -> Dict[str, int]:
        """カテゴリ特徴量の辞書を取得（エンコード後の名前とvocab_size）"""
        return {
            f'{feature}_encoded': encoder.get_vocab_size() 
            for feature, encoder in self.categorical_encoders.items()
        }
    
    def get_feature_configs(self) -> Dict[str, Any]:
        """モデル構築に必要な特徴量設定を取得"""
        return {
            'numerical_features': self.get_numerical_features(),
            'categorical_features': self.get_categorical_features(),
            'max_horses': self.max_horses
        }
    
    def transform(self):
        """データの前処理を実行"""
        logger.info("Applying preprocessing...")
        
        processed_data = self.data.copy()
        
        # 数値特徴量の標準化
        for feature, scaler in self.numerical_scalers.items():
            if feature in processed_data.columns:
                feature_data = processed_data[feature].values.reshape(-1, 1)
                normalized_values = scaler.transform(feature_data)
                processed_data[f'{feature}_normalized'] = normalized_values.flatten()
        
        # カテゴリ特徴量のエンコード
        for feature, encoder in self.categorical_encoders.items():
            if feature in processed_data.columns:
                encoded_values = encoder.transform(processed_data[feature])
                processed_data[f'{feature}_encoded'] = encoded_values
        
        self.data = processed_data
        return self
    
    def build_races(self):
        """レースサンプルを構築"""
        logger.info("Building race samples...")
        
        races = []
        
        for race_id, race_group in self.data.groupby('race_id'):
            race_sample = self._build_single_race(race_id, race_group)
            races.append(race_sample)
        
        logger.info(f"Built {len(races)} race samples.")
        self.races = races

        return self
    
    def _build_single_race(self, race_id: str, race_group: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """単一レースのデータを構築"""
        # 馬番を取得してインデックスに変換
        horse_numbers = race_group['horse_number_numeric'].values
        horse_indices = horse_numbers - 1  # 馬番-1をインデックスに
        
        # 有効な馬番のチェック
        if np.any((horse_indices < 0) | (horse_indices >= self.max_horses)):
            raise ValueError(f"Invalid horse numbers {horse_numbers} at race {race_id}")

        # 数値特徴量を構築
        x_num = {}
        for feature in self.get_numerical_features():
            if feature in race_group.columns:
                # NaNで初期化
                feature_array = np.full(self.max_horses, np.nan, dtype=np.float32)
                feature_array[horse_indices] = race_group[feature].values
                x_num[feature] = torch.tensor(feature_array, dtype=torch.float32)
        
        # カテゴリ特徴量を構築
        x_cat = {}
        for feature in list(self.get_categorical_features().keys()):
            if feature in race_group.columns:
                # 0で初期化（パディング値）
                feature_array = np.zeros(self.max_horses, dtype=np.int64)
                feature_array[horse_indices] = race_group[feature].values
                x_cat[feature] = torch.tensor(feature_array, dtype=torch.long)
        
        # 着順を構築
        rankings = np.full(self.max_horses, -1, dtype=np.int32)
        
        # 着順を設定（1位=0, 2位=1, ...）
        final_orders = race_group['final_order_numeric'].values
        rankings[horse_indices] = final_orders - 1
        
        # マスクを構築
        mask = np.zeros(self.max_horses, dtype=bool)
        mask[horse_indices] = True
        
        # メタデータを取得
        first_row = race_group.iloc[0]
        metadata = {
            'weather_code': first_row.get('weather_code'),
            'track_condition_code': first_row.get('track_condition_code'),
            'distance': first_row.get('distance_numeric'),
        }
        
        return {
            'race_id': race_id,
            'x_num': x_num,
            'x_cat': x_cat,
            'rankings': torch.tensor(rankings, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.bool),
            'num_horses': len(horse_indices),
            'metadata': metadata
        }
    
    def get_preprocessors(self) -> Dict[str, Any]:
        """前処理器を取得"""
        return {
            'numerical_scalers': self.numerical_scalers,
            'categorical_encoders': self.categorical_encoders
        }
    
    def set_preprocessors(self, preprocessors: Dict[str, Any]) -> 'HorguesDataset':
        """前処理器を設定"""
        self.numerical_scalers = preprocessors.get('numerical_scalers', {})
        self.categorical_encoders = preprocessors.get('categorical_encoders', {})
        return self
    
    def save_preprocessors(self, filepath: str):
        """前処理器を保存"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_preprocessors(), f)
        logger.info(f"Preprocessors saved to {filepath}")
    
    def load_preprocessors(self, filepath: str) -> 'HorguesDataset':
        """前処理器を読み込み"""
        with open(filepath, 'rb') as f:
            preprocessors = pickle.load(f)
        self.set_preprocessors(preprocessors)
        logger.info(f"Preprocessors loaded from {filepath}")
        return self
    
    def __len__(self):
        return len(self.races)
    
    def __getitem__(self, idx):
        return self.races[idx]

