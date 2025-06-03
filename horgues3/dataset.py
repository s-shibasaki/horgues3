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


logger = logging.getLogger(__name__)

class Horgues3Dataset(Dataset):
    def __init__(self, start_ymd, end_ymd, max_horses=18):
        super().__init__()
        self.max_horses = max_horses
        self.start_ymd = start_ymd
        self.end_ymd = end_ymd

        # ymd形式をyyyy, mmddに分割
        self.start_year = int(start_ymd[:4])
        self.start_month_day = start_ymd[4:]
        self.end_year = int(end_ymd[:4])
        self.end_month_day = end_ymd[4:]

        self.numerical_scalers = {}
        self.categorical_encoders = {}

    def fetch_data(self):
        logger.info("Fetching data from database...")
        engine = create_engine(f"postgresql://postgres:postgres@localhost/horgues3")

        # レースID、馬体重、確定着順を取得するSQL
        query = f"""
        SELECT 
            hri.kaisai_year || hri.kaisai_month_day || hri.track_code || hri.kaisai_kai || hri.kaisai_day || hri.race_number as race_id,
            hri.horse_number,
            hri.horse_weight,
            hri.weight_change_sign,
            hri.weight_change,
            hri.final_order,
            rd.weather_code
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
                hri.kaisai_year > '{self.start_year}'
                OR (hri.kaisai_year = '{self.start_year}' AND hri.kaisai_month_day >= '{self.start_month_day}')
            )
            AND (
                hri.kaisai_year < '{self.end_year}'
                OR (hri.kaisai_year = '{self.end_year}' AND hri.kaisai_month_day <= '{self.end_month_day}')
            )
        ORDER BY race_id, hri.horse_number;
        """

        # データを取得
        self.data = pd.read_sql_query(query, engine)
        logger.info(f"Data fetched successfully. Retrieved {len(self.data)} records.")

        logger.info("Post-processing data...")

        # 馬体重の数値変換（ベクトル化）
        self.data['horse_weight_int'] = pd.to_numeric(self.data['horse_weight'], errors='coerce')
        valid_weight_mask = (self.data['horse_weight_int'] >= 2) & (self.data['horse_weight_int'] <= 998)

        # レースごとの馬体重中央値を計算して無効値を補完
        valid_weights = self.data.loc[valid_weight_mask, ['race_id', 'horse_weight_int']]
        race_medians = valid_weights.groupby('race_id')['horse_weight_int'].median()
        
        # 無効な馬体重を各レースの中央値で補完
        invalid_weight_mask = ~valid_weight_mask
        self.data.loc[invalid_weight_mask, 'horse_weight_int'] = (
            self.data.loc[invalid_weight_mask, 'race_id'].map(race_medians)
        )

        # 増減差の処理（ベクトル化）
        weight_change_numeric = pd.to_numeric(self.data['weight_change'], errors='coerce')
        valid_change_mask = (weight_change_numeric >= 0) & (weight_change_numeric <= 998)

        # 増減差の符号を適用
        self.data['weight_change_value'] = np.where(
            valid_change_mask,
            np.where(self.data['weight_change_sign'] == '-', -weight_change_numeric, weight_change_numeric),
            0
        )

        logger.info("Post-processing completed.")
        return self

    def fit_preprocessors(self):
        """前処理器を学習データにフィットさせる"""
        logger.info("Fitting preprocessors...")

        # 数値特徴量の前処理器をフィット（特徴量ごとに個別のスケーラー）
        numerical_features = ['horse_weight_int', 'weight_change_value']
        for feature in numerical_features:
            scaler = StandardScaler()
            scaler.fit(self.data[[feature]])
            self.numerical_scalers[feature] = scaler

        # カテゴリ特徴量の前処理器をフィット
        categorical_features = ['weather_code']
        for feature in categorical_features:
            encoder = LabelEncoder()
            # 無効値を含む可能性があるので、fillnaで処理
            encoder.fit(self.data[feature].fillna('unknown').astype(str))
            self.categorical_encoders[feature] = encoder

        logger.info("Preprocessors fitted successfully.")
        return self

    def get_preprocessors(self):
        """前処理器を取得"""
        return {
            'numerical_scalers': self.numerical_scalers,
            'categorical_encoders': self.categorical_encoders
        }

    def set_preprocessors(self, preprocessors):
        """前処理器を設定"""
        self.numerical_scalers = preprocessors.get('numerical_scalers', {})
        self.categorical_encoders = preprocessors.get('categorical_encoders', {})
        return self

    def save_preprocessors(self, filepath):
        """前処理器を保存"""
        preprocessors = self.get_preprocessors()
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessors, f)
        logger.info(f"Preprocessors saved to {filepath}")
        return self

    def load_preprocessors(self, filepath):
        """前処理器を読み込み"""
        with open(filepath, 'rb') as f:
            preprocessors = pickle.load(f)
        self.set_preprocessors(preprocessors)
        logger.info(f"Preprocessors loaded from {filepath}")
        return self

    def preprocess_data(self):
        """データの前処理を実行"""
        logger.info("Preprocessing data...")
        
        # 数値特徴量の標準化（特徴量ごとに個別処理）
        for feature, scaler in self.numerical_scalers.items():
            normalized_values = scaler.transform(self.data[[feature]])
            self.data[f'{feature}_normalized'] = normalized_values.flatten()
        
        # カテゴリ特徴量のエンコード
        for feature, encoder in self.categorical_encoders.items():
            encoded_values = encoder.transform(self.data[feature].fillna('unknown').astype(str))
            self.data[f'{feature}_encoded'] = encoded_values
        
        logger.info("Data preprocessing completed.")
        return self
    
    def build_race_data(self):
        """レースデータを構築"""
        logger.info("Building race data...")
        
        self.races = []
        
        for race_id, race_group in self.data.groupby('race_id'):
            # 馬番を取得してインデックスに変換
            horse_numbers = race_group['horse_number'].astype(int).values
            horse_indices = horse_numbers - 1  # 馬番-1をインデックスに
            
            # 数値特徴量の初期化（NaNで埋める）
            x_num = np.full((self.max_horses, 2), np.nan, dtype=np.float32)
            
            # 数値特徴量をベクトル化で設定
            x_num[horse_indices, 0] = race_group['horse_weight_int_normalized'].values
            x_num[horse_indices, 1] = race_group['weight_change_value_normalized'].values
            
            # カテゴリ特徴量の初期化（0で埋める）
            x_cat = np.zeros((self.max_horses, 1), dtype=np.int64)
            
            # カテゴリ特徴量をベクトル化で設定
            x_cat[horse_indices, 0] = race_group['weather_code_encoded'].values
            
            # 着順の初期化（-1で埋める、無効な馬を示す）
            rankings = np.full(self.max_horses, -1, dtype=np.int32)
            
            # 着順をベクトル化で設定（1位=0, 2位=1, ...）
            final_orders = race_group['final_order'].astype(int).values
            rankings[horse_indices] = final_orders - 1
            
            # マスクの初期化（Falseで埋める）
            mask = np.zeros(self.max_horses, dtype=bool)
            
            # マスクをベクトル化で設定
            mask[horse_indices] = True
            
            race_data = {
                'race_id': race_id,
                'x_num': torch.tensor(x_num, dtype=torch.float32),
                'x_cat': torch.tensor(x_cat, dtype=torch.long),
                'rankings': torch.tensor(rankings, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.bool),
                'num_horses': len(horse_indices)
            }
            self.races.append(race_data)
        
        logger.info(f"Built {len(self.races)} races with valid data.")
        return self

    def get_feature_config(self):
        """モデル初期化用の特徴量設定を返す"""
        categorical_vocab_sizes = [
            len(encoder.classes_) 
            for encoder in self.categorical_encoders.values()
        ]
        
        return {
            'numerical_features': len(self.numerical_scalers),
            'categorical_features': categorical_vocab_sizes
        }

    def __len__(self):
        return len(self.races)

    def __getitem__(self, idx):
        race = self.races[idx]
        return race