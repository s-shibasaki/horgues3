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

        # 数値に変換
        self.data['horse_number_numeric'] = pd.to_numeric(self.data['horse_number'], errors='coerce')
        self.data['final_order_numeric'] = pd.to_numeric(self.data['final_order'], errors='coerce')

        # 馬体重の数値変換（2～998の範囲外の値をNaNにする）
        self.data['horse_weight_numeric'] = pd.to_numeric(self.data['horse_weight'], errors='coerce')
        valid_weight_mask = (self.data['horse_weight_numeric'] >= 2) & (self.data['horse_weight_numeric'] <= 998)
        self.data.loc[~valid_weight_mask, 'horse_weight_numeric'] = np.nan

        # 増減差の処理（-998～998の範囲外の値をNaNにする）
        weight_change_str = self.data['weight_change_sign'].fillna('') + self.data['weight_change'].fillna('')
        self.data['weight_change_numeric'] = pd.to_numeric(weight_change_str, errors='coerce')
        valid_change_mask = (self.data['weight_change_numeric'] >= -998) & (self.data['weight_change_numeric'] <= 998)
        self.data.loc[~valid_change_mask, 'weight_change_numeric'] = np.nan

        logger.info("Post-processing completed.")
        return self

    def fit_preprocessors(self):
        """前処理器を学習データにフィットさせる"""
        logger.info("Fitting preprocessors...")

        # 数値特徴量の前処理器をフィット（特徴量ごとに個別のスケーラー）
        numerical_features = ['horse_weight_numeric', 'weight_change_numeric']
        for feature in numerical_features:
            scaler = StandardScaler()
            scaler.fit(self.data[[feature]])
            self.numerical_scalers[feature] = scaler

        # カテゴリ特徴量の前処理器をフィット
        categorical_features = ['weather_code']
        for feature in categorical_features:
            encoder = CustomLabelEncoder()
            encoder.fit(self.data[feature])
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
            encoded_values = encoder.transform(self.data[feature])
            self.data[f'{feature}_encoded'] = encoded_values
        
        logger.info("Data preprocessing completed.")
        return self
    
    def build_race_data(self):
        """レースデータを構築"""
        logger.info("Building race data...")
        
        self.races = []
        
        for race_id, race_group in self.data.groupby('race_id'):
            # 馬番を取得してインデックスに変換
            horse_numbers = race_group['horse_number_numeric'].values
            horse_indices = horse_numbers - 1  # 馬番-1をインデックスに
            
            # 数値特徴量を辞書形式で初期化（NaNで埋める）
            x_num = {
                'horse_weight_numeric_normalized': np.full(self.max_horses, np.nan, dtype=np.float32),
                'weight_change_numeric_normalized': np.full(self.max_horses, np.nan, dtype=np.float32)
            }
            
            # 数値特徴量をベクトル化で設定
            x_num['horse_weight_numeric_normalized'][horse_indices] = race_group['horse_weight_numeric_normalized'].values
            x_num['weight_change_numeric_normalized'][horse_indices] = race_group['weight_change_numeric_normalized'].values
            
            # カテゴリ特徴量を辞書形式で初期化（0で埋める）
            x_cat = {
                'weather_code_encoded': np.zeros(self.max_horses, dtype=np.int64)
            }
            
            # カテゴリ特徴量をベクトル化で設定
            x_cat['weather_code_encoded'][horse_indices] = race_group['weather_code_encoded'].values
            
            # 着順の初期化（-1で埋める、無効な馬を示す）
            rankings = np.full(self.max_horses, -1, dtype=np.int32)
            
            # 着順をベクトル化で設定（1位=0, 2位=1, ...）
            final_orders = race_group['final_order_numeric'].values
            rankings[horse_indices] = final_orders - 1
            
            # マスクの初期化（Falseで埋める）
            mask = np.zeros(self.max_horses, dtype=bool)
            
            # マスクをベクトル化で設定
            mask[horse_indices] = True
            
            # テンソルに変換
            x_num_tensors = {key: torch.tensor(value, dtype=torch.float32) for key, value in x_num.items()}
            x_cat_tensors = {key: torch.tensor(value, dtype=torch.long) for key, value in x_cat.items()}
            
            race_data = {
                'race_id': race_id,
                'x_num': x_num_tensors,
                'x_cat': x_cat_tensors,
                'rankings': torch.tensor(rankings, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.bool),
                'num_horses': len(horse_indices)
            }
            self.races.append(race_data)
        
        logger.info(f"Built {len(self.races)} races with valid data.")
        return self

    def get_feature_config(self):
        """モデル初期化用の特徴量設定を返す"""
        numerical_features = list(self.numerical_scalers.keys())
        numerical_features = [name for name in numerical_features]
        
        categorical_features = {}
        for feature, encoder in self.categorical_encoders.items():
            categorical_features[feature] = encoder.get_vocab_size()
        
        return {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features
        }

    def __len__(self):
        return len(self.races)

    def __getitem__(self, idx):
        race = self.races[idx]
        return race