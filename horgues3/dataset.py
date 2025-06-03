import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import logging
from sqlalchemy import create_engine
import pandas as pd

logger = logging.getLogger(__name__)

class Horgues3Dataset(Dataset):
    def __init__(self, max_horses=18, start_ymd='20200101', end_ymd='20231231'):
        super().__init__()
        self.max_horses = max_horses
        self.start_ymd = start_ymd
        self.end_ymd = end_ymd

        # ymd形式をyyyy, mmddに分割
        self.start_year = int(start_ymd[:4])
        self.start_month_day = start_ymd[4:]
        self.end_year = int(end_ymd[:4])
        self.end_month_day = end_ymd[4:]

    def fetch_data(self):
        engine = create_engine(f"postgresql://postgres:postgres@localhost/horgues3")

        # レースID、馬体重、確定着順を取得するSQL
        query = f"""
        SELECT 
            kaisai_year || kaisai_month_day || track_code || kaisai_kai || kaisai_day || race_number as race_id,
            horse_number,
            horse_weight,
            final_order
        FROM public.horse_race_info
        WHERE 1=1 
            AND horse_number BETWEEN '01' AND '{self.max_horses:02}'
            AND horse_weight BETWEEN '002' AND '998'
            AND final_order BETWEEN '01' AND '{self.max_horses:02}'
            AND (
                kaisai_year > '{self.start_year}'
                OR (kaisai_year = '{self.start_year}' AND kaisai_month_day >= '{self.start_month_day}')
            )
            AND (
                kaisai_year < '{self.end_year}'
                OR (kaisai_year = '{self.end_year}' AND kaisai_month_day <= '{self.end_month_day}')
            )
        ORDER BY race_id, horse_number;
        """

        # データを取得
        self.data = pd.read_sql_query(query, engine)
        logger.info(f"Fetched {len(self.data)} records from the database.")

        return self

    def prepare_races(self):
        """レースごとにデータを整理"""
        self.races = []
        grouped = self.data.groupby('race_id')

        for race_id, race_data in grouped:
            # 各レースのデータを馬番でインデックスした配列として整理
            horse_weights = np.full(self.max_horses, np.nan)
            rankings = np.full(self.max_horses, -1, dtype=int)
            mask = np.zeros(self.max_horses, dtype=bool)

            for _, row in race_data.iterrows():
                horse_num = int(row['horse_number']) - 1  # 馬番を0ベースのインデックスに変換

                # 馬体重を正規化 (400-600kgが標準的)
                normalized_weight = (int(row['horse_weight']) - 450) / 100
                horse_weights[horse_num] = normalized_weight

                # 着順を0ベースに変換 (1位→0, 2位→1, ...)
                rankings[horse_num] = int(row['final_order']) - 1

                # この馬は有効
                mask[horse_num] = True
            
            # 有効な馬が2頭以上いるレースのみを保存
            if mask.sum() >= 2:
                race_info = {
                    'race_id': race_id,
                    'horse_weights': horse_weights,
                    'rankings': rankings,
                    'mask': mask,
                    'num_horses': np.sum(mask)
                }
                self.races.append(race_info)

        logger.info(f"Prepared {len(self.races)} races with 2+ horses each.")

        return self
    
    def __len__(self):
        return len(self.races)

    def __getitem__(self, idx):
        race = self.races[idx]

        # 数値特徴量
        horse_weights_tensor = torch.tensor(race['horse_weights'], dtype=torch.float32)
        x_num = torch.stack([horse_weights_tensor], dim=1)  # (max_horses, 1)

        # カテゴリ特徴量
        x_cat = torch.zeros(self.max_horses, 0, dtype=torch.long)  # (max_horses, 0)

        # 着順
        rankings = torch.tensor(race['rankings'], dtype=torch.long)  # (max_horses,)

        # マスク
        mask = torch.tensor(race['mask'], dtype=torch.bool)  # (max_horses,)
        
        return {
            'x_num': x_num,
            'x_cat': x_cat,
            'rankings': rankings,
            'mask': mask,
            'race_id': race['race_id']
        }