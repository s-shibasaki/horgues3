import numpy as np
from torch.utils.data import Dataset
from horgues3.database import create_database_engine
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import hashlib
import pickle
from horgues3.odds import get_odds_dataframes


class HorguesDataset(Dataset):
    def __init__(self,
        start_date: str,
        end_date: str,
        num_horses: int = 18,
        horse_history_length: int = 10,
        history_days: int = 365,
        exclude_hours_before_race: int = 2,
        preprocessing_params: Optional[Dict[str, Dict]] = None,
        cache_dir: Optional[str] = 'cache',
        use_cache: bool = True,
    ):
        self._start_date = start_date
        self._end_date = end_date
        self._num_horses = num_horses
        self._horse_history_length = horse_history_length
        self._history_days = history_days
        self._exclude_hours_before_race = exclude_hours_before_race
        self._cache_dir = Path(cache_dir)
        self._use_cache = use_cache

        self._feature_aliases = {
            # Numerical features

            # Categorical features

        }

        self._race_data = self._get_race_data()
        self._target_race_ids = self._get_target_race_ids()
        self._numerical_data = self._process_numerical_data()
        self._categorical_data = self._process_categorical_data()
        self._meta_data = self._process_meta_data()

        if preprocessing_params is not None:
            self._scaler_params = preprocessing_params.get('scaler', {})
            self._encoder_params = preprocessing_params.get('encoder', {})
        else:
            self._scaler_params = self._fit_scaler()
            self._encoder_params = self._fit_encoder()

        self._scaled_numerical_data = self._scale_numerical_features()
        self._encoded_categorical_data = self._encode_categorical_features()

        self._precompute_history_groups()

        if self._use_cache:
            self._cache_dir.mkdir(exist_ok=True, parents=True)
            self._cache_key = hashlib.md5(f"{self._start_date}_{self._end_date}_{self._num_horses}_{self._horse_history_length}_{self._history_days}_{self._exclude_hours_before_race}".encode()).hexdigest()


    def get_preprocessing_params(self) -> Dict[str, Dict]:
        return {
            'scaler': self._scaler_params,
            'encoder': self._encoder_params
        }

    def get_model_config(self) -> Dict[str, Any]:
        """モデル作成に必要な設定情報を取得"""
        # 数値特徴量のリスト
        numerical_features = list(self._numerical_data.columns)
        
        # カテゴリ特徴量の辞書（特徴量名: 語彙サイズ）
        categorical_features = {}
        for column in self._categorical_data.columns:
            alias = self._feature_aliases.get(column, column)
            if alias in self._encoder_params:
                # エンコーダーパラメータから語彙サイズを計算（0はパディング用）
                vocab_size = len(self._encoder_params[alias])
                categorical_features[column] = vocab_size
        
        # シーケンスデータの名前リスト
        sequence_names = ['horse_history']  # 現在は horse_history のみ
        
        return {
            'numerical_features': numerical_features,
            'categorical_features': categorical_features,
            'sequence_names': sequence_names,
            'feature_aliases': self._feature_aliases.copy()
        }

    def _get_race_data(self) -> None:
        start_dt = datetime.strptime(self._start_date, '%Y%m%d')
        history_start_dt = start_dt - pd.Timedelta(days=self._history_days)
        history_start_date = history_start_dt.strftime('%Y%m%d')

        engine = create_database_engine()
        query = f"""
        SELECT
            ra.kaisai_year || ra.kaisai_monthday || ra.keibajo_code || ra.kaisai_kai || ra.kaisai_nichime || ra.race_number as race_id
            , ra.hasso_jikoku
            , ra.kaisai_year || ra.kaisai_monthday AS kaisai_date
            , ra.keibajo_code
            , ra.data_kubun

            , se.umaban
            , se.ketto_toroku_number
            , se.bataiju
            , se.kakutei_chakujun

        FROM race_shosai AS ra
        INNER JOIN uma_goto_race_joho AS se
            ON ra.kaisai_year = se.kaisai_year
            AND ra.kaisai_monthday = se.kaisai_monthday
            AND ra.keibajo_code = se.keibajo_code
            AND ra.kaisai_kai = se.kaisai_kai
            AND ra.kaisai_nichime = se.kaisai_nichime
            AND ra.race_number = se.race_number
        WHERE ra.kaisai_year || ra.kaisai_monthday BETWEEN '{history_start_date}' AND '{self._end_date}'

        -- raテーブルの条件
        AND ra.data_kubun != '0'

        -- seテーブルの条件
        AND se.data_kubun != '0'

        ORDER BY race_id, umaban
        """

        with engine.connect() as conn:
            result = pd.read_sql(query, conn)
        
        result = result.set_index(['race_id', 'umaban', 'ketto_toroku_number'], drop=False).sort_index()
        return result

    def _get_target_race_ids(self) -> pd.Index:
        target_data = self._race_data[
            (self._race_data['data_kubun'].isin(['2', '7'])) &
            (self._race_data.index.get_level_values('race_id').str[:8] >= self._start_date)
        ]
        target_race_ids = target_data.index.get_level_values('race_id').unique()
        return target_race_ids

    def _process_numerical_data(self) -> pd.DataFrame:
        numerical_data = pd.DataFrame(index=self._race_data.index)

        # bataiju
        numerical_data['bataiju'] = pd.to_numeric(self._race_data['bataiju'], errors='coerce')
        mask = (numerical_data['bataiju'] < 2) | (numerical_data['bataiju'] > 998)
        numerical_data.loc[mask, 'bataiju'] = np.nan

        # umaban
        numerical_data['umaban'] = pd.to_numeric(self._race_data['umaban'], errors='coerce')
        mask = (numerical_data['umaban'] < 1)
        numerical_data.loc[mask, 'umaban'] = np.nan

        # kakutei_chakujun
        numerical_data['kakutei_chakujun'] = pd.to_numeric(self._race_data['kakutei_chakujun'], errors='coerce')
        mask = (numerical_data['kakutei_chakujun'] < 1)
        numerical_data.loc[mask, 'kakutei_chakujun'] = np.nan

        # float32に変換
        numerical_data = numerical_data.astype(np.float32)

        return numerical_data

    def _process_categorical_data(self) -> pd.DataFrame:
        categorical_data = pd.DataFrame(index=self._race_data.index)

        # ketto_toroku_number
        categorical_data['ketto_toroku_number'] = self._race_data['ketto_toroku_number']
        mask = (categorical_data['ketto_toroku_number'] == '0000000000')
        categorical_data.loc[mask, 'ketto_toroku_number'] = None

        # keibajo_code
        categorical_data['keibajo_code'] = self._race_data['keibajo_code']
        mask = (categorical_data['keibajo_code'] == '00')
        categorical_data.loc[mask, 'keibajo_code'] = None

        return categorical_data
    
    def _process_meta_data(self) -> pd.DataFrame:
        meta_data = pd.DataFrame(index=self._race_data.index)

        # kakutei_chakujun
        meta_data['kakutei_chakujun'] = pd.to_numeric(self._race_data['kakutei_chakujun'], errors='coerce')
        mask = (meta_data['kakutei_chakujun'] < 1) | (meta_data['kakutei_chakujun'] > self._num_horses)
        meta_data.loc[mask, 'kakutei_chakujun'] = 0

        # ketto_toroku_number
        meta_data['ketto_toroku_number'] = self._race_data['ketto_toroku_number']
        mask = (meta_data['ketto_toroku_number'] == '0000000000')
        meta_data.loc[mask, 'ketto_toroku_number'] = None

        # hasso_datetime
        meta_data['hasso_datetime'] = pd.to_datetime(self._race_data['kaisai_date'] + self._race_data['hasso_jikoku'], format='%Y%m%d%H%M')

        return meta_data
    
    def _fit_scaler(self) -> Dict[str, Dict[str, float]]:
        scaler_params = {}
        
        # エイリアスごとに特徴量をグループ化
        alias_to_features = {}
        for column in self._numerical_data.columns:
            alias = self._feature_aliases.get(column, column)
            if alias not in alias_to_features:
                alias_to_features[alias] = []
            alias_to_features[alias].append(column)
        
        # エイリアスごとにスケーラーをフィット
        for alias, features in alias_to_features.items():
            # 同じエイリアスを持つ全ての特徴量の値を結合
            all_values = []
            for feature in features:
                values = self._numerical_data[feature].dropna().values
                all_values.extend(values)
            
            if len(all_values) > 0:
                all_values = np.array(all_values)
                mean_val = float(all_values.mean())
                std_val = float(all_values.std())
                if std_val == 0:
                    std_val = 1.0
                scaler_params[alias] = {'mean': mean_val, 'std': std_val}
            else:
                scaler_params[alias] = {'mean': 0.0, 'std': 1.0}

        return scaler_params

    def _fit_encoder(self) -> Dict[str, Dict[Any, int]]:
        encoder_params = {}
        
        # エイリアスごとに特徴量をグループ化
        alias_to_features = {}
        for column in self._categorical_data.columns:
            alias = self._feature_aliases.get(column, column)
            if alias not in alias_to_features:
                alias_to_features[alias] = []
            alias_to_features[alias].append(column)
        
        # エイリアスごとにエンコーダーをフィット
        for alias, features in alias_to_features.items():
            # 同じエイリアスを持つ全ての特徴量の一意な値を結合
            all_values = set()
            for feature in features:
                values = self._categorical_data[feature].dropna().unique()
                all_values.update(values)
            
            if len(all_values) > 0:
                # ソートして一貫性を保つ
                sorted_values = sorted(all_values)
                encoder_params[alias] = {value: idx + 1 for idx, value in enumerate(sorted_values)}
            else:
                encoder_params[alias] = {}

        return encoder_params
    
    def _scale_numerical_features(self):
        scaled_data = pd.DataFrame(index=self._numerical_data.index)
        for column in self._numerical_data.columns:
            alias = self._feature_aliases.get(column, column)
            mean = self._scaler_params[alias]['mean']
            std = self._scaler_params[alias]['std']
            scaled_data[column] = (self._numerical_data[column] - mean) / std
        return scaled_data

    def _encode_categorical_features(self):
        encoded_data = pd.DataFrame(index=self._categorical_data.index)
        for column in self._categorical_data.columns:
            alias = self._feature_aliases.get(column, column)
            mapping = self._encoder_params[alias]
            encoded_data[column] = self._categorical_data[column].map(mapping).fillna(0).astype(np.int64)
        return encoded_data
    
    def _precompute_history_groups(self):
        history_groups = {}

        for key in ['ketto_toroku_number']:
            valid_meta_data = self._meta_data[self._meta_data[key].notna()].sort_values('hasso_datetime')
            history_groups[key] = {value: data for value, data in valid_meta_data.groupby(valid_meta_data['ketto_toroku_number'])}

        self._history_groups = history_groups

    def _compute_item(self, idx: int) -> Dict[str, Any]:
        race_id = self._target_race_ids[idx]

        data = {
            'race_id': race_id,
            'x_num': {key: np.full((self._num_horses,), np.nan, dtype=np.float32) for key in [
                'umaban',
                'bataiju',
            ]},
            'x_cat': {key: np.zeros((self._num_horses,), dtype=np.int64) for key in [
                'ketto_toroku_number',
                'keibajo_code',
            ]},
            'mask': np.zeros((self._num_horses,), dtype=np.bool_),
            'sequence_data': {
                'horse_history': {
                    'x_num': {key: np.full((self._num_horses, self._horse_history_length), np.nan, dtype=np.float32) for key in [
                        'umaban',
                        'bataiju',
                        'kakutei_chakujun',
                    ]},
                    'x_cat': {key: np.zeros((self._num_horses, self._horse_history_length), dtype=np.int64) for key in [
                        'keibajo_code',
                    ]},
                    'mask': np.zeros((self._num_horses, self._horse_history_length), dtype=np.bool_),
                }
            },
            'rankings': np.zeros((self._num_horses,), dtype=np.int64),
        }

        # 指定されたレースのデータを取得
        race_meta_data = self._meta_data.loc[race_id]
        race_scaled_data = self._scaled_numerical_data.loc[race_id]
        race_encoded_data = self._encoded_categorical_data.loc[race_id]

        # 現在のレースの馬データを設定
        for (umaban, _), horse_meta_data in race_meta_data.iterrows():
            horse_idx = int(umaban) - 1
            
            # 数値データの設定
            scaled_horse_data = race_scaled_data.loc[umaban].iloc[0]
            for key in data['x_num'].keys():
                data['x_num'][key][horse_idx] = scaled_horse_data[key]

            # カテゴリデータの設定
            encoded_horse_data = race_encoded_data.loc[umaban].iloc[0]
            for key in data['x_cat'].keys():
                data['x_cat'][key][horse_idx] = encoded_horse_data[key]

            # マスクの設定
            data['mask'][horse_idx] = True

            # 履歴データの設定
            current_race_datetime = horse_meta_data['hasso_datetime']

            ketto_toroku_number = horse_meta_data['ketto_toroku_number']
            history_meta_data = self._history_groups['ketto_toroku_number'].get(ketto_toroku_number)
            if history_meta_data is not None:
                history_meta_data = history_meta_data[(
                    (history_meta_data['hasso_datetime'] < current_race_datetime - pd.Timedelta(hours=self._exclude_hours_before_race)) &
                    (history_meta_data['hasso_datetime'] >= current_race_datetime - pd.Timedelta(days=self._history_days))
                )].sort_values('hasso_datetime', ascending=False).tail(self._horse_history_length)
            
                scaled_history_data = self._scaled_numerical_data.loc[history_meta_data.index]
                for key in data['sequence_data']['horse_history']['x_num'].keys():
                    data['sequence_data']['horse_history']['x_num'][key][horse_idx, :len(history_meta_data)] = scaled_history_data[key].values

                encoded_history_data = self._encoded_categorical_data.loc[history_meta_data.index]
                for key in data['sequence_data']['horse_history']['x_cat'].keys():
                    data['sequence_data']['horse_history']['x_cat'][key][horse_idx, :len(history_meta_data)] = encoded_history_data[key].values

                data['sequence_data']['horse_history']['mask'][horse_idx, :len(history_meta_data)] = True

            # ランキングの設定
            data['rankings'][horse_idx] = horse_meta_data['kakutei_chakujun']

        return data

    def __len__(self) -> int:
        return len(self._target_race_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._use_cache:
            cache_path = self._cache_dir / f"{self._cache_key}_{idx}.pkl"
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
                
        data = self._compute_item(idx)

        if self._use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
            except:
                pass
        
        return data
