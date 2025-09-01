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
        horse_history_length: int = 50,
        jockey_history_length: int = 50,
        trainer_history_length: int = 50,
        chokyo_history_length: int = 50,
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
        self._jockey_history_length = jockey_history_length
        self._trainer_history_length = trainer_history_length
        self._chokyo_history_length = chokyo_history_length
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

        self._chokyo_data = self._get_chokyo_data()
        self._chokyo_numerical_data = self._process_chokyo_numerical_data()
        self._chokyo_categorical_data = self._process_chokyo_categorical_data()
        self._chokyo_meta_data = self._process_chokyo_meta_data()

        if preprocessing_params is not None:
            self._scaler_params = preprocessing_params.get('scaler', {})
            self._encoder_params = preprocessing_params.get('encoder', {})
            self._chokyo_scaler_params = preprocessing_params.get('chokyo_scaler', {})
            self._chokyo_encoder_params = preprocessing_params.get('chokyo_encoder', {})
        else:
            self._scaler_params = self._fit_scaler()
            self._encoder_params = self._fit_encoder()
            self._chokyo_scaler_params = self._fit_chokyo_scaler()
            self._chokyo_encoder_params = self._fit_chokyo_encoder()

        self._scaled_numerical_data = self._scale_numerical_features()
        self._encoded_categorical_data = self._encode_categorical_features()
        self._scaled_chokyo_numerical_data = self._scale_chokyo_numerical_features()
        self._encoded_chokyo_categorical_data = self._encode_chokyo_categorical_features()

        self._precompute_history_groups()

        if self._use_cache:
            self._cache_dir.mkdir(exist_ok=True, parents=True)
            self._cache_key = hashlib.md5(f"{self._start_date}_{self._end_date}_{self._num_horses}_{self._horse_history_length}_{self._jockey_history_length}_{self._trainer_history_length}_{self._history_days}_{self._exclude_hours_before_race}".encode()).hexdigest()


    def get_preprocessing_params(self) -> Dict[str, Dict]:
        return {
            'scaler': self._scaler_params,
            'encoder': self._encoder_params,
            'chokyo_scaler': self._chokyo_scaler_params,
            'chokyo_encoder': self._chokyo_encoder_params
        }

    def get_model_config(self) -> Dict[str, Any]:
        """モデル作成に必要な設定情報を取得"""
        # 数値特徴量のリスト
        numerical_features = list(self._numerical_data.columns)
        numerical_features.extend(list(self._chokyo_numerical_data.columns))

        # カテゴリ特徴量の辞書（特徴量名: 語彙サイズ）
        categorical_features = {}
        for column in self._categorical_data.columns:
            alias = self._feature_aliases.get(column, column)
            if alias in self._encoder_params:
                # エンコーダーパラメータから語彙サイズを計算（0はパディング用）
                vocab_size = len(self._encoder_params[alias])
                categorical_features[column] = vocab_size

        # 調教特徴量のカテゴリ特徴量を処理
        for column in self._chokyo_categorical_data.columns:
            alias = self._feature_aliases.get(column, column)
            if alias in self._chokyo_encoder_params:
                # エンコーダーパラメータから語彙サイズを計算（0はパディング用）
                vocab_size = len(self._chokyo_encoder_params[alias])
                categorical_features[column] = vocab_size
        

        # シーケンスデータの名前リスト
        sequence_names = ['horse_history', 'jockey_history', 'trainer_history', 'chokyo_history']
        
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
            , ra.data_kubun
            , ra.kaisai_year || ra.kaisai_monthday AS kaisai_date
            , ra.keibajo_code
            , ra.kyori
            , ra.track_code
            , ra.course_kubun
            , CASE WHEN ra.prev_hon_shokin_1 = '00000000' THEN ra.hon_shokin_1 ELSE ra.prev_hon_shokin_1 END AS hon_shokin_1
            , CASE WHEN ra.prev_hon_shokin_2 = '00000000' THEN ra.hon_shokin_2 ELSE ra.prev_hon_shokin_2 END AS hon_shokin_2
            , CASE WHEN ra.prev_hon_shokin_3 = '00000000' THEN ra.hon_shokin_3 ELSE ra.prev_hon_shokin_3 END AS hon_shokin_3
            , CASE WHEN ra.prev_hon_shokin_4 = '00000000' THEN ra.hon_shokin_4 ELSE ra.prev_hon_shokin_4 END AS hon_shokin_4
            , CASE WHEN ra.prev_hon_shokin_5 = '00000000' THEN ra.hon_shokin_5 ELSE ra.prev_hon_shokin_5 END AS hon_shokin_5
            , ra.hasso_jikoku
            , ra.toroku_tosu
            , ra.tenko_code
            , CASE 
                WHEN ra.track_code IN ('10','11','12','13','14','15','16','17','18','19','20','21','22','51','53','54','55','56','57','58','59') THEN ra.shiba_baba_jotai_code
                WHEN ra.track_code IN ('23','24','25','26','27','28','29','52') THEN ra.dirt_baba_jotai_code
                ELSE '0'
              END AS baba_jotai_code

            , se.umaban
            , se.wakuban
            , se.ketto_toroku_number
            , se.uma_kigo_code
            , se.seibetsu_code
            , se.hinshu_code
            , se.keiro_code
            , se.barei
            , se.tozai_shozoku_code
            , se.chokyoshi_code
            , se.banushi_code
            , se.futan_juryo
            , se.blinker_shiyo_kubun
            , TRIM(se.kishu_name_short) AS kishu_name_short
            , se.kishu_minarai_code
            , se.bataiju
            , se.zogen_fugo
            , se.zogen_sa
            , se.ijo_kubun_code
            , se.kakutei_chakujun
            , se.soha_time
            , se.corner_juni_1
            , se.corner_juni_2
            , se.corner_juni_3
            , se.corner_juni_4
            , se.kakutoku_hon_shokin
            , se.ushiro_3_furlongs_time
            , se.konkai_race_kyakushitsu_hantei

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
    
    def _get_chokyo_data(self) -> pd.DataFrame:
        # hanro_chokyo テーブルから training_center_kubun, chokyo_date, chokyo_jikoku, ketto_toroku_number, time_gokei_4_furlongs ~ time_gokei_2_furlongs, lap_time_800_600 ~ lap_time_200_0 を取得。
        # wood_chip_chokyoからも取得し、ユニオンする。wood_chip_chokyoは10ハロンまで（2000メートルまで）あるほか、course, baba_mawari, カラムもある。
        # data_kubun!='0'、調教年月日は _get_race_dataと同様に指定する。
        start_dt = datetime.strptime(self._start_date, '%Y%m%d')
        history_start_dt = start_dt - pd.Timedelta(days=self._history_days)
        history_start_date = history_start_dt.strftime('%Y%m%d')

        engine = create_database_engine()
        query = f"""
        SELECT
            hc.training_center_kubun
            , hc.chokyo_date
            , hc.chokyo_jikoku
            , hc.ketto_toroku_number
            , hc.data_kubun
            , NULL as time_gokei_10_furlongs
            , NULL as time_gokei_9_furlongs
            , NULL as time_gokei_8_furlongs
            , NULL as time_gokei_7_furlongs
            , NULL as time_gokei_6_furlongs
            , NULL as time_gokei_5_furlongs
            , hc.time_gokei_4_furlongs
            , hc.time_gokei_3_furlongs
            , hc.time_gokei_2_furlongs
            , NULL as lap_time_2000_1800
            , NULL as lap_time_1800_1600
            , NULL as lap_time_1600_1400
            , NULL as lap_time_1400_1200
            , NULL as lap_time_1200_1000
            , NULL as lap_time_1000_800
            , hc.lap_time_800_600
            , hc.lap_time_600_400
            , hc.lap_time_400_200
            , hc.lap_time_200_0
            , NULL as course
            , NULL as baba_mawari
        FROM hanro_chokyo AS hc
        WHERE hc.chokyo_date BETWEEN '{history_start_date}' AND '{self._end_date}'
        AND hc.data_kubun != '0'
        
        UNION ALL
        
        SELECT
            wc.training_center_kubun
            , wc.chokyo_date
            , wc.chokyo_jikoku
            , wc.ketto_toroku_number
            , wc.data_kubun
            , wc.time_gokei_10_furlongs
            , wc.time_gokei_9_furlongs
            , wc.time_gokei_8_furlongs
            , wc.time_gokei_7_furlongs
            , wc.time_gokei_6_furlongs
            , wc.time_gokei_5_furlongs
            , wc.time_gokei_4_furlongs
            , wc.time_gokei_3_furlongs
            , wc.time_gokei_2_furlongs
            , wc.lap_time_2000_1800
            , wc.lap_time_1800_1600
            , wc.lap_time_1600_1400
            , wc.lap_time_1400_1200
            , wc.lap_time_1200_1000
            , wc.lap_time_1000_800
            , wc.lap_time_800_600
            , wc.lap_time_600_400
            , wc.lap_time_400_200
            , wc.lap_time_200_0
            , wc.course
            , wc.baba_mawari
        FROM wood_chip_chokyo AS wc
        WHERE wc.chokyo_date BETWEEN '{history_start_date}' AND '{self._end_date}'
        AND wc.data_kubun != '0'
        
        ORDER BY ketto_toroku_number, chokyo_date, chokyo_jikoku
        """

        with engine.connect() as conn:
            result = pd.read_sql(query, conn)
        
        # インデックスを設定
        result = result.set_index(['ketto_toroku_number', 'chokyo_date', 'chokyo_jikoku'], drop=False).sort_index()
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

        # umaban
        numerical_data['umaban'] = pd.to_numeric(self._race_data['umaban'], errors='coerce')
        mask = (numerical_data['umaban'] < 1)
        numerical_data.loc[mask, 'umaban'] = np.nan

        # wakuban
        numerical_data['wakuban'] = pd.to_numeric(self._race_data['wakuban'], errors='coerce')
        mask = (numerical_data['wakuban'] < 1)
        numerical_data.loc[mask, 'wakuban'] = np.nan

        # barei
        numerical_data['barei'] = pd.to_numeric(self._race_data['barei'], errors='coerce')
        mask = (numerical_data['barei'] < 1)
        numerical_data.loc[mask, 'barei'] = np.nan

        # futan_juryo
        numerical_data['futan_juryo'] = pd.to_numeric(self._race_data['futan_juryo'], errors='coerce')
        mask = (numerical_data['futan_juryo'] < 1)
        numerical_data.loc[mask, 'futan_juryo'] = np.nan

        # bataiju
        numerical_data['bataiju'] = pd.to_numeric(self._race_data['bataiju'], errors='coerce')
        mask = (numerical_data['bataiju'] < 2) | (numerical_data['bataiju'] > 998)
        numerical_data.loc[mask, 'bataiju'] = np.nan

        # zogen_sa
        numerical_data['zogen_sa'] = pd.to_numeric(self._race_data['zogen_fugo'] + self._race_data['zogen_sa'], errors='coerce')
        mask = (numerical_data['zogen_sa'] < -998) | (numerical_data['zogen_sa'] > 998)
        numerical_data.loc[mask, 'zogen_sa'] = np.nan

        # soha_sokudo (計算: kyori(m) / soha_time(秒) * 1000 -> m/s)
        soha_time_numeric = pd.to_numeric(self._race_data['soha_time'].str.slice(0, 1), errors='coerce') * 60 + pd.to_numeric(self._race_data['soha_time'].str.slice(1, 3), errors='coerce') * 0.1
        kyori_numeric = pd.to_numeric(self._race_data['kyori'], errors='coerce')
        # soha_timeが0より大きく、kyoriが有効な場合のみ計算
        mask = (soha_time_numeric > 0) & (kyori_numeric > 0)
        numerical_data['soha_sokudo'] = np.nan
        numerical_data.loc[mask, 'soha_sokudo'] = kyori_numeric[mask] / (soha_time_numeric[mask] / 1000)

        # kakutei_chakujun
        numerical_data['kakutei_chakujun'] = pd.to_numeric(self._race_data['kakutei_chakujun'], errors='coerce')
        mask = (numerical_data['kakutei_chakujun'] < 1)
        numerical_data.loc[mask, 'kakutei_chakujun'] = np.nan

        # kyori
        numerical_data['kyori'] = pd.to_numeric(self._race_data['kyori'], errors='coerce')
        mask = (numerical_data['kyori'] < 1)
        numerical_data.loc[mask, 'kyori'] = np.nan

        # toroku_tosu
        numerical_data['toroku_tosu'] = pd.to_numeric(self._race_data['toroku_tosu'], errors='coerce')
        mask = (numerical_data['toroku_tosu'] < 1)
        numerical_data.loc[mask, 'toroku_tosu'] = np.nan

        # hon_shokin_1~5
        for i in range(1, 6):
            col_name = f'hon_shokin_{i}'
            numerical_data[col_name] = pd.to_numeric(self._race_data[col_name], errors='coerce')
            # 賞金が0万円は無効とする
            mask = (numerical_data[col_name] < 1)
            numerical_data.loc[mask, col_name] = np.nan

        # corner_juni_1~4
        for i in range(1, 5):
            col_name = f'corner_juni_{i}'
            numerical_data[col_name] = pd.to_numeric(self._race_data[col_name], errors='coerce')
            mask = (numerical_data[col_name] < 1)
            numerical_data.loc[mask, col_name] = np.nan

        # kakutoku_hon_shokin
        numerical_data['kakutoku_hon_shokin'] = pd.to_numeric(self._race_data['kakutoku_hon_shokin'], errors='coerce')
        mask = (numerical_data['kakutoku_hon_shokin'] < 1)
        numerical_data.loc[mask, 'kakutoku_hon_shokin'] = np.nan

        # ushiro_3_furlongs_time
        numerical_data['ushiro_3_furlongs_time'] = pd.to_numeric(self._race_data['ushiro_3_furlongs_time'], errors='coerce') * 0.1
        mask = (numerical_data['ushiro_3_furlongs_time'] <= 0)
        numerical_data.loc[mask, 'ushiro_3_furlongs_time'] = np.nan

        # float32に変換
        numerical_data = numerical_data.astype(np.float32)

        return numerical_data

    def _process_categorical_data(self) -> pd.DataFrame:
        categorical_data = pd.DataFrame(index=self._race_data.index)

        # keibajo_code
        categorical_data['keibajo_code'] = self._race_data['keibajo_code']
        mask = (categorical_data['keibajo_code'] == '00')
        categorical_data.loc[mask, 'keibajo_code'] = None

        # track_code
        categorical_data['track_code'] = self._race_data['track_code']
        mask = (categorical_data['track_code'] == '00')
        categorical_data.loc[mask, 'track_code'] = None

        # course_kubun
        categorical_data['course_kubun'] = self._race_data['course_kubun']
        mask = (categorical_data['course_kubun'] == '0')
        categorical_data.loc[mask, 'course_kubun'] = None

        # tenko_code
        categorical_data['tenko_code'] = self._race_data['tenko_code']
        mask = (categorical_data['tenko_code'] == '0')
        categorical_data.loc[mask, 'tenko_code'] = None

        # baba_jotai_code
        categorical_data['baba_jotai_code'] = self._race_data['baba_jotai_code']
        mask = (categorical_data['baba_jotai_code'] == '0')
        categorical_data.loc[mask, 'baba_jotai_code'] = None

        # ketto_toroku_number
        categorical_data['ketto_toroku_number'] = self._race_data['ketto_toroku_number']
        mask = (categorical_data['ketto_toroku_number'] == '0000000000')
        categorical_data.loc[mask, 'ketto_toroku_number'] = None

        # uma_kigo_code
        categorical_data['uma_kigo_code'] = self._race_data['uma_kigo_code']
        mask = (categorical_data['uma_kigo_code'] == '00')
        categorical_data.loc[mask, 'uma_kigo_code'] = None

        # seibetsu_code
        categorical_data['seibetsu_code'] = self._race_data['seibetsu_code']
        mask = (categorical_data['seibetsu_code'] == '0')
        categorical_data.loc[mask, 'seibetsu_code'] = None

        # hinshu_code
        categorical_data['hinshu_code'] = self._race_data['hinshu_code']
        mask = (categorical_data['hinshu_code'] == '0')
        categorical_data.loc[mask, 'hinshu_code'] = None

        # keiro_code
        categorical_data['keiro_code'] = self._race_data['keiro_code']
        mask = (categorical_data['keiro_code'] == '00')
        categorical_data.loc[mask, 'keiro_code'] = None

        # tozai_shozoku_code
        categorical_data['tozai_shozoku_code'] = self._race_data['tozai_shozoku_code']
        mask = (categorical_data['tozai_shozoku_code'] == '0')
        categorical_data.loc[mask, 'tozai_shozoku_code'] = None

        # chokyoshi_code
        categorical_data['chokyoshi_code'] = self._race_data['chokyoshi_code']
        mask = (categorical_data['chokyoshi_code'] == '00000')
        categorical_data.loc[mask, 'chokyoshi_code'] = None

        # banushi_code
        categorical_data['banushi_code'] = self._race_data['banushi_code']
        mask = (categorical_data['banushi_code'] == '000000')
        categorical_data.loc[mask, 'banushi_code'] = None

        # blinker_shiyo_kubun
        categorical_data['blinker_shiyo_kubun'] = self._race_data['blinker_shiyo_kubun']
        mask = (categorical_data['blinker_shiyo_kubun'] == '0')
        categorical_data.loc[mask, 'blinker_shiyo_kubun'] = None

        # kishu_name_short
        categorical_data['kishu_name_short'] = self._race_data['kishu_name_short']
        mask = (categorical_data['kishu_name_short'] == '') | (categorical_data['kishu_name_short'].str.strip() == '')
        categorical_data.loc[mask, 'kishu_name_short'] = None

        # kishu_minarai_code
        categorical_data['kishu_minarai_code'] = self._race_data['kishu_minarai_code']
        mask = (categorical_data['kishu_minarai_code'] == '0')
        categorical_data.loc[mask, 'kishu_minarai_code'] = None

        # konkai_race_kyakushitsu_hantei
        categorical_data['konkai_race_kyakushitsu_hantei'] = self._race_data['konkai_race_kyakushitsu_hantei']
        mask = (categorical_data['konkai_race_kyakushitsu_hantei'] == '0')
        categorical_data.loc[mask, 'konkai_race_kyakushitsu_hantei'] = None

        return categorical_data
    
    def _process_chokyo_numerical_data(self):
        numerical_data = pd.DataFrame(index=self._chokyo_data.index)

        # time_gokei_* fields (convert to seconds)
        time_fields = ['time_gokei_10_furlongs', 'time_gokei_9_furlongs', 'time_gokei_8_furlongs', 
                      'time_gokei_7_furlongs', 'time_gokei_6_furlongs', 'time_gokei_5_furlongs',
                      'time_gokei_4_furlongs', 'time_gokei_3_furlongs', 'time_gokei_2_furlongs']
        
        for field in time_fields:
            if field in self._chokyo_data.columns:
                # Convert time format (e.g., "1234" -> 12.34 seconds)
                numerical_data[field] = pd.to_numeric(self._chokyo_data[field], errors='coerce') * 0.01
                mask = (numerical_data[field] <= 0)
                numerical_data.loc[mask, field] = np.nan

        # lap_time_* fields (convert to seconds)
        lap_fields = ['lap_time_2000_1800', 'lap_time_1800_1600', 'lap_time_1600_1400',
                     'lap_time_1400_1200', 'lap_time_1200_1000', 'lap_time_1000_800',
                     'lap_time_800_600', 'lap_time_600_400', 'lap_time_400_200', 'lap_time_200_0']
        
        for field in lap_fields:
            if field in self._chokyo_data.columns:
                # Convert time format (e.g., "123" -> 12.3 seconds)
                numerical_data[field] = pd.to_numeric(self._chokyo_data[field], errors='coerce') * 0.1
                mask = (numerical_data[field] <= 0)
                numerical_data.loc[mask, field] = np.nan

        # Convert to float32
        numerical_data = numerical_data.astype(np.float32)

        return numerical_data

    def _process_chokyo_categorical_data(self):
        categorical_data = pd.DataFrame(index=self._chokyo_data.index)

        # training_center_kubun
        categorical_data['training_center_kubun'] = self._chokyo_data['training_center_kubun']
        mask = (categorical_data['training_center_kubun'] == '0')
        categorical_data.loc[mask, 'training_center_kubun'] = None

        # course (only available in wood_chip_chokyo)
        if 'course' in self._chokyo_data.columns:
            categorical_data['course'] = self._chokyo_data['course']
            mask = (categorical_data['course'].isna()) | (categorical_data['course'] == '')
            categorical_data.loc[mask, 'course'] = None

        # baba_mawari (only available in wood_chip_chokyo)
        if 'baba_mawari' in self._chokyo_data.columns:
            categorical_data['baba_mawari'] = self._chokyo_data['baba_mawari']
            mask = (categorical_data['baba_mawari'] == '0')
            categorical_data.loc[mask, 'baba_mawari'] = None

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

        # kishu_name_short (騎手名)
        meta_data['kishu_name_short'] = self._race_data['kishu_name_short']
        mask = (meta_data['kishu_name_short'] == '') | (meta_data['kishu_name_short'].str.strip() == '')
        meta_data.loc[mask, 'kishu_name_short'] = None

        # chokyoshi_code (調教師コード)
        meta_data['chokyoshi_code'] = self._race_data['chokyoshi_code']
        mask = (meta_data['chokyoshi_code'] == '00000')
        meta_data.loc[mask, 'chokyoshi_code'] = None

        # hasso_datetime
        meta_data['hasso_datetime'] = pd.to_datetime(self._race_data['kaisai_date'] + self._race_data['hasso_jikoku'], format='%Y%m%d%H%M')

        # mask
        meta_data['mask'] = self._race_data['ijo_kubun_code'].isin(['0', '4', '5', '6', '7'])

        return meta_data
    
    def _process_chokyo_meta_data(self):
        meta_data = pd.DataFrame(index=self._chokyo_data.index)

        # ketto_toroku_number
        meta_data['ketto_toroku_number'] = self._chokyo_data['ketto_toroku_number']
        mask = (meta_data['ketto_toroku_number'] == '0000000000')
        meta_data.loc[mask, 'ketto_toroku_number'] = None

        # chokyo_datetime
        meta_data['chokyo_datetime'] = pd.to_datetime(self._chokyo_data['chokyo_date'] + self._chokyo_data['chokyo_jikoku'], format='%Y%m%d%H%M')
        
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
    
    def _fit_chokyo_scaler(self) -> Dict[str, Dict[str, float]]:
        """調教データ用のスケーラーパラメータを計算"""
        scaler_params = {}
        
        # エイリアスごとに特徴量をグループ化
        alias_to_features = {}
        for column in self._chokyo_numerical_data.columns:
            alias = self._feature_aliases.get(column, column)
            if alias not in alias_to_features:
                alias_to_features[alias] = []
            alias_to_features[alias].append(column)
        
        # エイリアスごとにスケーラーをフィット
        for alias, features in alias_to_features.items():
            # 同じエイリアスを持つ全ての特徴量の値を結合
            all_values = []
            for feature in features:
                values = self._chokyo_numerical_data[feature].dropna().values
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

    def _fit_chokyo_encoder(self) -> Dict[str, Dict[Any, int]]:
        """調教データ用のエンコーダーパラメータを計算"""
        encoder_params = {}
        
        # エイリアスごとに特徴量をグループ化
        alias_to_features = {}
        for column in self._chokyo_categorical_data.columns:
            alias = self._feature_aliases.get(column, column)
            if alias not in alias_to_features:
                alias_to_features[alias] = []
            alias_to_features[alias].append(column)
        
        # エイリアスごとにエンコーダーをフィット
        for alias, features in alias_to_features.items():
            # 同じエイリアスを持つ全ての特徴量の一意な値を結合
            all_values = set()
            for feature in features:
                values = self._chokyo_categorical_data[feature].dropna().unique()
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
    
    def _scale_chokyo_numerical_features(self) -> pd.DataFrame:
        """調教データの数値特徴量をスケーリング"""
        scaled_data = pd.DataFrame(index=self._chokyo_numerical_data.index)
        for column in self._chokyo_numerical_data.columns:
            alias = self._feature_aliases.get(column, column)
            mean = self._chokyo_scaler_params[alias]['mean']
            std = self._chokyo_scaler_params[alias]['std']
            scaled_data[column] = (self._chokyo_numerical_data[column] - mean) / std
        return scaled_data

    def _encode_chokyo_categorical_features(self) -> pd.DataFrame:
        """調教データのカテゴリカル特徴量をエンコーディング"""
        encoded_data = pd.DataFrame(index=self._chokyo_categorical_data.index)
        for column in self._chokyo_categorical_data.columns:
            alias = self._feature_aliases.get(column, column)
            mapping = self._chokyo_encoder_params[alias]
            encoded_data[column] = self._chokyo_categorical_data[column].map(mapping).fillna(0).astype(np.int64)
        return encoded_data

    def _precompute_history_groups(self):
        history_groups = {}

        for key in ['ketto_toroku_number', 'kishu_name_short', 'chokyoshi_code']:
            valid_meta_data = self._meta_data[self._meta_data[key].notna()].sort_values('hasso_datetime')
            history_groups[key] = {value: data for value, data in valid_meta_data.groupby(valid_meta_data[key])}

        self._history_groups = history_groups

    def _compute_item(self, idx: int) -> Dict[str, Any]:
        race_id = self._target_race_ids[idx]

        data = {
            'race_id': race_id,
            'x_num': {key: np.full((self._num_horses,), np.nan, dtype=np.float32) for key in [
                'umaban',
                'wakuban',
                'barei',
                'futan_juryo',
                'zogen_sa',
                'bataiju',
                'kyori',
                'toroku_tosu',
                'hon_shokin_1',
                'hon_shokin_2',
                'hon_shokin_3',
                'hon_shokin_4',
                'hon_shokin_5',
            ]},
            'x_cat': {key: np.zeros((self._num_horses,), dtype=np.int64) for key in [
                # 'ketto_toroku_number',
                'uma_kigo_code',
                'seibetsu_code',
                'hinshu_code',
                'keiro_code',
                'tozai_shozoku_code',
                # 'chokyoshi_code',
                'banushi_code',
                'blinker_shiyo_kubun',
                # 'kishu_name_short',
                'kishu_minarai_code',
                'keibajo_code',
                'track_code',
                'course_kubun',
                'tenko_code',
                'baba_jotai_code',
            ]},
            'mask': np.zeros((self._num_horses,), dtype=np.bool_),
            'sequence_data': {
                'horse_history': {
                    'x_num': {key: np.full((self._num_horses, self._horse_history_length), np.nan, dtype=np.float32) for key in [
                        'umaban',
                        'wakuban',
                        'barei',
                        'futan_juryo',
                        'zogen_sa',
                        'soha_sokudo',
                        'bataiju',
                        'kakutei_chakujun',
                        'kyori',
                        'toroku_tosu',
                        'hon_shokin_1',
                        'hon_shokin_2',
                        'hon_shokin_3',
                        'hon_shokin_4',
                        'hon_shokin_5',
                        'corner_juni_1',
                        'corner_juni_2',
                        'corner_juni_3',
                        'corner_juni_4',
                        'kakutoku_hon_shokin',
                        'ushiro_3_furlongs_time',
                    ]},
                    'x_cat': {key: np.zeros((self._num_horses, self._horse_history_length), dtype=np.int64) for key in [
                        'uma_kigo_code',
                        'tozai_shozoku_code',
                        'chokyoshi_code',
                        'banushi_code',
                        'blinker_shiyo_kubun',
                        'kishu_name_short',
                        'kishu_minarai_code',
                        'keibajo_code',
                        'track_code',
                        'course_kubun',
                        'tenko_code',
                        'baba_jotai_code',
                        'konkai_race_kyakushitsu_hantei',
                    ]},
                    'mask': np.zeros((self._num_horses, self._horse_history_length), dtype=np.bool_),
                },
                'jockey_history': {
                    'x_num': {key: np.full((self._num_horses, self._jockey_history_length), np.nan, dtype=np.float32) for key in [
                        'umaban',
                        'wakuban',
                        'barei',
                        'futan_juryo',
                        'zogen_sa',
                        'soha_sokudo',
                        'bataiju',
                        'kakutei_chakujun',
                        'kyori',
                        'toroku_tosu',
                        'hon_shokin_1',
                        'hon_shokin_2',
                        'hon_shokin_3',
                        'hon_shokin_4',
                        'hon_shokin_5',
                        'corner_juni_1',
                        'corner_juni_2',
                        'corner_juni_3',
                        'corner_juni_4',
                        'kakutoku_hon_shokin',
                        'ushiro_3_furlongs_time',
                    ]},
                    'x_cat': {key: np.zeros((self._num_horses, self._jockey_history_length), dtype=np.int64) for key in [
                        'ketto_toroku_number',
                        'uma_kigo_code',
                        'tozai_shozoku_code',
                        'chokyoshi_code',
                        'banushi_code',
                        'blinker_shiyo_kubun',
                        'kishu_minarai_code',
                        'keibajo_code',
                        'track_code',
                        'course_kubun',
                        'tenko_code',
                        'baba_jotai_code',
                        'konkai_race_kyakushitsu_hantei',
                    ]},
                    'mask': np.zeros((self._num_horses, self._jockey_history_length), dtype=np.bool_),
                },
                'trainer_history': {
                    'x_num': {key: np.full((self._num_horses, self._trainer_history_length), np.nan, dtype=np.float32) for key in [
                        'umaban',
                        'wakuban',
                        'barei',
                        'futan_juryo',
                        'zogen_sa',
                        'soha_sokudo',
                        'bataiju',
                        'kakutei_chakujun',
                        'kyori',
                        'toroku_tosu',
                        'hon_shokin_1',
                        'hon_shokin_2',
                        'hon_shokin_3',
                        'hon_shokin_4',
                        'hon_shokin_5',
                        'corner_juni_1',
                        'corner_juni_2',
                        'corner_juni_3',
                        'corner_juni_4',
                        'kakutoku_hon_shokin',
                        'ushiro_3_furlongs_time',
                    ]},
                    'x_cat': {key: np.zeros((self._num_horses, self._trainer_history_length), dtype=np.int64) for key in [
                        'ketto_toroku_number',
                        'uma_kigo_code',
                        'tozai_shozoku_code',
                        'banushi_code',
                        'blinker_shiyo_kubun',
                        'kishu_name_short',
                        'kishu_minarai_code',
                        'keibajo_code',
                        'track_code',
                        'course_kubun',
                        'tenko_code',
                        'baba_jotai_code',
                        'konkai_race_kyakushitsu_hantei',
                    ]},
                    'mask': np.zeros((self._num_horses, self._trainer_history_length), dtype=np.bool_),
                },
                'chokyo_history': {
                    'x_num': {key: np.full((self._num_horses, self._chokyo_history_length), np.nan, dtype=np.float32) for key in [
                        'time_gokei_10_furlongs',
                        'time_gokei_9_furlongs', 
                        'time_gokei_8_furlongs',
                        'time_gokei_7_furlongs',
                        'time_gokei_6_furlongs',
                        'time_gokei_5_furlongs',
                        'time_gokei_4_furlongs',
                        'time_gokei_3_furlongs',
                        'time_gokei_2_furlongs',
                        'lap_time_2000_1800',
                        'lap_time_1800_1600',
                        'lap_time_1600_1400',
                        'lap_time_1400_1200',
                        'lap_time_1200_1000',
                        'lap_time_1000_800',
                        'lap_time_800_600',
                        'lap_time_600_400',
                        'lap_time_400_200',
                        'lap_time_200_0',
                    ]},
                    'x_cat': {key: np.zeros((self._num_horses, self._chokyo_history_length), dtype=np.int64) for key in [
                        'training_center_kubun',
                        'course',
                        'baba_mawari',
                    ]},
                    'mask': np.zeros((self._num_horses, self._chokyo_history_length), dtype=np.bool_),
                }
            },
            'target': np.full((self._num_horses, 3), np.nan, dtype=np.float32),
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
            data['mask'][horse_idx] = horse_meta_data['mask']

            # 履歴データの設定
            current_race_datetime = horse_meta_data['hasso_datetime']

            ketto_toroku_number = horse_meta_data['ketto_toroku_number']
            history_meta_data = self._history_groups['ketto_toroku_number'].get(ketto_toroku_number)
            if history_meta_data is not None:
                history_meta_data = history_meta_data[(
                    (history_meta_data['hasso_datetime'] < current_race_datetime - pd.Timedelta(hours=self._exclude_hours_before_race)) &
                    (history_meta_data['hasso_datetime'] >= current_race_datetime - pd.Timedelta(days=self._history_days))
                )].sort_values('hasso_datetime', ascending=False).head(self._horse_history_length)
            
                scaled_history_data = self._scaled_numerical_data.loc[history_meta_data.index]
                for key in data['sequence_data']['horse_history']['x_num'].keys():
                    data['sequence_data']['horse_history']['x_num'][key][horse_idx, :len(history_meta_data)] = scaled_history_data[key].values

                encoded_history_data = self._encoded_categorical_data.loc[history_meta_data.index]
                for key in data['sequence_data']['horse_history']['x_cat'].keys():
                    data['sequence_data']['horse_history']['x_cat'][key][horse_idx, :len(history_meta_data)] = encoded_history_data[key].values

                data['sequence_data']['horse_history']['mask'][horse_idx, :len(history_meta_data)] = True

            # 騎手履歴データの設定
            kishu_name = horse_meta_data['kishu_name_short']
            jockey_history_meta_data = self._history_groups['kishu_name_short'].get(kishu_name)
            if jockey_history_meta_data is not None:
                jockey_history_meta_data = jockey_history_meta_data[(
                    (jockey_history_meta_data['hasso_datetime'] < current_race_datetime - pd.Timedelta(hours=self._exclude_hours_before_race)) &
                    (jockey_history_meta_data['hasso_datetime'] >= current_race_datetime - pd.Timedelta(days=self._history_days))
                )].sort_values('hasso_datetime', ascending=False).head(self._jockey_history_length)
            
                scaled_jockey_history_data = self._scaled_numerical_data.loc[jockey_history_meta_data.index]
                for key in data['sequence_data']['jockey_history']['x_num'].keys():
                    data['sequence_data']['jockey_history']['x_num'][key][horse_idx, :len(jockey_history_meta_data)] = scaled_jockey_history_data[key].values

                encoded_jockey_history_data = self._encoded_categorical_data.loc[jockey_history_meta_data.index]
                for key in data['sequence_data']['jockey_history']['x_cat'].keys():
                    data['sequence_data']['jockey_history']['x_cat'][key][horse_idx, :len(jockey_history_meta_data)] = encoded_jockey_history_data[key].values

                data['sequence_data']['jockey_history']['mask'][horse_idx, :len(jockey_history_meta_data)] = True

            # 調教師履歴データの設定
            chokyoshi_code = horse_meta_data['chokyoshi_code']
            trainer_history_meta_data = self._history_groups['chokyoshi_code'].get(chokyoshi_code)
            if trainer_history_meta_data is not None:
                trainer_history_meta_data = trainer_history_meta_data[(
                    (trainer_history_meta_data['hasso_datetime'] < current_race_datetime - pd.Timedelta(hours=self._exclude_hours_before_race)) &
                    (trainer_history_meta_data['hasso_datetime'] >= current_race_datetime - pd.Timedelta(days=self._history_days))
                )].sort_values('hasso_datetime', ascending=False).head(self._trainer_history_length)
            
                scaled_trainer_history_data = self._scaled_numerical_data.loc[trainer_history_meta_data.index]
                for key in data['sequence_data']['trainer_history']['x_num'].keys():
                    data['sequence_data']['trainer_history']['x_num'][key][horse_idx, :len(trainer_history_meta_data)] = scaled_trainer_history_data[key].values

                encoded_trainer_history_data = self._encoded_categorical_data.loc[trainer_history_meta_data.index]
                for key in data['sequence_data']['trainer_history']['x_cat'].keys():
                    data['sequence_data']['trainer_history']['x_cat'][key][horse_idx, :len(trainer_history_meta_data)] = encoded_trainer_history_data[key].values

                data['sequence_data']['trainer_history']['mask'][horse_idx, :len(trainer_history_meta_data)] = True

            # 調教履歴データの設定
            if ketto_toroku_number in self._chokyo_meta_data['ketto_toroku_number'].values:
                # 該当する馬の調教データを取得
                chokyo_meta_data = self._chokyo_meta_data[self._chokyo_meta_data['ketto_toroku_number'] == ketto_toroku_number]

                # レース日時より前の調教データのみを取得
                chokyo_meta_data = chokyo_meta_data[
                    (chokyo_meta_data['chokyo_datetime'] < current_race_datetime - pd.Timedelta(hours=self._exclude_hours_before_race)) &
                    (chokyo_meta_data['chokyo_datetime'] >= current_race_datetime - pd.Timedelta(days=self._history_days))
                ].sort_values('chokyo_datetime', ascending=False).head(self._chokyo_history_length)

                if len(chokyo_meta_data) > 0:
                    # スケーリング済みの数値データを設定
                    scaled_chokyo_history_data = self._scaled_chokyo_numerical_data.loc[chokyo_meta_data.index]
                    for key in data['sequence_data']['chokyo_history']['x_num'].keys():
                        if key in scaled_chokyo_history_data.columns:
                            data['sequence_data']['chokyo_history']['x_num'][key][horse_idx, :len(chokyo_meta_data)] = scaled_chokyo_history_data[key].values

                    # エンコーディング済みのカテゴリカルデータを設定
                    encoded_chokyo_history_data = self._encoded_chokyo_categorical_data.loc[chokyo_meta_data.index]
                    for key in data['sequence_data']['chokyo_history']['x_cat'].keys():
                        if key in encoded_chokyo_history_data.columns:
                            data['sequence_data']['chokyo_history']['x_cat'][key][horse_idx, :len(chokyo_meta_data)] = encoded_chokyo_history_data[key].values

                    data['sequence_data']['chokyo_history']['mask'][horse_idx, :len(chokyo_meta_data)] = True

            # ターゲットの設定
            # 1着以内フラグ、2着以内フラグ、3着以内フラグ
            chakujun = horse_meta_data['kakutei_chakujun']
            if pd.notna(chakujun) and chakujun > 0:
                # 1着以内フラグ (1st place)
                data['target'][horse_idx, 0] = 1.0 if chakujun <= 1 else 0.0
                # 2着以内フラグ (within 2nd place)
                data['target'][horse_idx, 1] = 1.0 if chakujun <= 2 else 0.0
                # 3着以内フラグ (within 3rd place)
                data['target'][horse_idx, 2] = 1.0 if chakujun <= 3 else 0.0
            else:
                # Invalid finishing position - set all targets to NaN
                data['target'][horse_idx, :] = np.nan

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
