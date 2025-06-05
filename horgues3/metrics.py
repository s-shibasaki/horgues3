import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from scipy.stats import spearmanr, kendalltau
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseMetric:
    """
    メトリクス計算の基底クラス
    
    Args:
        rankings: 正解の着順データ (num_races, num_horses) - 1-indexed、0は無効
        mask: 有効な馬のマスク (num_races, num_horses) - 1=有効、0=無効
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None):
        self.rankings = rankings
        if mask is None:
            self.mask = (rankings > 0).astype(bool)
        else:
            self.mask = mask.astype(bool)
    
    def __call__(self, scores: np.ndarray) -> Dict[str, float]:
        """
        メトリクスを計算する
        
        Args:
            scores: 予測スコア (num_races, num_horses)
            
        Returns:
            計算されたメトリクス値の辞書
        """
        raise NotImplementedError


class AccuracyMetric(BaseMetric):
    """
    着順予測精度メトリクス
    
    概要:
        予測スコアの順位と実際の着順がどの程度一致するかを測定する。
        1位、複勝（1-3位）、上位k位などの的中率を計算。
    
    計算方法:
        - 1位的中率: 予測1位と実際1位が一致するレースの割合
        - 複勝的中率: 予測上位3位が実際の上位3位に含まれる的中数の割合
        - 上位k位的中率: 予測上位k位と実際の上位k位の重複数の平均
    
    値の読み方:
        - 0.0〜1.0の範囲
        - 1.0に近いほど予測精度が高い
        - ランダム予測の場合、1位的中率は1/馬数、複勝的中率は3/馬数程度
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None, 
                 top_k_list: List[int] = [1, 3, 5]):
        super().__init__(rankings, mask)
        self.top_k_list = top_k_list
    
    def __call__(self, scores: np.ndarray) -> Dict[str, float]:
        num_races = scores.shape[0]
        metrics = {}
        
        for k in self.top_k_list:
            total_hits = 0
            total_possible = 0
            valid_races = 0
            
            for race_idx in range(num_races):
                race_scores = scores[race_idx]
                race_rankings = self.rankings[race_idx]
                race_mask = self.mask[race_idx]
                
                # 有効な馬のみを抽出
                valid_horses = np.where(race_mask)[0]
                if len(valid_horses) < k:
                    continue
                
                valid_scores = race_scores[valid_horses]
                valid_rankings = race_rankings[valid_horses]
                
                # 無効な着順を除外
                valid_rank_mask = valid_rankings > 0
                if not valid_rank_mask.any() or valid_rank_mask.sum() < k:
                    continue
                
                final_scores = valid_scores[valid_rank_mask]
                final_rankings = valid_rankings[valid_rank_mask]
                
                # 予測上位k位の馬のインデックス
                pred_top_k = np.argsort(final_scores)[::-1][:k]
                
                # 実際の上位k位の馬のインデックス
                actual_top_k = np.argsort(final_rankings)[:k]
                
                # 的中数を計算
                hits = len(set(pred_top_k) & set(actual_top_k))
                total_hits += hits
                total_possible += k
                valid_races += 1
            
            if valid_races > 0:
                if k == 1:
                    # 1位的中率は0または1の平均
                    metrics[f'win_accuracy'] = total_hits / valid_races
                else:
                    # 上位k位の平均的中率
                    metrics[f'top_{k}_accuracy'] = total_hits / total_possible
            
        return metrics


class RankCorrelationMetric(BaseMetric):
    """
    順位相関メトリクス
    
    概要:
        予測スコアによる順位と実際の着順の相関を測定する。
        スピアマン順位相関係数とケンドールのタウを計算。
    
    計算方法:
        - スピアマン相関: ピアソン相関の順位版、線形関係を測定
        - ケンドールのタウ: 順位の一致・不一致ペア数から計算、単調関係を測定
        - 各レースで計算し、有効なレースの平均値を取る
    
    値の読み方:
        - -1.0〜1.0の範囲
        - 1.0に近いほど予測順位と実際の順位が正の相関
        - 0.0はランダム、-1.0は完全に逆の順位
        - 0.3以上で弱い相関、0.5以上で中程度の相関
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None):
        super().__init__(rankings, mask)
    
    def __call__(self, scores: np.ndarray) -> Dict[str, float]:
        num_races = scores.shape[0]
        spearman_corrs = []
        kendall_corrs = []
        
        for race_idx in range(num_races):
            race_scores = scores[race_idx]
            race_rankings = self.rankings[race_idx]
            race_mask = self.mask[race_idx]
            
            # 有効な馬のみを抽出
            valid_horses = np.where(race_mask)[0]
            if len(valid_horses) < 3:  # 相関計算には最低3頭必要
                continue
            
            valid_scores = race_scores[valid_horses]
            valid_rankings = race_rankings[valid_horses]
            
            # 無効な着順を除外
            valid_rank_mask = valid_rankings > 0
            if valid_rank_mask.sum() < 3:
                continue
            
            final_scores = valid_scores[valid_rank_mask]
            final_rankings = valid_rankings[valid_rank_mask]
            
            # 予測順位を計算（スコアが高いほど上位）
            pred_ranks = len(final_scores) + 1 - np.argsort(np.argsort(final_scores)) - 1
            
            try:
                # スピアマン相関
                spear_corr, _ = spearmanr(pred_ranks, final_rankings)
                if not np.isnan(spear_corr):
                    spearman_corrs.append(spear_corr)
                
                # ケンドールのタウ
                kendall_corr, _ = kendalltau(pred_ranks, final_rankings)
                if not np.isnan(kendall_corr):
                    kendall_corrs.append(kendall_corr)
                    
            except Exception as e:
                continue
        
        metrics = {}
        if spearman_corrs:
            metrics['spearman_corr'] = np.mean(spearman_corrs)
            metrics['spearman_corr_std'] = np.std(spearman_corrs)
        
        if kendall_corrs:
            metrics['kendall_tau'] = np.mean(kendall_corrs)
            metrics['kendall_tau_std'] = np.std(kendall_corrs)
        
        return metrics


class NDCGMetric(BaseMetric):
    """
    正規化割引累積利得（NDCG）メトリクス
    
    概要:
        情報検索分野で使用される評価指標。上位の順位により大きな重みを付けて
        予測順位の品質を評価する。競馬では上位の的中がより重要なため適している。
    
    計算方法:
        - DCG = Σ(rel_i / log2(i+1)) where rel_iは位置iでの関連度
        - IDCG = 理想的な順序でのDCG（完璧な予測時のDCG）
        - NDCG = DCG / IDCG で正規化
        - 関連度は着順の逆数を使用（1位=最高関連度）
    
    値の読み方:
        - 0.0〜1.0の範囲
        - 1.0に近いほど上位順位の予測精度が高い
        - 0.5以上で良好、0.7以上で優秀な予測
        - 上位重視の評価なので、下位の誤差より上位の誤差を重く評価
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None, 
                 k: int = 10):
        super().__init__(rankings, mask)
        self.k = k
    
    def _dcg_at_k(self, relevances: np.ndarray, k: int) -> float:
        """DCG@kを計算"""
        k = min(k, len(relevances))
        if k == 0:
            return 0.0
        
        dcg = relevances[0]
        for i in range(1, k):
            dcg += relevances[i] / np.log2(i + 1)
        return dcg
    
    def __call__(self, scores: np.ndarray) -> Dict[str, float]:
        num_races = scores.shape[0]
        ndcg_scores = []
        
        for race_idx in range(num_races):
            race_scores = scores[race_idx]
            race_rankings = self.rankings[race_idx]
            race_mask = self.mask[race_idx]
            
            # 有効な馬のみを抽出
            valid_horses = np.where(race_mask)[0]
            if len(valid_horses) < 2:
                continue
            
            valid_scores = race_scores[valid_horses]
            valid_rankings = race_rankings[valid_horses]
            
            # 無効な着順を除外
            valid_rank_mask = valid_rankings > 0
            if valid_rank_mask.sum() < 2:
                continue
            
            final_scores = valid_scores[valid_rank_mask]
            final_rankings = valid_rankings[valid_rank_mask]
            
            # 関連度を計算（着順の逆数、1位が最大）
            max_rank = len(final_rankings)
            relevances = (max_rank + 1 - final_rankings) / max_rank
            
            # 予測順位でソート
            pred_order = np.argsort(final_scores)[::-1]
            pred_relevances = relevances[pred_order]
            
            # 理想的な順序（関連度の降順）
            ideal_relevances = np.sort(relevances)[::-1]
            
            # DCG計算
            dcg = self._dcg_at_k(pred_relevances, self.k)
            idcg = self._dcg_at_k(ideal_relevances, self.k)
            
            if idcg > 0:
                ndcg = dcg / idcg
                ndcg_scores.append(ndcg)
        
        metrics = {}
        if ndcg_scores:
            metrics[f'ndcg@{self.k}'] = np.mean(ndcg_scores)
            metrics[f'ndcg@{self.k}_std'] = np.std(ndcg_scores)
        
        return metrics


class ReciprocalRankMetric(BaseMetric):
    """
    平均逆順位（MRR）メトリクス
    
    概要:
        1位の馬が予測順位の何位に位置するかを評価する。
        予測1位なら1.0、予測2位なら0.5、予測3位なら0.33...となる。
    
    計算方法:
        - 各レースで1位馬の予測順位を特定
        - 逆数を計算（1/予測順位）
        - 全レースの平均を取る
    
    値の読み方:
        - 0.0〜1.0の範囲
        - 1.0に近いほど1位馬を上位に予測できている
        - 0.5なら平均的に1位馬を2位に予測
        - ランダム予測なら約1/馬数の値
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None):
        super().__init__(rankings, mask)
    
    def __call__(self, scores: np.ndarray) -> Dict[str, float]:
        num_races = scores.shape[0]
        reciprocal_ranks = []
        
        for race_idx in range(num_races):
            race_scores = scores[race_idx]
            race_rankings = self.rankings[race_idx]
            race_mask = self.mask[race_idx]
            
            # 有効な馬のみを抽出
            valid_horses = np.where(race_mask)[0]
            if len(valid_horses) < 2:
                continue
            
            valid_scores = race_scores[valid_horses]
            valid_rankings = race_rankings[valid_horses]
            
            # 無効な着順を除外
            valid_rank_mask = valid_rankings > 0
            if not valid_rank_mask.any():
                continue
            
            final_scores = valid_scores[valid_rank_mask]
            final_rankings = valid_rankings[valid_rank_mask]
            
            # 1位馬を特定
            winner_idx = np.where(final_rankings == 1)[0]
            if len(winner_idx) == 0:
                continue
            
            winner_idx = winner_idx[0]
            winner_score = final_scores[winner_idx]
            
            # 予測順位を計算（何位に予測されたか）
            pred_rank = np.sum(final_scores > winner_score) + 1
            reciprocal_rank = 1.0 / pred_rank
            reciprocal_ranks.append(reciprocal_rank)
        
        metrics = {}
        if reciprocal_ranks:
            metrics['mrr'] = np.mean(reciprocal_ranks)
            metrics['mrr_std'] = np.std(reciprocal_ranks)
        
        return metrics


class RankErrorMetric(BaseMetric):
    """
    順位誤差メトリクス
    
    概要:
        予測順位と実際の着順の差を直接測定する。
        平均絶対誤差（MAE）と平均二乗誤差（MSE）を計算。
    
    計算方法:
        - 各馬について予測順位と実際の着順の差を計算
        - MAE = |予測順位 - 実際着順|の平均
        - MSE = (予測順位 - 実際着順)²の平均
        - RMSE = √MSE
    
    値の読み方:
        - 0に近いほど順位予測が正確
        - MAE=1.0なら平均1順位の誤差
        - 完璧な予測なら全て0.0
        - ランダム予測なら馬数に依存した値
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None):
        super().__init__(rankings, mask)
    
    def __call__(self, scores: np.ndarray) -> Dict[str, float]:
        num_races = scores.shape[0]
        all_errors = []
        
        for race_idx in range(num_races):
            race_scores = scores[race_idx]
            race_rankings = self.rankings[race_idx]
            race_mask = self.mask[race_idx]
            
            # 有効な馬のみを抽出
            valid_horses = np.where(race_mask)[0]
            if len(valid_horses) < 2:
                continue
            
            valid_scores = race_scores[valid_horses]
            valid_rankings = race_rankings[valid_horses]
            
            # 無効な着順を除外
            valid_rank_mask = valid_rankings > 0
            if valid_rank_mask.sum() < 2:
                continue
            
            final_scores = valid_scores[valid_rank_mask]
            final_rankings = valid_rankings[valid_rank_mask]
            
            # 予測順位を計算
            pred_ranks = len(final_scores) + 1 - np.argsort(np.argsort(final_scores)) - 1
            
            # 順位誤差を計算
            errors = np.abs(pred_ranks - final_rankings)
            all_errors.extend(errors)
        
        metrics = {}
        if all_errors:
            all_errors = np.array(all_errors)
            metrics['rank_mae'] = np.mean(all_errors)
            metrics['rank_mse'] = np.mean(all_errors ** 2)
            metrics['rank_rmse'] = np.sqrt(metrics['rank_mse'])
            metrics['rank_std'] = np.std(all_errors)
        
        return metrics


class PairwiseAccuracyMetric(BaseMetric):
    """
    ペアワイズ精度メトリクス
    
    概要:
        全ての馬のペアについて、どちらが上位に来るかの予測精度を測定。
        「AがBより上位」という予測が実際の着順と一致する割合を計算。
    
    計算方法:
        - 各レース内の全ての馬のペア(i,j)について判定
        - 予測: スコアi > スコアj なら馬iが上位
        - 実際: 着順i < 着順j なら馬iが上位
        - 一致率 = 正しく予測したペア数 / 全ペア数
    
    値の読み方:
        - 0.0〜1.0の範囲
        - 0.5はランダム予測（コイン投げと同じ）
        - 0.7以上で良好、0.8以上で優秀な予測
        - 順位相関よりも直感的で理解しやすい指標
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None):
        super().__init__(rankings, mask)
    
    def __call__(self, scores: np.ndarray) -> Dict[str, float]:
        num_races = scores.shape[0]
        total_pairs = 0
        correct_pairs = 0
        
        for race_idx in range(num_races):
            race_scores = scores[race_idx]
            race_rankings = self.rankings[race_idx]
            race_mask = self.mask[race_idx]
            
            # 有効な馬のみを抽出
            valid_horses = np.where(race_mask)[0]
            if len(valid_horses) < 2:
                continue
            
            valid_scores = race_scores[valid_horses]
            valid_rankings = race_rankings[valid_horses]
            
            # 無効な着順を除外
            valid_rank_mask = valid_rankings > 0
            if valid_rank_mask.sum() < 2:
                continue
            
            final_scores = valid_scores[valid_rank_mask]
            final_rankings = valid_rankings[valid_rank_mask]
            
            # 全ペアについて判定
            n_horses = len(final_scores)
            for i in range(n_horses):
                for j in range(i + 1, n_horses):
                    # 予測: スコアが高い方が上位
                    pred_i_better = final_scores[i] > final_scores[j]
                    
                    # 実際: 着順が小さい方が上位
                    actual_i_better = final_rankings[i] < final_rankings[j]
                    
                    if pred_i_better == actual_i_better:
                        correct_pairs += 1
                    
                    total_pairs += 1
        
        metrics = {}
        if total_pairs > 0:
            metrics['pairwise_accuracy'] = correct_pairs / total_pairs
        
        return metrics


class OddsBasedMetric(BaseMetric):
    """
    オッズデータベースのメトリクス基底クラス
    
    Args:
        rankings: 正解の着順データ (num_races, num_horses) - 1-indexed、0は無効
        mask: 有効な馬のマスク (num_races, num_horses) - 1=有効、0=無効
        odds_data: オッズデータの辞書 (bet_type -> DataFrame)
        race_ids: レースIDのリスト (num_races,)
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None, 
                 odds_data: Dict[str, pd.DataFrame] = None, race_ids: np.ndarray = None):
        super().__init__(rankings, mask)
        self.odds_data = odds_data or {}
        self.race_ids = race_ids
        
        # レースID -> インデックスのマッピングを作成
        self.race_id_to_idx = {}
        if race_ids is not None:
            for idx, race_id in enumerate(race_ids):
                self.race_id_to_idx[race_id] = idx
    
    def _get_race_odds(self, race_id: str, bet_type: str = 'tansho') -> Optional[pd.DataFrame]:
        """指定されたレースのオッズデータを取得"""
        if bet_type not in self.odds_data:
            return None
        
        odds_df = self.odds_data[bet_type]
        race_odds = odds_df[odds_df['race_id'] == race_id]
        return race_odds if not race_odds.empty else None


class MarketCorrelationMetric(OddsBasedMetric):
    """
    市場予測（オッズ）との相関メトリクス
    
    概要:
        予測スコアと市場のオッズ（人気）がどの程度相関するかを測定。
        高い相関は市場と同様の予測、低い相関は独自の視点を示す。
    
    計算方法:
        - オッズを確率に変換（確率 = 1/オッズ、正規化）
        - 予測スコアと市場確率のスピアマン相関を計算
        - 人気順位との相関も計算
    
    値の読み方:
        - -1.0〜1.0の範囲
        - 0.7以上: 市場とほぼ同じ予測
        - 0.3-0.7: 市場の傾向を捉えつつ独自性あり
        - 0.3未満: 市場と異なる独自の予測
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None, 
                 odds_data: Dict[str, pd.DataFrame] = None, race_ids: np.ndarray = None):
        super().__init__(rankings, mask, odds_data, race_ids)
    
    def __call__(self, scores: np.ndarray) -> Dict[str, float]:
        if 'tansho' not in self.odds_data or self.race_ids is None:
            return {}
        
        market_corrs = []
        popularity_corrs = []
        
        for race_idx in range(len(self.race_ids)):
            race_id = self.race_ids[race_idx]
            race_odds = self._get_race_odds(race_id, 'tansho')
            
            if race_odds is None:
                continue
            
            race_scores = scores[race_idx]
            race_mask = self.mask[race_idx]
            
            # 有効な馬のスコアとオッズを対応付け
            valid_horses = np.where(race_mask)[0]
            if len(valid_horses) < 3:
                continue
            
            matched_scores = []
            matched_probs = []
            matched_popularity = []
            
            for horse_idx in valid_horses:
                horse_number = horse_idx + 1  # 0-indexed -> 1-indexed
                horse_odds_data = race_odds[race_odds['horse_number'] == horse_number]
                
                if not horse_odds_data.empty:
                    odds_val = horse_odds_data.iloc[0]['odds']
                    popularity = horse_odds_data.iloc[0]['popularity']
                    
                    if not pd.isna(odds_val) and odds_val > 0 and popularity > 0:
                        matched_scores.append(race_scores[horse_idx])
                        matched_probs.append(1.0 / odds_val)  # オッズを確率に変換
                        matched_popularity.append(popularity)
            
            if len(matched_scores) >= 3:
                try:
                    # 市場確率との相関
                    market_corr, _ = spearmanr(matched_scores, matched_probs)
                    if not np.isnan(market_corr):
                        market_corrs.append(market_corr)
                    
                    # 人気順位との相関（人気が高いほど小さい数値なので逆相関）
                    pop_corr, _ = spearmanr(matched_scores, [-p for p in matched_popularity])
                    if not np.isnan(pop_corr):
                        popularity_corrs.append(pop_corr)
                        
                except Exception:
                    continue
        
        metrics = {}
        if market_corrs:
            metrics['market_correlation'] = np.mean(market_corrs)
            metrics['market_correlation_std'] = np.std(market_corrs)
        
        if popularity_corrs:
            metrics['popularity_correlation'] = np.mean(popularity_corrs)
            metrics['popularity_correlation_std'] = np.std(popularity_corrs)
        
        return metrics


class MarketBeatMetric(OddsBasedMetric):
    """
    市場予測を上回る精度メトリクス
    
    概要:
        人気順位と予測順位のどちらが実際の着順により近いかを比較。
        市場より優秀な予測ができているかを直接測定。
    
    計算方法:
        - 各レースで人気順位と予測順位を実際の着順と比較
        - より正確な予測をした馬の数をカウント
        - Win率 = 予測が市場より正確だったレース数 / 全レース数
    
    値の読み方:
        - 0.0〜1.0の範囲
        - 0.5: 市場と同等の予測精度
        - 0.6以上: 市場を上回る予測精度
        - 0.4未満: 市場に劣る予測精度
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None, 
                 odds_data: Dict[str, pd.DataFrame] = None, race_ids: np.ndarray = None):
        super().__init__(rankings, mask, odds_data, race_ids)
    
    def __call__(self, scores: np.ndarray) -> Dict[str, float]:
        if 'tansho' not in self.odds_data or self.race_ids is None:
            return {}
        
        wins = 0
        ties = 0
        total_races = 0
        
        for race_idx in range(len(self.race_ids)):
            race_id = self.race_ids[race_idx]
            race_odds = self._get_race_odds(race_id, 'tansho')
            
            if race_odds is None:
                continue
            
            race_scores = scores[race_idx]
            race_rankings = self.rankings[race_idx]
            race_mask = self.mask[race_idx]
            
            # 有効な馬のデータを収集
            valid_horses = np.where(race_mask)[0]
            if len(valid_horses) < 3:
                continue
            
            matched_data = []
            for horse_idx in valid_horses:
                horse_number = horse_idx + 1
                horse_odds_data = race_odds[race_odds['horse_number'] == horse_number]
                
                if not horse_odds_data.empty and race_rankings[horse_idx] > 0:
                    popularity = horse_odds_data.iloc[0]['popularity']
                    if not pd.isna(popularity) and popularity > 0:
                        matched_data.append({
                            'score': race_scores[horse_idx],
                            'popularity': popularity,
                            'actual_rank': race_rankings[horse_idx]
                        })
            
            if len(matched_data) >= 3:
                # 予測順位と人気順位を計算
                scores_list = [d['score'] for d in matched_data]
                popularity_list = [d['popularity'] for d in matched_data]
                actual_ranks = [d['actual_rank'] for d in matched_data]
                
                pred_ranks = len(scores_list) + 1 - np.argsort(np.argsort(scores_list)) - 1
                market_ranks = np.argsort(np.argsort(popularity_list))
                
                # スピアマン相関で比較
                try:
                    pred_corr, _ = spearmanr(pred_ranks, actual_ranks)
                    market_corr, _ = spearmanr(market_ranks, actual_ranks)
                    
                    if not np.isnan(pred_corr) and not np.isnan(market_corr):
                        if pred_corr > market_corr:
                            wins += 1
                        elif pred_corr == market_corr:
                            ties += 1
                        total_races += 1
                except Exception:
                    continue
        
        metrics = {}
        if total_races > 0:
            metrics['market_beat_rate'] = wins / total_races
            metrics['market_tie_rate'] = ties / total_races
            metrics['market_win_rate'] = (wins + ties * 0.5) / total_races
        
        return metrics


class UnpopularHorseMetric(OddsBasedMetric):
    """
    穴馬発見能力メトリクス
    
    概要:
        人気薄の馬（高オッズ）を上位予測し、実際に好走した場合を評価。
        大穴馬の発見能力を測定する。
    
    計算方法:
        - 人気下位（例：8番人気以下）の馬の中で予測上位の馬を特定
        - その馬が実際に好走（例：3位以内）した場合を的中とカウント
        - 的中率と平均オッズを計算
    
    値の読み方:
        - unpopular_hit_rate: 0.0〜1.0、穴馬的中率
        - average_unpopular_odds: 的中した穴馬の平均オッズ
        - 高い的中率は穴馬発見能力の高さを示す
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None, 
                 odds_data: Dict[str, pd.DataFrame] = None, race_ids: np.ndarray = None,
                 unpopular_threshold: int = 8, good_finish_threshold: int = 3):
        super().__init__(rankings, mask, odds_data, race_ids)
        self.unpopular_threshold = unpopular_threshold  # この人気以下を穴馬とする
        self.good_finish_threshold = good_finish_threshold  # この着順以内を好走とする
    
    def __call__(self, scores: np.ndarray) -> Dict[str, float]:
        if 'tansho' not in self.odds_data or self.race_ids is None:
            return {}
        
        unpopular_predictions = 0  # 穴馬を上位予測した回数
        unpopular_hits = 0  # 穴馬予測が的中した回数
        hit_odds = []  # 的中した穴馬のオッズ
        
        for race_idx in range(len(self.race_ids)):
            race_id = self.race_ids[race_idx]
            race_odds = self._get_race_odds(race_id, 'tansho')
            
            if race_odds is None:
                continue
            
            race_scores = scores[race_idx]
            race_rankings = self.rankings[race_idx]
            race_mask = self.mask[race_idx]
            
            # 有効な馬のデータを収集
            valid_horses = np.where(race_mask)[0]
            if len(valid_horses) < self.unpopular_threshold:
                continue
            
            horse_data = []
            for horse_idx in valid_horses:
                horse_number = horse_idx + 1
                horse_odds_data = race_odds[race_odds['horse_number'] == horse_number]
                
                if not horse_odds_data.empty and race_rankings[horse_idx] > 0:
                    popularity = horse_odds_data.iloc[0]['popularity']
                    odds_val = horse_odds_data.iloc[0]['odds']
                    
                    if (not pd.isna(popularity) and popularity >= self.unpopular_threshold and
                        not pd.isna(odds_val) and odds_val > 0):
                        horse_data.append({
                            'horse_idx': horse_idx,
                            'score': race_scores[horse_idx],
                            'popularity': popularity,
                            'odds': odds_val,
                            'actual_rank': race_rankings[horse_idx]
                        })
            
            if len(horse_data) == 0:
                continue
            
            # 穴馬の中で最も予測スコアが高い馬を選択
            best_unpopular = max(horse_data, key=lambda x: x['score'])
            
            unpopular_predictions += 1
            
            # その馬が好走したかチェック
            if best_unpopular['actual_rank'] <= self.good_finish_threshold:
                unpopular_hits += 1
                hit_odds.append(best_unpopular['odds'])
        
        metrics = {}
        if unpopular_predictions > 0:
            metrics['unpopular_hit_rate'] = unpopular_hits / unpopular_predictions
            metrics['unpopular_prediction_count'] = unpopular_predictions
            
            if hit_odds:
                metrics['average_unpopular_odds'] = np.mean(hit_odds)
                metrics['max_unpopular_odds'] = np.max(hit_odds)
        
        return metrics


class FavoriteAvoidanceMetric(OddsBasedMetric):
    """
    人気馬回避能力メトリクス
    
    概要:
        人気馬（低オッズ）を下位予測し、実際に凡走した場合を評価。
        過大評価された人気馬を見抜く能力を測定。
    
    計算方法:
        - 人気上位（例：3番人気以内）の馬の中で予測下位の馬を特定
        - その馬が実際に凡走（例：5位以下）した場合を的中とカウント
        - 回避成功率を計算
    
    値の読み方:
        - favorite_avoidance_rate: 0.0〜1.0、人気馬回避成功率
        - 高い値は過大評価された人気馬を見抜く能力を示す
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None, 
                 odds_data: Dict[str, pd.DataFrame] = None, race_ids: np.ndarray = None,
                 favorite_threshold: int = 3, poor_finish_threshold: int = 5):
        super().__init__(rankings, mask, odds_data, race_ids)
        self.favorite_threshold = favorite_threshold  # この人気以内を人気馬とする
        self.poor_finish_threshold = poor_finish_threshold  # この着順以下を凡走とする
    
    def __call__(self, scores: np.ndarray) -> Dict[str, float]:
        if 'tansho' not in self.odds_data or self.race_ids is None:
            return {}
        
        favorite_avoidances = 0  # 人気馬を下位予測した回数
        avoidance_hits = 0  # 人気馬回避が的中した回数
        
        for race_idx in range(len(self.race_ids)):
            race_id = self.race_ids[race_idx]
            race_odds = self._get_race_odds(race_id, 'tansho')
            
            if race_odds is None:
                continue
            
            race_scores = scores[race_idx]
            race_rankings = self.rankings[race_idx]
            race_mask = self.mask[race_idx]
            
            # 有効な馬のデータを収集
            valid_horses = np.where(race_mask)[0]
            if len(valid_horses) < self.favorite_threshold:
                continue
            
            # 人気馬を特定
            favorite_horses = []
            for horse_idx in valid_horses:
                horse_number = horse_idx + 1
                horse_odds_data = race_odds[race_odds['horse_number'] == horse_number]
                
                if not horse_odds_data.empty and race_rankings[horse_idx] > 0:
                    popularity = horse_odds_data.iloc[0]['popularity']
                    
                    if not pd.isna(popularity) and popularity <= self.favorite_threshold:
                        favorite_horses.append({
                            'horse_idx': horse_idx,
                            'score': race_scores[horse_idx],
                            'popularity': popularity,
                            'actual_rank': race_rankings[horse_idx]
                        })
            
            if len(favorite_horses) == 0:
                continue
            
            # 人気馬の中で最も予測スコアが低い馬を選択（最も避けたい馬）
            worst_favorite = min(favorite_horses, key=lambda x: x['score'])
            
            favorite_avoidances += 1
            
            # その馬が凡走したかチェック
            if worst_favorite['actual_rank'] >= self.poor_finish_threshold:
                avoidance_hits += 1
        
        metrics = {}
        if favorite_avoidances > 0:
            metrics['favorite_avoidance_rate'] = avoidance_hits / favorite_avoidances
            metrics['favorite_avoidance_count'] = favorite_avoidances
        
        return metrics


class ExpectedValueMetric(OddsBasedMetric):
    """
    期待値メトリクス
    
    概要:
        予測に基づく投資戦略の期待値を計算。
        上位予測馬への投資が長期的に利益を生むかを評価。
    
    計算方法:
        - 予測上位k頭への等額投資を想定
        - 的中時の払い戻しと投資額から期待値を計算
        - 回収率 = 総払い戻し / 総投資額
    
    値の読み方:
        - 1.0以上: 利益が出る予測
        - 0.8-1.0: 市場に近い予測（JRA控除率考慮）
        - 0.8未満: 投資戦略として不適切
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None, 
                 odds_data: Dict[str, pd.DataFrame] = None, race_ids: np.ndarray = None,
                 top_k: int = 3):
        super().__init__(rankings, mask, odds_data, race_ids)
        self.top_k = top_k
    
    def __call__(self, scores: np.ndarray) -> Dict[str, float]:
        if 'tansho' not in self.odds_data or self.race_ids is None:
            return {}
        
        total_investment = 0
        total_return = 0
        win_count = 0
        total_bets = 0
        
        for race_idx in range(len(self.race_ids)):
            race_id = self.race_ids[race_idx]
            race_odds = self._get_race_odds(race_id, 'tansho')
            
            if race_odds is None:
                continue
            
            race_scores = scores[race_idx]
            race_rankings = self.rankings[race_idx]
            race_mask = self.mask[race_idx]
            
            # 有効な馬のデータを収集
            valid_horses = np.where(race_mask)[0]
            if len(valid_horses) < self.top_k:
                continue
            
            horse_data = []
            for horse_idx in valid_horses:
                horse_number = horse_idx + 1
                horse_odds_data = race_odds[race_odds['horse_number'] == horse_number]
                
                if not horse_odds_data.empty and race_rankings[horse_idx] > 0:
                    odds_val = horse_odds_data.iloc[0]['odds']
                    
                    if not pd.isna(odds_val) and odds_val > 0:
                        horse_data.append({
                            'horse_idx': horse_idx,
                            'score': race_scores[horse_idx],
                            'odds': odds_val,
                            'actual_rank': race_rankings[horse_idx]
                        })
            
            if len(horse_data) < self.top_k:
                continue
            
            # 予測上位k頭を選択
            top_horses = sorted(horse_data, key=lambda x: x['score'], reverse=True)[:self.top_k]
            
            # 各馬に100円投資と仮定
            bet_amount = 100
            race_investment = bet_amount * self.top_k
            race_return = 0
            
            for horse in top_horses:
                if horse['actual_rank'] == 1:  # 1位なら的中
                    race_return += bet_amount * horse['odds']
                    win_count += 1
                total_bets += 1
            
            total_investment += race_investment
            total_return += race_return
        
        metrics = {}
        if total_investment > 0:
            metrics['return_rate'] = total_return / total_investment
            metrics['win_rate'] = win_count / total_bets if total_bets > 0 else 0
            metrics['total_races'] = len([race_id for race_id in self.race_ids if self._get_race_odds(race_id, 'tansho') is not None])
            metrics['profit'] = total_return - total_investment
        
        return metrics
