import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class RankingMetrics:
    """競馬予測用の包括的なメトリクス計算クラス"""
    
    def __init__(self, rankings: np.ndarray, mask: np.ndarray = None):
        """
        Args:
            rankings: np.ndarray - 真の着順 (n_races, max_horses) - 1-indexed、0は無効
            mask: np.ndarray - 有効な馬のマスク (n_races, max_horses) - Trueが有効
        """
        self.rankings = rankings
        self.mask = mask if mask is not None else (rankings > 0)
        
    def __call__(self, scores: np.ndarray) -> Dict[str, Any]:
        """
        Args:
            scores: np.ndarray - 予測スコア (n_races, max_horses)
        Returns:
            Dict[str, Any] - 各種メトリクスの辞書
        """
        metrics = {}
        
        # 基本統計
        metrics.update(self._basic_stats(scores))
        
        # 順位相関
        metrics.update(self._ranking_correlations(scores))
        
        # 的中率メトリクス
        metrics.update(self._hit_rate_metrics(scores))
        
        # 順位精度メトリクス
        metrics.update(self._ranking_accuracy_metrics(scores))
        
        # NDCG (Normalized Discounted Cumulative Gain)
        metrics.update(self._ndcg_metrics(scores))
        
        return metrics
    
    def _basic_stats(self, scores: np.ndarray) -> Dict[str, float]:
        """基本統計量"""
        return {
            'n_races': len(scores),
            'avg_horses_per_race': self.mask.sum(axis=1).mean(),
            'score_mean': np.nanmean(scores[self.mask]),
            'score_std': np.nanstd(scores[self.mask]),
        }
    
    def _ranking_correlations(self, scores: np.ndarray) -> Dict[str, float]:
        """順位相関の計算"""
        spearman_corrs = []
        kendall_corrs = []
        
        for race_idx in range(len(scores)):
            race_mask = self.mask[race_idx]
            if race_mask.sum() < 2:  # 有効な馬が2頭未満の場合はスキップ
                continue
                
            race_scores = scores[race_idx][race_mask]
            race_rankings = self.rankings[race_idx][race_mask]
            
            # 無効な着順を除外
            valid_rank_mask = race_rankings > 0
            if valid_rank_mask.sum() < 2:
                continue
            
            final_scores = race_scores[valid_rank_mask]
            final_rankings = race_rankings[valid_rank_mask]
            
            # スピアマン相関 (順位相関)
            try:
                score_ranks = (-final_scores).argsort().argsort() + 1  # 高スコア=1位
                true_ranks = final_rankings
                
                spearman_corr = self._calculate_spearman(score_ranks, true_ranks)
                if not np.isnan(spearman_corr):
                    spearman_corrs.append(spearman_corr)
                
                # ケンドールのタウ
                kendall_tau = self._calculate_kendall_tau(score_ranks, true_ranks)
                if not np.isnan(kendall_tau):
                    kendall_corrs.append(kendall_tau)
                    
            except Exception as e:
                logger.debug(f"Correlation calculation failed for race {race_idx}: {e}")
                continue
        
        return {
            'spearman_corr_mean': np.mean(spearman_corrs) if spearman_corrs else 0.0,
            'spearman_corr_std': np.std(spearman_corrs) if spearman_corrs else 0.0,
            'kendall_tau_mean': np.mean(kendall_corrs) if kendall_corrs else 0.0,
            'kendall_tau_std': np.std(kendall_corrs) if kendall_corrs else 0.0,
            'n_valid_races_for_corr': len(spearman_corrs),
        }
    
    def _hit_rate_metrics(self, scores: np.ndarray) -> Dict[str, float]:
        """的中率メトリクス（1位、3位以内等）"""
        win_hits = 0  # 1位的中
        place_hits = 0  # 3位以内的中
        top5_hits = 0  # 5位以内的中
        valid_races = 0
        
        for race_idx in range(len(scores)):
            race_mask = self.mask[race_idx]
            if race_mask.sum() < 2:
                continue
                
            race_scores = scores[race_idx][race_mask]
            race_rankings = self.rankings[race_idx][race_mask]
            
            valid_rank_mask = race_rankings > 0
            if valid_rank_mask.sum() < 2:
                continue
            
            final_scores = race_scores[valid_rank_mask]
            final_rankings = race_rankings[valid_rank_mask]
            
            # 予測1位の馬（最高スコア）
            pred_winner_idx = np.argmax(final_scores)
            true_winner_rank = final_rankings[pred_winner_idx]
            
            valid_races += 1
            
            # 1位的中
            if true_winner_rank == 1:
                win_hits += 1
            
            # 3位以内的中
            if true_winner_rank <= 3:
                place_hits += 1
            
            # 5位以内的中
            if true_winner_rank <= 5:
                top5_hits += 1
        
        return {
            'win_hit_rate': win_hits / valid_races if valid_races > 0 else 0.0,
            'place_hit_rate': place_hits / valid_races if valid_races > 0 else 0.0,
            'top5_hit_rate': top5_hits / valid_races if valid_races > 0 else 0.0,
            'n_valid_races_for_hit_rate': valid_races,
        }
    
    def _ranking_accuracy_metrics(self, scores: np.ndarray) -> Dict[str, float]:
        """順位精度メトリクス"""
        mae_errors = []  # Mean Absolute Error for rankings
        top3_precision_scores = []
        top3_recall_scores = []
        
        for race_idx in range(len(scores)):
            race_mask = self.mask[race_idx]
            if race_mask.sum() < 3:  # 最低3頭必要
                continue
                
            race_scores = scores[race_idx][race_mask]
            race_rankings = self.rankings[race_idx][race_mask]
            
            valid_rank_mask = race_rankings > 0
            if valid_rank_mask.sum() < 3:
                continue
            
            final_scores = race_scores[valid_rank_mask]
            final_rankings = race_rankings[valid_rank_mask]
            
            # 予測順位を計算
            pred_ranks = (-final_scores).argsort().argsort() + 1
            
            # MAE (Mean Absolute Error) for ranking
            rank_mae = np.mean(np.abs(pred_ranks - final_rankings))
            mae_errors.append(rank_mae)
            
            # Top-3精度（予測上位3頭のうち実際に上位3位以内の馬の割合）
            pred_top3_mask = pred_ranks <= 3
            true_top3_mask = final_rankings <= 3
            
            if pred_top3_mask.sum() > 0:
                precision = (pred_top3_mask & true_top3_mask).sum() / pred_top3_mask.sum()
                top3_precision_scores.append(precision)
            
            if true_top3_mask.sum() > 0:
                recall = (pred_top3_mask & true_top3_mask).sum() / true_top3_mask.sum()
                top3_recall_scores.append(recall)
        
        return {
            'ranking_mae_mean': np.mean(mae_errors) if mae_errors else float('inf'),
            'ranking_mae_std': np.std(mae_errors) if mae_errors else 0.0,
            'top3_precision_mean': np.mean(top3_precision_scores) if top3_precision_scores else 0.0,
            'top3_recall_mean': np.mean(top3_recall_scores) if top3_recall_scores else 0.0,
            'n_valid_races_for_ranking_acc': len(mae_errors),
        }
    
    def _ndcg_metrics(self, scores: np.ndarray, k_values: List[int] = [3, 5, 10]) -> Dict[str, float]:
        """NDCG (Normalized Discounted Cumulative Gain) メトリクス"""
        ndcg_results = {f'ndcg_at_{k}': [] for k in k_values}
        
        for race_idx in range(len(scores)):
            race_mask = self.mask[race_idx]
            if race_mask.sum() < 2:
                continue
                
            race_scores = scores[race_idx][race_mask]
            race_rankings = self.rankings[race_idx][race_mask]
            
            valid_rank_mask = race_rankings > 0
            if valid_rank_mask.sum() < 2:
                continue
            
            final_scores = race_scores[valid_rank_mask]
            final_rankings = race_rankings[valid_rank_mask]
            
            # 関連度スコア（順位が高いほど高スコア）
            max_rank = len(final_rankings)
            relevance_scores = max_rank + 1 - final_rankings  # 1位=max_rank, 最下位=1
            
            for k in k_values:
                if len(final_scores) >= k:
                    ndcg_k = self._calculate_ndcg(final_scores, relevance_scores, k)
                    if not np.isnan(ndcg_k):
                        ndcg_results[f'ndcg_at_{k}'].append(ndcg_k)
        
        # 平均値を計算
        result = {}
        for k in k_values:
            scores_list = ndcg_results[f'ndcg_at_{k}']
            result[f'ndcg_at_{k}_mean'] = np.mean(scores_list) if scores_list else 0.0
            result[f'ndcg_at_{k}_std'] = np.std(scores_list) if scores_list else 0.0
        
        return result
    
    def _calculate_spearman(self, x: np.ndarray, y: np.ndarray) -> float:
        """スピアマン順位相関係数の計算"""
        if len(x) != len(y) or len(x) < 2:
            return np.nan
        
        n = len(x)
        
        # 同順位の処理
        x_ranks = self._assign_ranks(x)
        y_ranks = self._assign_ranks(y)
        
        # 順位差の二乗和
        d_squared = np.sum((x_ranks - y_ranks) ** 2)
        
        # スピアマン相関係数
        rho = 1 - (6 * d_squared) / (n * (n**2 - 1))
        return rho
    
    def _calculate_kendall_tau(self, x: np.ndarray, y: np.ndarray) -> float:
        """ケンドールのタウの計算"""
        if len(x) != len(y) or len(x) < 2:
            return np.nan
        
        n = len(x)
        concordant = 0
        discordant = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                sign_x = np.sign(x[i] - x[j])
                sign_y = np.sign(y[i] - y[j])
                
                if sign_x * sign_y > 0:
                    concordant += 1
                elif sign_x * sign_y < 0:
                    discordant += 1
        
        total_pairs = n * (n - 1) // 2
        if total_pairs == 0:
            return np.nan
        
        tau = (concordant - discordant) / total_pairs
        return tau
    
    def _assign_ranks(self, x: np.ndarray) -> np.ndarray:
        """順位を割り当て（同順位は平均順位）"""
        sorted_indices = np.argsort(x)
        ranks = np.empty(len(x), dtype=float)
        
        i = 0
        while i < len(x):
            j = i
            # 同じ値の範囲を見つける
            while j < len(x) - 1 and x[sorted_indices[j]] == x[sorted_indices[j + 1]]:
                j += 1
            
            # 平均順位を計算
            avg_rank = (i + j + 2) / 2  # 1-indexedなので+1, さらに範囲の平均なので+1
            
            # 同じ値の要素に平均順位を割り当て
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            
            i = j + 1
        
        return ranks
    
    def _calculate_ndcg(self, scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
        """NDCG@kの計算"""
        if len(scores) == 0 or k <= 0:
            return 0.0
        
        # 予測順序でソート
        pred_order = np.argsort(-scores)[:k]
        
        # DCG@k計算
        dcg = 0.0
        for i, idx in enumerate(pred_order):
            rel = relevance[idx]
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1)=0
        
        # IDCG@k計算（理想的な順序）
        ideal_order = np.argsort(-relevance)[:k]
        idcg = 0.0
        for i, idx in enumerate(ideal_order):
            rel = relevance[idx]
            idcg += rel / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
