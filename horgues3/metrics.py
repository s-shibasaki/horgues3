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
        winning_tickets: 的中馬券データ (各馬券種のboolean配列)
        odds: オッズデータ (各馬券種のオッズ配列)
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None, 
                 winning_tickets: Optional[Dict[str, np.ndarray]] = None, 
                 odds: Optional[Dict[str, np.ndarray]] = None):
        self.rankings = rankings
        if mask is None:
            self.mask = (rankings > 0).astype(bool)
        else:
            self.mask = mask.astype(bool)
        self.winning_tickets = winning_tickets
        self.odds = odds
    
    def __call__(self, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        メトリクスを計算する
        
        Args:
            probabilities: 予測確率辞書 (各馬券種の確率配列)
            
        Returns:
            計算されたメトリクス値の階層化された辞書
        """
        raise NotImplementedError


class BettingAccuracyMetric(BaseMetric):
    """
    馬券的中精度メトリクス
    
    概要:
        予測モデルの基本的な的中能力を測定します。
        最高確率の組み合わせや上位候補の的中率を評価します。
    
    出力メトリクス:
        - hit_rate: 最高確率組み合わせの的中率
        - top_3_hit_rate: 上位3位以内の的中率
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None,
                 winning_tickets: Optional[Dict[str, np.ndarray]] = None,
                 odds: Optional[Dict[str, np.ndarray]] = None):
        super().__init__(rankings, mask, winning_tickets, odds)
        self.bet_types = ['tansho', 'fukusho', 'umaren', 'wide', 'umatan', 'sanrenfuku', 'sanrentan']
    
    def __call__(self, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if self.winning_tickets is None:
            return {}
        
        metrics = {'accuracy': {}}
        
        for bet_type in self.bet_types:
            if bet_type not in probabilities or bet_type not in self.winning_tickets:
                continue
            
            pred_probs = probabilities[bet_type]
            winning_tickets = self.winning_tickets[bet_type]
            num_races = pred_probs.shape[0]
            
            top1_hits = 0
            top3_hits = 0
            valid_races = 0
            
            for race_idx in range(num_races):
                race_probs = pred_probs[race_idx]
                race_winning = winning_tickets[race_idx]
                
                valid_combos = race_probs > 0
                if not valid_combos.any():
                    continue
                
                valid_races += 1
                
                # 最高確率の組み合わせ
                top_combo_idx = np.argmax(race_probs)
                if race_winning[top_combo_idx]:
                    top1_hits += 1
                
                # 上位3位以内
                top_3_indices = np.argsort(race_probs)[::-1][:3]
                if race_winning[top_3_indices].any():
                    top3_hits += 1
            
            if valid_races > 0:
                metrics['accuracy'][bet_type] = {
                    'hit_rate': top1_hits / valid_races,
                    'top_3_hit_rate': top3_hits / valid_races,
                    'num_races': valid_races
                }
        
        return metrics


class BettingCalibrationMetric(BaseMetric):
    """
    予測確率較正メトリクス
    
    概要:
        モデルが出力する確率が実際の的中確率と一致しているかを評価します。
        確率予測の信頼性を測定する重要なメトリクスです。
    
    出力メトリクス:
        - brier_score: ブライアスコア（確率予測の精度）
        - calibration_error: 較正エラー（予測確率と実確率の差）
        - bin_details: 各ビンの詳細情報（confidence, accuracy, error, count）
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None,
                 winning_tickets: Optional[Dict[str, np.ndarray]] = None,
                 odds: Optional[Dict[str, np.ndarray]] = None,
                 n_bins: int = 10):
        super().__init__(rankings, mask, winning_tickets, odds)
        self.n_bins = n_bins
        self.bet_types = ['tansho', 'fukusho', 'umaren', 'wide', 'umatan', 'sanrenfuku', 'sanrentan']
    
    def __call__(self, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if self.winning_tickets is None:
            return {}
        
        metrics = {'calibration': {}}
        
        for bet_type in self.bet_types:
            if bet_type not in probabilities or bet_type not in self.winning_tickets:
                continue
            
            pred_probs = probabilities[bet_type]
            winning_tickets = self.winning_tickets[bet_type]
            
            all_probs = []
            all_outcomes = []
            
            num_races = pred_probs.shape[0]
            for race_idx in range(num_races):
                race_probs = pred_probs[race_idx]
                race_winning = winning_tickets[race_idx]
                
                valid_mask = race_probs > 0
                all_probs.extend(race_probs[valid_mask])
                all_outcomes.extend(race_winning[valid_mask].astype(float))
            
            if len(all_probs) == 0:
                continue
            
            all_probs = np.array(all_probs)
            all_outcomes = np.array(all_outcomes)
            
            # ブライアスコア
            brier_score = np.mean((all_probs - all_outcomes) ** 2)
            
            # パーセンタイルベースでビンを作成（各ビンの件数を等しくする）
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            bin_boundaries = np.percentile(all_probs, percentiles)
            
            calibration_error = 0.0
            total_samples = len(all_probs)
            bin_details = []
            
            for i in range(self.n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                # 最初のビン以外は左端を除外、最後のビンは右端を含む
                if not i == self.n_bins - 1:
                    in_bin = (all_probs >= bin_lower) & (all_probs < bin_upper)
                else:
                    in_bin = (all_probs >= bin_lower) & (all_probs <= bin_upper)
                
                bin_size = in_bin.sum()
                
                if bin_size > 0:
                    bin_accuracy = all_outcomes[in_bin].mean()
                    bin_confidence = all_probs[in_bin].mean()
                    bin_weight = bin_size / total_samples
                    bin_error = abs(bin_confidence - bin_accuracy)
                    calibration_error += bin_weight * bin_error
                    
                    bin_details.append({
                        'bin_lower': float(bin_lower),
                        'bin_upper': float(bin_upper),
                        'confidence': float(bin_confidence),
                        'accuracy': float(bin_accuracy),
                        'error': float(bin_error),
                        'count': int(bin_size),
                        'weight': float(bin_weight)
                    })
            
            metrics['calibration'][bet_type] = {
                'brier_score': brier_score,
                'calibration_error': calibration_error,
                'bin_details': bin_details,
                'total_samples': total_samples
            }
        
        return metrics


class RankCorrelationMetric(BaseMetric):
    """
    順位相関メトリクス
    
    概要:
        単勝確率に基づく予測順位と実際の着順の相関を測定します。
        馬の強さ予測の基本的な妥当性を評価します。
    
    出力メトリクス:
        - spearman_corr: スピアマン順位相関係数
    """
    
    def __call__(self, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if 'tansho' not in probabilities:
            return {}
        
        scores = probabilities['tansho']
        num_races = scores.shape[0]
        spearman_corrs = []
        
        for race_idx in range(num_races):
            race_scores = scores[race_idx]
            race_rankings = self.rankings[race_idx]
            race_mask = self.mask[race_idx]
            
            valid_horses = np.where(race_mask)[0]
            if len(valid_horses) < 3:
                continue
            
            valid_scores = race_scores[valid_horses]
            valid_rankings = race_rankings[valid_horses]
            
            valid_rank_mask = valid_rankings > 0
            if valid_rank_mask.sum() < 3:
                continue
            
            final_scores = valid_scores[valid_rank_mask]
            final_rankings = valid_rankings[valid_rank_mask]
            
            pred_ranks = len(final_scores) + 1 - np.argsort(np.argsort(final_scores)) - 1
            
            try:
                spear_corr, _ = spearmanr(pred_ranks, final_rankings)
                if not np.isnan(spear_corr):
                    spearman_corrs.append(spear_corr)
            except Exception:
                continue
        
        metrics = {}
        if spearman_corrs:
            metrics['rank_correlation'] = {
                'spearman_corr': np.mean(spearman_corrs),
                'num_races': len(spearman_corrs)
            }
        
        return metrics


class KellyCriterionBettingMetric(BaseMetric):
    """
    ケリー基準ベース投資戦略メトリクス
    
    概要:
        ケリー基準を使用して最適投資額を決定し、収益性を評価します。
        
    出力メトリクス:
        - recovery_rate: 回収率
        - profit_rate: 利益率
        - bet_frequency: 投資頻度
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None,
                 winning_tickets: Optional[Dict[str, np.ndarray]] = None,
                 odds: Optional[Dict[str, np.ndarray]] = None,
                 bankroll: float = 10000.0, max_bet_fraction: float = 0.25,
                 min_kelly_fraction: float = 0.01):
        super().__init__(rankings, mask, winning_tickets, odds)
        self.bankroll = bankroll
        self.max_bet_fraction = max_bet_fraction
        self.min_kelly_fraction = min_kelly_fraction
        self.bet_types = ['tansho', 'fukusho', 'umaren', 'wide', 'umatan', 'sanrenfuku', 'sanrentan']
    
    def _calculate_kelly_fractions(self, probs: np.ndarray, odds: np.ndarray) -> np.ndarray:
        """ケリー基準による投資比率をベクトル計算"""
        # 期待値チェック（prob * odds > 1）
        expected_values = probs * odds
        profitable = expected_values > 1.0
        
        # ケリー基準: f* = (prob * odds - 1) / (odds - 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            kelly_fractions = (expected_values - 1) / (odds - 1)
        
        # 条件に合わないものを0にセット
        kelly_fractions = np.where(
            profitable & (kelly_fractions >= self.min_kelly_fraction) & 
            np.isfinite(kelly_fractions),
            np.minimum(kelly_fractions, self.max_bet_fraction),
            0.0
        )
        
        return kelly_fractions
    
    def _evaluate_betting_strategy(self, probs: np.ndarray, winning_tickets: np.ndarray, 
                                  odds: np.ndarray, threshold: float = 0.0) -> Dict[str, float]:
        """投資戦略を評価"""
        # 有効なデータのマスク
        valid_mask = (probs > threshold) & np.isfinite(odds) & (odds > 1.0)
        
        if not valid_mask.any():
            return None
        
        # ケリー基準による投資比率計算
        kelly_fractions = self._calculate_kelly_fractions(probs, odds)
        
        # 投資対象のマスク（閾値とケリー基準を満たす）
        bet_mask = valid_mask & (kelly_fractions > 0)
        
        if not bet_mask.any():
            return None
        
        # 投資額計算
        bet_amounts = kelly_fractions * self.bankroll
        total_bet = np.sum(bet_amounts[bet_mask])

        # 払戻計算
        payouts = bet_amounts * odds * winning_tickets
        total_payout = np.sum(payouts[bet_mask])

        # 利益
        total_profit = total_payout - total_bet

        # 馬券単位の的中率
        bet_count = np.sum(bet_mask)
        hit_count = np.sum(winning_tickets[bet_mask])
        hit_rate = hit_count / bet_count if bet_count > 0 else 0.0

        # レース単位の的中率
        race_bet_count = np.sum(np.any(bet_mask.reshape(-1, probs.shape[1]), axis=1))
        race_hit_count = np.sum(winning_tickets[bet_mask.reshape(-1, probs.shape[1])])
        race_hit_rate = race_hit_count / race_bet_count if race_bet_count > 0 else 0.0

        # 回収率とスコア
        recovery_rate = total_payout / total_bet if total_bet > 0 else 0.0
        score = race_hit_rate * total_profit
        
        return {
            'total_bet': total_bet,
            'total_payout': total_payout,
            'total_profit': total_profit,
            'hit_rate': hit_rate,
            'race_hit_rate': race_hit_rate,
            'recovery_rate': recovery_rate,
            'score': score,
        }
    
    def __call__(self, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if self.winning_tickets is None or self.odds is None:
            return {}
        
        metrics = {'kelly_betting': {}}
        
        for bet_type in self.bet_types:
            if not all(key in data for key in [bet_type] 
                      for data in [probabilities, self.winning_tickets, self.odds]):
                continue
            
            result = self._evaluate_betting_strategy(
                probabilities[bet_type],
                self.winning_tickets[bet_type],
                self.odds[bet_type]
            )

            if result is not None:
                metrics['kelly_betting'][bet_type] = result
        
        return metrics


class AdaptiveKellyCriterionBettingMetric(KellyCriterionBettingMetric):
    """
    適応的ケリー基準ベース投資戦略メトリクス
    
    概要:
        確率閾値を適応的に調整してリスクを管理する投資戦略を評価します。
        
    出力メトリクス:
        - 基本ケリー基準メトリクス + optimal_threshold
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None,
                 winning_tickets: Optional[Dict[str, np.ndarray]] = None,
                 odds: Optional[Dict[str, np.ndarray]] = None,
                 bankroll: float = 10000.0, max_bet_fraction: float = 0.25,
                 min_kelly_fraction: float = 0.01, threshold_points: int = 10):
        super().__init__(rankings, mask, winning_tickets, odds, bankroll, 
                        max_bet_fraction, min_kelly_fraction)
        self.threshold_points = threshold_points
        
        # 馬券種別ごとの閾値範囲
        self.threshold_ranges = {
            'tansho': (0.05, 0.3), 'fukusho': (0.1, 0.5), 'umaren': (0.02, 0.15),
            'wide': (0.05, 0.25), 'umatan': (0.01, 0.08), 'sanrenfuku': (0.005, 0.05),
            'sanrentan': (0.002, 0.02)
        }
    
    def _find_optimal_threshold(self, probs: np.ndarray, winning_tickets: np.ndarray, 
                               odds: np.ndarray, bet_type: str) -> Tuple[float, Dict[str, Any]]:
        """最適閾値を探索"""
        if bet_type not in self.threshold_ranges or len(probs) < 20:
            return 0.0, {}
        
        min_thresh, max_thresh = self.threshold_ranges[bet_type]
        thresholds = np.linspace(min_thresh, max_thresh, self.threshold_points)
        
        # 各閾値で評価
        best_threshold = min_thresh
        best_score = -float('inf')
        results = []
        
        for threshold in thresholds:
            result = self._evaluate_betting_strategy(probs, winning_tickets, odds, threshold)
            
            if result is None:
                continue

            results.append({'threshold': threshold, **result})
            
            if result['score'] > best_score:
                best_score = result['score']
                best_threshold = threshold
        
        return best_threshold, best_score
    
    def __call__(self, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if self.winning_tickets is None or self.odds is None:
            return {}
        
        metrics = {'adaptive_kelly_betting': {}}
        
        for bet_type in self.bet_types:
            if not all(key in data for key in [bet_type] 
                      for data in [probabilities, self.winning_tickets, self.odds]):
                continue
            
            probs = probabilities[bet_type]
            winning = self.winning_tickets[bet_type]
            odds_data = self.odds[bet_type]

            # データ分割（前半で調整、後半で検証）
            split_idx = len(probs) // 2
            train_data = (probs[:split_idx], winning[:split_idx], odds_data[:split_idx])
            valid_data = (probs[split_idx:], winning[split_idx:], odds_data[split_idx:])
            
            # 最適閾値探索
            optimal_threshold, best_score = self._find_optimal_threshold(
                *train_data, bet_type
            )
            
            # 検証データで最終評価
            validation_result = self._evaluate_betting_strategy(
                *valid_data, optimal_threshold
            )

            if validation_result is not None:
                metrics['adaptive_kelly_betting'][bet_type] = {
                    'optimal_threshold': optimal_threshold,
                    'best_score': best_score,
                    **validation_result
                }
        
        return metrics


class RankingPercentileMetric(BaseMetric):
    """
    正解組合せの予測確率順位パーセンタイルメトリクス
    
    概要:
        正解の組合せが予測確率でソートした際に上位何パーセントに位置するかを測定します。
        モデルの識別能力と予測精度を評価する重要な指標です。
    
    出力メトリクス:
        - mean_percentile: 正解組合せの平均順位パーセンタイル
        - median_percentile: 正解組合せの中央値順位パーセンタイル
        - top_10_percent_rate: 上位10%以内に入った割合
        - top_25_percent_rate: 上位25%以内に入った割合
        - bottom_25_percent_rate: 下位25%以内に入った割合（悪い予測の割合）
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None,
                 winning_tickets: Optional[Dict[str, np.ndarray]] = None,
                 odds: Optional[Dict[str, np.ndarray]] = None):
        super().__init__(rankings, mask, winning_tickets, odds)
        self.bet_types = ['tansho', 'fukusho', 'umaren', 'wide', 'umatan', 'sanrenfuku', 'sanrentan']
    
    def __call__(self, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if self.winning_tickets is None:
            return {}
        
        metrics = {'ranking_percentile': {}}
        
        for bet_type in self.bet_types:
            if bet_type not in probabilities or bet_type not in self.winning_tickets:
                continue
            
            pred_probs = probabilities[bet_type]
            winning_tickets = self.winning_tickets[bet_type]
            num_races = pred_probs.shape[0]
            
            percentiles = []
            
            for race_idx in range(num_races):
                race_probs = pred_probs[race_idx]
                race_winning = winning_tickets[race_idx]
                
                # 有効な組合せのマスク
                valid_mask = race_probs > 0
                if not valid_mask.any():
                    continue
                
                # 正解組合せの確認
                winning_indices = np.where(race_winning & valid_mask)[0]
                if len(winning_indices) == 0:
                    continue
                
                # 確率で降順ソート（高い確率が上位）
                valid_probs = race_probs[valid_mask]
                sorted_indices = np.argsort(valid_probs)[::-1]
                total_combinations = len(valid_probs)
                
                # 各正解組合せの順位パーセンタイルを計算
                for winning_idx in winning_indices:
                    # 元のインデックスから有効インデックス内での位置を特定
                    valid_idx_map = np.where(valid_mask)[0]
                    winning_pos_in_valid = np.where(valid_idx_map == winning_idx)[0]
                    
                    if len(winning_pos_in_valid) > 0:
                        winning_pos = winning_pos_in_valid[0]
                        rank = np.where(sorted_indices == winning_pos)[0][0] + 1
                        percentile = (rank / total_combinations) * 100
                        percentiles.append(percentile)
            
            if percentiles:
                percentiles = np.array(percentiles)
                
                metrics['ranking_percentile'][bet_type] = {
                    'mean_percentile': float(np.mean(percentiles)),
                    'median_percentile': float(np.median(percentiles)),
                    'std_percentile': float(np.std(percentiles)),
                    'top_10_percent_rate': float(np.mean(percentiles <= 10)),
                    'top_25_percent_rate': float(np.mean(percentiles <= 25)),
                    'top_50_percent_rate': float(np.mean(percentiles <= 50)),
                    'bottom_25_percent_rate': float(np.mean(percentiles >= 75)),
                    'num_samples': len(percentiles)
                }
        
        return metrics

