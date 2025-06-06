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


class ExpectedValueBettingMetric(BaseMetric):
    """
    期待値ベース投資戦略メトリクス
    
    概要:
        期待値が1.0を超える馬券のみを購入する戦略での収益性を評価します。
        期待値 = 予測確率 × オッズ が1.0を超える場合のみ投資し、
        長期的な収益性を測定します。
    
    使い道:
        - 期待値理論に基づく投資戦略の評価
        - 長期的な収益安定性の確認
        - リスク管理された投資手法の検証
        - 予測精度とオッズのバランス評価
        
    出力メトリクス:
        - recovery_rate: 回収率（払戻金/投資金額）
        - profit_rate: 利益率（(払戻金-投資金額)/投資金額）
        - bet_frequency: 投資頻度（期待値>1.0のレース割合）
        - avg_expected_value: 投資時の平均期待値
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None,
                 winning_tickets: Optional[Dict[str, np.ndarray]] = None,
                 odds: Optional[Dict[str, np.ndarray]] = None,
                 bet_amount: float = 100.0, min_expected_value: float = 1.05):
        super().__init__(rankings, mask, winning_tickets, odds)
        self.bet_amount = bet_amount
        self.min_expected_value = min_expected_value
        self.bet_types = ['tansho', 'fukusho', 'umaren', 'wide', 'umatan', 'sanrenfuku', 'sanrentan']
    
    def __call__(self, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if self.winning_tickets is None or self.odds is None:
            return {}
        
        metrics = {'expected_value_betting': {}}
        
        for bet_type in self.bet_types:
            if (bet_type not in probabilities or 
                bet_type not in self.winning_tickets or 
                bet_type not in self.odds):
                continue
            
            pred_probs = probabilities[bet_type]
            winning_tickets = self.winning_tickets[bet_type]
            odds_data = self.odds[bet_type]
            num_races = pred_probs.shape[0]
            
            total_bet = 0.0
            total_payout = 0.0
            bet_count = 0
            total_races = 0
            expected_values = []
            
            for race_idx in range(num_races):
                race_probs = pred_probs[race_idx]
                race_winning = winning_tickets[race_idx]
                race_odds = odds_data[race_idx]
                
                valid_combos = (race_probs > 0) & ~np.isnan(race_odds) & (race_odds > 0)
                if not valid_combos.any():
                    continue
                
                total_races += 1
                
                # 期待値計算
                expected_values_race = race_probs * race_odds
                valid_expected_values = expected_values_race[valid_combos]
                
                # 期待値が閾値を超える組み合わせを探す
                best_combo_idx = None
                best_expected_value = 0
                
                for combo_idx in np.where(valid_combos)[0]:
                    ev = expected_values_race[combo_idx]
                    if ev >= self.min_expected_value and ev > best_expected_value:
                        best_expected_value = ev
                        best_combo_idx = combo_idx
                
                if best_combo_idx is not None:
                    bet_count += 1
                    total_bet += self.bet_amount
                    expected_values.append(best_expected_value)
                    
                    if race_winning[best_combo_idx]:
                        payout = self.bet_amount * race_odds[best_combo_idx]
                        total_payout += payout
            
            bet_metrics = {}
            if total_bet > 0:
                bet_metrics['recovery_rate'] = total_payout / total_bet
                bet_metrics['profit_rate'] = (total_payout - total_bet) / total_bet
                bet_metrics['bet_frequency'] = bet_count / total_races if total_races > 0 else 0
                bet_metrics['avg_expected_value'] = np.mean(expected_values) if expected_values else 0
                
                bet_metrics['summary'] = {
                    'total_bet': total_bet,
                    'total_payout': total_payout,
                    'net_profit': total_payout - total_bet,
                    'bet_count': bet_count,
                    'total_races': total_races,
                    'min_ev_threshold': self.min_expected_value
                }
            
            metrics['expected_value_betting'][bet_type] = bet_metrics
        
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
            
            # 較正エラー
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            calibration_error = 0.0
            total_samples = len(all_probs)
            
            for i in range(self.n_bins):
                bin_lower = bin_boundaries[i]
                bin_upper = bin_boundaries[i + 1]
                
                in_bin = (all_probs > bin_lower) & (all_probs <= bin_upper)
                bin_size = in_bin.sum()
                
                if bin_size > 0:
                    bin_accuracy = all_outcomes[in_bin].mean()
                    bin_confidence = all_probs[in_bin].mean()
                    bin_weight = bin_size / total_samples
                    bin_error = abs(bin_confidence - bin_accuracy)
                    calibration_error += bin_weight * bin_error
            
            metrics['calibration'][bet_type] = {
                'brier_score': brier_score,
                'calibration_error': calibration_error
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