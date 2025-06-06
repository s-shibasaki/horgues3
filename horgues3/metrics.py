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
    馬券種ごとの的中率メトリクス
    
    概要:
        予測モデルが各馬券種で的中を狙える精度を測定するメトリクスです。
        最高確率の組み合わせや上位k件の組み合わせがどの程度的中するかを評価します。
    
    使い道:
        - モデルの予測精度の基本評価
        - 各馬券種における予測の信頼性確認
        - 実際の馬券購入戦略の効果測定
        - 異なるモデル間の性能比較
        
    出力メトリクス:
        - hit_rate: 最高確率の組み合わせの的中率
        - top_k_hit_rate: 上位k件以内の的中率 (k=1,3,5)
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None,
                 winning_tickets: Optional[Dict[str, np.ndarray]] = None,
                 odds: Optional[Dict[str, np.ndarray]] = None,
                 top_k_list: List[int] = [1, 3, 5]):
        super().__init__(rankings, mask, winning_tickets, odds)
        self.top_k_list = top_k_list
        self.bet_types = ['tansho', 'fukusho', 'umaren', 'wide', 'umatan', 'sanrenfuku', 'sanrentan']
    
    def __call__(self, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if self.winning_tickets is None:
            return {}
        
        metrics = {'betting_accuracy': {}}
        
        for bet_type in self.bet_types:
            if bet_type not in probabilities or bet_type not in self.winning_tickets:
                continue
            
            pred_probs = probabilities[bet_type]
            winning_tickets = self.winning_tickets[bet_type]
            num_races = pred_probs.shape[0]
            
            bet_metrics = {}
            
            # 基本的中率計算
            top1_hits = 0
            valid_races = 0
            
            for race_idx in range(num_races):
                race_probs = pred_probs[race_idx]
                race_winning = winning_tickets[race_idx]
                
                valid_combos = race_probs > 0
                if not valid_combos.any():
                    continue
                
                valid_races += 1
                
                # 最高確率の組み合わせが的中しているか
                top_combo_idx = np.argmax(race_probs)
                if race_winning[top_combo_idx]:
                    top1_hits += 1
            
            if valid_races > 0:
                bet_metrics['hit_rate'] = top1_hits / valid_races
            
            # 上位k位以内的中率
            top_k_metrics = {}
            for k in self.top_k_list:
                topk_hits = 0
                valid_races_k = 0
                
                for race_idx in range(num_races):
                    race_probs = pred_probs[race_idx]
                    race_winning = winning_tickets[race_idx]
                    
                    valid_combos = race_probs > 0
                    if not valid_combos.any():
                        continue
                    
                    valid_races_k += 1
                    top_k_indices = np.argsort(race_probs)[::-1][:k]
                    
                    if race_winning[top_k_indices].any():
                        topk_hits += 1
                
                if valid_races_k > 0:
                    top_k_metrics[f'top_{k}'] = topk_hits / valid_races_k
            
            bet_metrics['top_k_hit_rates'] = top_k_metrics
            metrics['betting_accuracy'][bet_type] = bet_metrics
        
        return metrics


class BettingProfitMetric(BaseMetric):
    """
    馬券種ごとの収益性メトリクス
    
    概要:
        実際の馬券購入を想定した収益性を測定するメトリクスです。
        予測に基づいて馬券を購入した場合の回収率、利益率、勝率を計算します。
    
    使い道:
        - 実際の馬券投資での収益性評価
        - リスク管理とポートフォリオ最適化
        - 馬券種別の投資戦略立案
        - 長期的な投資効果の検証
        
    出力メトリクス:
        - recovery_rate: 回収率（払戻金/投資金額）
        - profit_rate: 利益率（(払戻金-投資金額)/投資金額）
        - win_rate: 勝率（利益が出たレースの割合）
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None,
                 winning_tickets: Optional[Dict[str, np.ndarray]] = None,
                 odds: Optional[Dict[str, np.ndarray]] = None,
                 bet_amount: float = 100.0):
        super().__init__(rankings, mask, winning_tickets, odds)
        self.bet_amount = bet_amount
        self.bet_types = ['tansho', 'fukusho', 'umaren', 'wide', 'umatan', 'sanrenfuku', 'sanrentan']
    
    def __call__(self, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if self.winning_tickets is None or self.odds is None:
            return {}
        
        metrics = {'betting_profit': {}}
        
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
            profits = []
            
            for race_idx in range(num_races):
                race_probs = pred_probs[race_idx]
                race_winning = winning_tickets[race_idx]
                race_odds = odds_data[race_idx]
                
                valid_combos = (race_probs > 0) & ~np.isnan(race_odds) & (race_odds > 0)
                if not valid_combos.any():
                    continue
                
                valid_probs = race_probs.copy()
                valid_probs[~valid_combos] = 0
                top_combo_idx = np.argmax(valid_probs)
                
                total_bet += self.bet_amount
                race_profit = -self.bet_amount
                
                if race_winning[top_combo_idx]:
                    payout = self.bet_amount * race_odds[top_combo_idx]
                    total_payout += payout
                    race_profit += payout
                
                profits.append(race_profit)
            
            bet_metrics = {}
            if total_bet > 0:
                bet_metrics['recovery_rate'] = total_payout / total_bet
                bet_metrics['profit_rate'] = (total_payout - total_bet) / total_bet
                
                if profits:
                    bet_metrics['win_rate'] = np.mean(np.array(profits) > 0)
                
                bet_metrics['summary'] = {
                    'total_bet': total_bet,
                    'total_payout': total_payout,
                    'net_profit': total_payout - total_bet,
                    'num_races': len(profits)
                }
            
            metrics['betting_profit'][bet_type] = bet_metrics
        
        return metrics


class BettingCalibrationMetric(BaseMetric):
    """
    予測確率の較正メトリクス
    
    概要:
        予測確率が実際の的中確率とどの程度一致しているかを測定するメトリクスです。
        モデルが出力する確率の信頼性と較正度を評価します。
    
    使い道:
        - 予測確率の信頼性評価
        - モデルの較正性能の診断
        - 確率的予測の品質改善
        - 異なるモデルの確率精度比較
        
    出力メトリクス:
        - brier_score: ブライアスコア（確率予測の二乗誤差）
        - calibration_error: 較正エラー（予測確率と実際の確率の差）
    """
    
    def __init__(self, rankings: np.ndarray, mask: Optional[np.ndarray] = None,
                 winning_tickets: Optional[Dict[str, np.ndarray]] = None,
                 odds: Optional[Dict[str, np.ndarray]] = None,
                 n_bins: int = 10):
        super().__init__(rankings, mask, winning_tickets, odds)
        self.n_bins = n_bins
        self.bet_types = ['tansho', 'fukusho', 'umaren', 'wide', 'umatan']
    
    def __call__(self, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if self.winning_tickets is None:
            return {}
        
        metrics = {'betting_calibration': {}}
        
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
            
            bet_metrics = {}
            
            # ブライアスコア
            brier_score = np.mean((all_probs - all_outcomes) ** 2)
            bet_metrics['brier_score'] = brier_score
            
            # 較正エラー
            bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
            calibration_error = 0.0
            total_samples = len(all_probs)
            bin_details = {}
            
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
                    
                    bin_details[f'bin_{i+1}'] = {
                        'range': f'{bin_lower:.2f}-{bin_upper:.2f}',
                        'confidence': bin_confidence,
                        'accuracy': bin_accuracy,
                        'error': bin_error,
                        'count': bin_size
                    }
            
            bet_metrics['calibration_error'] = calibration_error
            bet_metrics['bin_analysis'] = bin_details
            
            metrics['betting_calibration'][bet_type] = bet_metrics
        
        return metrics


class RankCorrelationMetric(BaseMetric):
    """
    順位相関メトリクス
    
    概要:
        予測した馬の強さ順位と実際の着順の相関を測定するメトリクスです。
        単勝確率に基づく予測順位と実際の着順の一致度を評価します。
    
    使い道:
        - 馬の強さ予測の精度評価
        - 順位予測モデルの性能測定
        - レース予測の基本的な妥当性確認
        - 回帰問題としての着順予測評価
        
    出力メトリクス:
        - spearman_corr: スピアマン順位相関係数（順位の一致度）
    """
    
    def __call__(self, probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        if 'tansho' not in probabilities:
            return {}
        
        scores = probabilities['tansho']
        num_races = scores.shape[0]
        spearman_corrs = []
        race_details = []
        
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
                    race_details.append({
                        'race_idx': race_idx,
                        'correlation': spear_corr,
                        'num_horses': len(final_scores)
                    })
            except Exception:
                continue
        
        metrics = {'rank_correlation': {}}
        if spearman_corrs:
            metrics['rank_correlation'] = {
                'spearman_corr': np.mean(spearman_corrs),
                'summary': {
                    'mean_correlation': np.mean(spearman_corrs),
                    'std_correlation': np.std(spearman_corrs),
                    'min_correlation': np.min(spearman_corrs),
                    'max_correlation': np.max(spearman_corrs),
                    'num_races': len(spearman_corrs)
                }
            }
        
        return metrics