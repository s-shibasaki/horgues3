import torch
from typing import Optional
from torch import nn

class WeightedPlackettLuceLoss(nn.Module):
    """
    重み付きPlackett-Luce損失 - 1着/2着/3着の各スコアを使い分ける
    
    Args:
        temperature: ソフトマックスの温度パラメータ
        weight_decay: 重みの減衰率（1.0で等重み、<1.0で上位重視）
        reduction: 'mean', 'sum', 'none'
    """
    
    def __init__(self, temperature: float = 1.0, weight_decay: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.weight_decay = weight_decay
        self.reduction = reduction
    
    def forward(self, scores: torch.Tensor, rankings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (batch_size, num_horses, 3) - 各馬の1着/2着/3着スコア
            rankings: (batch_size, num_horses) - 各馬の順位（1着=1, 2着=2, 3着=3, 0=圏外）
            mask: (batch_size, num_horses) - 有効な馬のマスク
        Returns:
            loss: 重み付きPlackett-Luce損失
        """
        batch_size, num_horses, num_positions = scores.shape
        device = scores.device
        
        # 温度でスケール
        scaled_scores = scores / self.temperature
        
        # ランキングマスク（1-3着のみ）
        ranking_mask = (rankings >= 1) & (rankings <= num_positions) & mask
        
        # バッチ内に有効なレースがない場合
        if not ranking_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 有効なバッチのマスク
        valid_batch_mask = ranking_mask.any(dim=1)
        
        # 有効なバッチのみ処理
        valid_scores = scaled_scores[valid_batch_mask]
        valid_rankings = rankings[valid_batch_mask] 
        valid_ranking_mask = ranking_mask[valid_batch_mask]
        
        # 位置ごとの重み（事前計算）
        position_weights = torch.tensor([self.weight_decay ** i for i in range(num_positions)], 
                                      device=device, dtype=scores.dtype)
        
        total_loss = 0.0
        num_valid_samples = 0
        
        # バッチごとに処理（ベクトル化が困難な部分）
        for batch_idx in range(valid_scores.size(0)):
            batch_rankings = valid_rankings[batch_idx]
            batch_mask = valid_ranking_mask[batch_idx]
            
            if not batch_mask.any():
                continue
                
            # 有効な馬のインデックスと順位を取得
            valid_indices = torch.where(batch_mask)[0]
            valid_ranks = batch_rankings[valid_indices]
            
            # 順位でソート
            sort_order = torch.argsort(valid_ranks)
            sorted_indices = valid_indices[sort_order]
            
            # 位置ごとの損失を計算
            batch_loss = 0.0
            max_pos = min(len(sorted_indices), num_positions)
            
            for pos in range(max_pos):
                current_horse = sorted_indices[pos]
                current_score = valid_scores[batch_idx, current_horse, pos]
                
                # 残りの馬のスコア（同じ位置）
                remaining_indices = sorted_indices[pos:]
                remaining_scores = valid_scores[batch_idx, remaining_indices, pos]
                
                # Plackett-Luce確率の対数
                log_prob = current_score - torch.logsumexp(remaining_scores, dim=0)
                
                # 重み付き損失
                batch_loss -= position_weights[pos] * log_prob
            
            total_loss += batch_loss
            num_valid_samples += 1
        
        if num_valid_samples == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # reduction処理
        if self.reduction == 'mean':
            return total_loss / num_valid_samples
        elif self.reduction == 'sum':
            return total_loss
        else:  # 'none'
            # バッチごとの損失を返す場合は元の形状に合わせる
            batch_losses = torch.zeros(batch_size, device=device, dtype=scores.dtype)
            if num_valid_samples > 0:
                batch_losses[valid_batch_mask] = total_loss / num_valid_samples
            return batch_losses
