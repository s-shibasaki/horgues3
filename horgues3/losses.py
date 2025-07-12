import torch
from typing import Optional
from torch import nn

class WeightedPlackettLuceLoss(nn.Module):
    """
    重み付きPlackett-Luce損失 - 上位の順位により大きな重みを付ける
    
    Args:
        temperature: ソフトマックスの温度パラメータ
        top_k: 上位k位までの損失を計算
        weight_decay: 重みの減衰率（1.0で等重み、<1.0で上位重視）
        reduction: 'mean', 'sum', 'none'
    """
    
    def __init__(self, temperature: float = 1.0, top_k: Optional[int] = None, 
                 weight_decay: float = 0.8, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.weight_decay = weight_decay
        self.reduction = reduction
    
    def forward(self, scores: torch.Tensor, rankings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_horses = scores.shape
        scaled_scores = scores / self.temperature
        masked_scores = torch.where(mask, scaled_scores, torch.full_like(scaled_scores, -1e9))
        
        losses = []
        
        for batch_idx in range(batch_size):
            batch_scores = masked_scores[batch_idx]
            batch_rankings = rankings[batch_idx]
            batch_mask = mask[batch_idx]
            
            valid_horses = torch.where(batch_mask)[0]
            if len(valid_horses) == 0:
                continue
                
            valid_scores = batch_scores[valid_horses]
            valid_rankings = batch_rankings[valid_horses]
            
            valid_rank_mask = valid_rankings > 0
            if not valid_rank_mask.any():
                continue
                
            final_horses = valid_horses[valid_rank_mask]
            final_scores = valid_scores[valid_rank_mask]
            final_rankings = valid_rankings[valid_rank_mask]
            
            if len(final_horses) <= 1:
                continue
            
            sorted_indices = torch.argsort(final_rankings)
            sorted_scores = final_scores[sorted_indices]
            
            if self.top_k is not None:
                k = min(self.top_k, len(sorted_scores))
                sorted_scores = sorted_scores[:k]
            
            # 重み付きPlackett-Luce損失を計算
            batch_loss = 0.0
            
            for i in range(len(sorted_scores) - 1):
                # i位の重み（1位が最大、下位ほど小さくなる）
                weight = self.weight_decay ** i
                
                log_prob = sorted_scores[i] - torch.logsumexp(sorted_scores[i:], dim=0)
                batch_loss -= weight * log_prob
            
            losses.append(batch_loss)
        
        if not losses:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        loss_tensor = torch.stack(losses)
        
        if self.reduction == 'mean':
            return loss_tensor.mean()
        elif self.reduction == 'sum':
            return loss_tensor.sum()
        else:
            return loss_tensor
