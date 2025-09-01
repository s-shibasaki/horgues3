import torch
import torch.nn as nn
import torch.nn.functional as F


class HorguesLoss(nn.Module):
    """
    scoresとlog(targets + 0.1)の二乗誤差を計算する損失関数
    targetがnanの部分はマスクして除外する
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: (batch_size, num_horses) - 馬の強さスコア
            targets: (batch_size, num_horses) - ターゲット値（nanを含む可能性あり）
        
        Returns:
            loss: スカラー損失値
        """
        # nanでない要素のマスクを作成
        valid_mask = ~torch.isnan(targets)
        
        # 有効な要素が存在しない場合は0を返す
        if not valid_mask.any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        # 有効な要素のみで計算
        valid_scores = scores[valid_mask]
        valid_targets = targets[valid_mask]
        
        # BCEWithLogitsLoss計算
        losses = self.bce_loss(valid_scores, valid_targets)  # (N,)
        
        # reduction適用
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:  # 'none'
            # 元の形状に戻すため、nanで埋めた結果を返す
            result = torch.full_like(targets, float('nan'))
            result[valid_mask] = losses
            return result