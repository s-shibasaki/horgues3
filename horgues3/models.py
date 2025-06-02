import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
import logging

logger = logging.getLogger(__name__)


class HorseStrengthModel(nn.Module):
    """馬埋め込みとレース埋め込みから馬の強さを予測するモデル"""

    def __init__(self, horse_embed_dim, race_embed_dim, hidden_dim=128, dropout_rate=0.1):
        super(HorseStrengthModel, self).__init__()
        
        self.horse_embed_dim = horse_embed_dim
        self.race_embed_dim = race_embed_dim
        self.hidden_dim = hidden_dim

        # 馬の埋め込みベクトルを処理する層
        self.horse_fc = nn.Linear(horse_embed_dim, hidden_dim)

        # レースの埋め込みベクトルを処理する層
        self.race_fc = nn.Linear(race_embed_dim, hidden_dim)

        # 結合後の隠れ層
        self.hidden_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)  # 強さスコア (スカラ値)
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, horse_embeds, race_embed, mask=None):
        """
        Args:
            horse_embeds: 馬の埋め込みベクトル (batch_size, num_horses, horse_embed_dim)
            race_embed: レースの埋め込みベクトル (batch_size, race_embed_dim)
            mask: 有効な馬のマスク (batch_size, num_horses)、1=有効、0=無効/パディング

        Returns:
            strength_scores: 馬の強さスコア (batch_size, num_horses)
        """
        batch_size, num_horses, _ = horse_embeds.shape

        # 馬の埋め込みベクトルを処理
        horse_features = F.relu(self.horse_fc(horse_embeds))  # (batch_size, num_horses, hidden_dim)
        horse_features = self.dropout(horse_features)

        # レース条件の埋め込みベクトルを処理
        race_features = F.relu(self.race_fc(race_embed))  # (batch_size, hidden_dim)
        race_features = self.dropout(race_features)

        # レース条件を各馬に対して複製
        race_features_expanded = race_features.unsqueeze(1).expand(-1, num_horses, -1)  # (batch_size, num_horses, hidden_dim)

        # 馬の特徴量とレース条件を結合
        combined_features = torch.cat([horse_features, race_features_expanded], dim=-1)  # (batch_size, num_horses, hidden_dim * 2)

        # 強さスコアを計算
        strength_scores = self.hidden_layers(combined_features).squeeze(-1)  # (batch_size, num_horses)

        # マスクが指定されている場合、無効な馬のスコアを非常に小さい値に設定
        if mask is not None:
            strength_scores = strength_scores.masked_fill(mask == 0, -1e9)

        return strength_scores  # (batch_size, num_horses)

class PlackettLuceLoss(nn.Module):
    """
    Plackett-Luce損失関数
    """

    def __init__(self, temperature=1.0):
        super(PlackettLuceLoss, self).__init__()
        self.temperature = temperature

    def forward(self, scores, rankings, mask=None):
        """
        Args:
            scores: (batch_size, num_horses) - 各馬の強さスコア 
            rankings: (batch_size, num_horses) - 実際の着順（1位=0, 2位=1, ...）
            mask: (batch_size, num_horses) - 有効な馬のマスク (1=有効, 0=無効/パディング)
        """
        batch_size, num_horses = scores.shape

        # マスクが指定されていない場合はすべて有効とみなす
        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        else:
            mask = mask.bool()

        # スコアをtemperatureでスケール
        scaled_scores = scores / self.temperature

        # 各レースの損失を計算
        total_loss = 0
        valid_races = 0

        for b in range(batch_size):
            race_scores = scaled_scores[b]
            race_rankings = rankings[b]
            race_mask = mask[b]
            
            # 有効な馬のみを考慮
            valid_indices = torch.where(race_mask)[0]
            
            # 有効な馬が1頭以下の場合はスキップ
            if len(valid_indices) <= 1:
                continue
                
            valid_races += 1
            
            # 着順でソート（有効な馬のみ）
            valid_rankings = race_rankings[valid_indices]
            sorted_valid_indices = valid_indices[torch.argsort(valid_rankings)]

            race_loss = 0
            remaining_horses = valid_indices.clone()

            # 各着順について確率を計算
            for position in range(len(valid_indices) - 1):  # 最後の馬は確率1なので除外
                current_horse = sorted_valid_indices[position]

                # 残っている馬の中でのsoftmax確率を計算
                remaining_scores = race_scores[remaining_horses]
                log_prob = F.log_softmax(remaining_scores, dim=0)

                # 現在の馬のインデックスを残っている馬の中で見つける
                horse_idx_in_remaining = (remaining_horses == current_horse).nonzero(as_tuple=True)[0]

                # クロスエントロピー損失を加算
                race_loss -= log_prob[horse_idx_in_remaining]

                # この馬を残りの候補から除外
                mask = remaining_horses != current_horse
                remaining_horses = remaining_horses[mask]

            total_loss += race_loss

        # 有効なレースがない場合は0を返す
        if valid_races == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
            
        return total_loss / valid_races


