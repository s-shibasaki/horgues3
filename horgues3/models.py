import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
import logging
from typing import List, Optional, Union, Dict
import numpy as np

logger = logging.getLogger(__name__)


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
                mask_remaining = remaining_horses != current_horse
                remaining_horses = remaining_horses[mask_remaining]

            total_loss += race_loss

        # 有効なレースがない場合は0を返す
        if valid_races == 0:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
            
        return total_loss / valid_races


class FeatureTokenizer(nn.Module):
    """数値特徴量とカテゴリ特徴量をトークン化"""

    def __init__(self, numerical_features: int, categorical_features: List[int], d_token: int = 192):
        super().__init__()
        self.d_token = d_token
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        # 数値特徴量用の線形変換
        if numerical_features > 0:
            self.numerical_tokenizer = nn.Linear(1, d_token)

        # カテゴリ特徴量用の埋め込み
        self.categorical_tokenizers = nn.ModuleList([
            nn.Embedding(vocab_size, d_token) for vocab_size in categorical_features
        ])

        # [CLS]トークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

    def forward(self, x_num: Optional[torch.Tensor] = None, x_cat: Optional[torch.Tensor] = None):
        batch_size = x_num.size(0) if x_num is not None else x_cat.size(0)

        tokens = []

        # [CLS]トークンを追加
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens.append(cls_tokens)

        # 数値特徴量のトークン化
        if x_num is not None:
            for i in range(self.numerical_features):
                token = self.numerical_tokenizer(x_num[:, i:i+1].unsqueeze(-1))  # (batch_size, 1, d_token)
                tokens.append(token)

        # カテゴリ特徴量のトークン化
        if x_cat is not None:
            for i, tokenizer in enumerate(self.categorical_tokenizers):
                token = tokenizer(x_cat[:, i]).unsqueeze(1)  # (batch_size, 1, d_token)
                tokens.append(token)

        return torch.cat(tokens, dim=1)  # (batch_size, num_tokens, d_token)
    
class MultiHeadAttention(nn.Module):
    """マルチヘッドアテンション"""

    def __init__(self, d_token: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_token % n_heads == 0

        self.d_token = d_token
        self.n_heads = n_heads
        self.d_head = d_token // n_heads

        self.q_linear = nn.Linear(d_token, d_token)
        self.k_linear = nn.Linear(d_token, d_token)
        self.v_linear = nn.Linear(d_token, d_token)
        self.out_linear = nn.Linear(d_token, d_token)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape

        # Q, K, Vの計算
        Q = self.q_linear(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # アテンションスコアを計算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # アテンションを適用
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_token)

        return self.out_linear(context)  # (batch_size, seq_len, d_token)
    

class FeedForward(nn.Module):
    """フィードフォワードネットワーク"""

    def __init__(self, d_token: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_token, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))  # (batch_size, seq_len, d_token)
    

class TransformerBlock(nn.Module):
    """Transformerブロック"""

    def __init__(self, d_token: int, n_heads: int = 8, d_ffn: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ffn is None:
            d_ffn = d_token * 4

        self.attention = MultiHeadAttention(d_token, n_heads, dropout)
        self.feed_forward = FeedForward(d_token, d_ffn, dropout)

        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # アテンション + 残差接続
        attention_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attention_output))

        # フィードフォワード + 残差接続
        ffn_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x  # (batch_size, seq_len, d_token)
    

class FTTransformer(nn.Module):
    """FT Transformer メインモデル"""

    def __init__(self, d_token: int = 192, n_layers: int = 3, n_heads: int = 8, d_ffn: int = None, dropout: float = 0.1):
        super().__init__()

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor):
        # Transformerブロックを通す
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # [CLS]トークンを使用して分類
        cls_token = tokens[:, 0]
        cls_token = self.norm(cls_token)
        cls_token = self.dropout(cls_token)

        return cls_token  # (batch_size, d_token)


class RaceInteractionLayer(nn.Module):
    """レース内の馬の相互作用を考慮したスコア計算層"""

    def __init__(self, d_token: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token
        self.n_heads = n_heads

        # 馬同士の相互作用を計算するためのアテンション
        self.interaction_attention = MultiHeadAttention(d_token, n_heads, dropout)

        # 相互作用後の特徴量を統合
        self.interaction_norm = nn.LayerNorm(d_token)
        self.interaction_dropout = nn.Dropout(dropout)

        # 最終的な強さスコアを計算
        self.score_head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, 1)
        )

    def forward(self, horse_features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            horse_features: (batch_size, num_horses, d_token) - 各馬の特徴量
            mask: (batch_size, num_horses) - 有効な馬のマスク (1=有効, 0=無効/パディング)

        Returns:
            scores: (batch_size, num_horses) - 相互作用を考慮した馬の強さスコア
        """
        batch_size, num_horses, _ = horse_features.shape

        # NaNをゼロで置換（マスクされた馬の特徴量）
        horse_features = torch.where(torch.isnan(horse_features), torch.zeros_like(horse_features), horse_features)

        # アテンションマスクの準備 (パディングされた馬を除外)
        if mask is not None:
            # (batch_size, num_horses) -> (batch_size, 1, 1, num_horses)
            attention_mask = mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(batch_size, self.n_heads, num_horses, num_horses)
        else:
            attention_mask = None

        # 馬同士の相互作用を計算
        interaction_features = self.interaction_attention(horse_features, attention_mask)

        # 残差接続と正規化
        combined_features = self.interaction_norm(horse_features + self.interaction_dropout(interaction_features))

        # 各馬の強さスコアを計算
        scores = self.score_head(combined_features).squeeze(-1)  # (batch_size, num_horses)

        # マスクされた馬のスコアを-infに設定 (softmax時に0になる)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))

        return scores


class Horgues3Model(nn.Module):
    """統合された競馬予測モデル"""

    def __init__(self, d_token=192, n_layers=3, n_heads=8, d_ffn=None, dropout=0.1, max_horses=18):
        super().__init__()
        self.d_token = d_token
        self.max_horses = max_horses

        # 単純な特徴量用のトークナイザー
        self.tokenizer = FeatureTokenizer(
            numerical_features=1,  # 馬体重
            categorical_features=[],  # カテゴリ特徴量はなし
            d_token=d_token
        )

        # FT Transformer
        self.ft_transformer = FTTransformer(
            d_token=d_token,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ffn=d_ffn,
            dropout=dropout
        )

        # レース内相互作用層
        self.race_interaction = RaceInteractionLayer(
            d_token=d_token,
            n_heads=n_heads,
            dropout=dropout
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x_num: (batch_size, num_horses, num_numericals) - 馬体重などの数値特徴量
            x_cat: (batch_size, num_horses, num_categoricals) - カテゴリ特徴量
            mask: (batch_size, num_horses) - 有効な馬のマスク (1=有効, 0=無効/パディング)
        """
        batch_size, max_horses, _ = x_num.shape

        # 各馬の特徴量を個別にトークン化
        horse_features = []

        for i in range(max_horses):
            # i番目の馬の特徴量
            tokens = self.tokenizer(x_num[:, i], x_cat[:, i])

            # 時系列データによるトークンの追加はここで行う
            # tokens += additional_tokens

            # FT Transformerで特徴抽出
            features = self.ft_transformer(tokens)  # (batch_size, d_token)
            horse_features.append(features)

        # 全ての馬の特徴量を結合
        horse_features = torch.stack(horse_features, dim=1)  # (batch_size, max_horses, d_token)

        # レース内相互作用を考慮したスコア計算
        scores = self.race_interaction(horse_features, mask)

        return scores
