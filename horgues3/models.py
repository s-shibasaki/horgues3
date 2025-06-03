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
    ランキング予測のための損失関数で、各馬の強さスコアから着順確率を計算し、
    実際の着順との交差エントロピーを最小化する。
    """

    def __init__(self, temperature=1.0, eps=1e-8):
        super(PlackettLuceLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps  # 数値安定性のための微小値

    def forward(self, scores, rankings, mask=None):
        """
        Args:
            scores: (batch_size, num_horses) - 各馬の強さスコア 
            rankings: (batch_size, num_horses) - 実際の着順（1位=0, 2位=1, ...、無効馬=-1）
            mask: (batch_size, num_horses) - 有効な馬のマスク (True=有効, False=無効/パディング)
        
        Returns:
            loss: スカラーテンソル
        """
        batch_size, num_horses = scores.shape
        device = scores.device
        
        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        
        total_loss = 0.0
        valid_races = 0
        
        # バッチ内の各レースについて処理
        for b in range(batch_size):
            race_scores = scores[b]  # (num_horses,)
            race_rankings = rankings[b]  # (num_horses,)
            race_mask = mask[b]  # (num_horses,)
            
            # 有効な馬のインデックスを取得
            valid_horses = torch.where(race_mask & (race_rankings >= 0))[0]
            
            if len(valid_horses) < 2:
                # 有効な馬が2頭未満の場合はスキップ
                continue
            
            # 有効な馬のスコアと着順を取得
            valid_scores = race_scores[valid_horses]  # (valid_horses,)
            valid_rankings = race_rankings[valid_horses]  # (valid_horses,)
            
            # スコアの範囲をクリップして数値安定性を向上
            valid_scores = torch.clamp(valid_scores, min=-10.0, max=10.0)
            
            # 着順でソート（1位から順番に）
            sorted_indices = torch.argsort(valid_rankings)
            sorted_scores = valid_scores[sorted_indices]  # 着順順にソートされたスコア
            
            race_loss = 0.0
            
            # Plackett-Luce確率の計算
            # 各位置での選択確率を順次計算
            remaining_scores = sorted_scores.clone()
            
            for pos in range(len(sorted_scores) - 1):  # 最後の1頭は確率1なので除外
                # 温度パラメータでスケール
                scaled_scores = remaining_scores / self.temperature
                
                # 数値安定性のために最大値を引く
                max_score = torch.max(scaled_scores)
                scaled_scores = scaled_scores - max_score
                
                # より安定したsoftmax計算
                exp_scores = torch.exp(torch.clamp(scaled_scores, min=-20.0, max=20.0))
                sum_exp = torch.sum(exp_scores)
                
                # ゼロ除算を避ける
                if sum_exp < self.eps:
                    sum_exp = self.eps
                
                selected_prob = exp_scores[0] / sum_exp
                
                # 確率のクリップ
                selected_prob = torch.clamp(selected_prob, min=self.eps, max=1.0 - self.eps)
                
                race_loss += -torch.log(selected_prob)
                
                # 選ばれた馬を除外して次の位置へ
                remaining_scores = remaining_scores[1:]
            
            total_loss += race_loss
            valid_races += 1
        
        if valid_races == 0:
            # 有効なレースがない場合は0を返す
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 平均損失を返す
        return total_loss / valid_races

        
class FeatureTokenizer(nn.Module):
    """数値特徴量とカテゴリ特徴量をトークン化"""

    def __init__(self, numerical_features: List[str], categorical_features: Dict[str, int], d_token: int = 192):
        super().__init__()
        self.d_token = d_token
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        # 数値特徴量用の線形変換（特徴量ごとに個別）
        self.numerical_tokenizers = nn.ModuleDict({
            name: nn.Linear(1, d_token) for name in numerical_features
        })

        # カテゴリ特徴量用の埋め込み
        self.categorical_tokenizers = nn.ModuleDict({
            name: nn.Embedding(vocab_size, d_token, padding_idx=0) 
            for name, vocab_size in categorical_features.items()
        })

        # [CLS]トークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

    def forward(self, x_num: Optional[Dict[str, torch.Tensor]] = None, 
                x_cat: Optional[Dict[str, torch.Tensor]] = None):
        # 入力の形状を取得（バッチサイズと追加次元）
        if x_num is not None:
            sample_tensor = next(iter(x_num.values()))
        else:
            sample_tensor = next(iter(x_cat.values()))
        
        shape = sample_tensor.shape  # (batch_size, ..., feature_dim) or (batch_size, ...)
        
        tokens = []

        # [CLS]トークンを追加（形状に合わせて拡張）
        cls_tokens = self.cls_token.expand(*shape, -1)  # (..., d_token)
        tokens.append(cls_tokens)

        # 数値特徴量のトークン化
        if x_num is not None:
            for name in self.numerical_features:
                if name in x_num:
                    feature_values = x_num[name]  # (...,)
                    
                    # NaN処理
                    nan_mask = torch.isnan(feature_values)
                    clean_values = torch.where(nan_mask, torch.zeros_like(feature_values), feature_values)
                    
                    # Linearレイヤーのために最後に次元を追加
                    clean_values = clean_values.unsqueeze(-1)  # (..., 1)
                    token = self.numerical_tokenizers[name](clean_values)  # (..., d_token)

                    # NaN位置のトークンをゼロにする
                    nan_mask = nan_mask.unsqueeze(-1)  # (..., 1)
                    token = torch.where(nan_mask, torch.zeros_like(token), token)
                    tokens.append(token)

        # カテゴリ特徴量のトークン化
        if x_cat is not None:
            for name in self.categorical_features.keys():
                if name in x_cat:
                    feature_values = x_cat[name]  # (...,)
                    token = self.categorical_tokenizers[name](feature_values)  # (..., d_token)
                    tokens.append(token)

        # トークンを結合
        result = torch.stack(tokens, dim=-2)  # (..., num_tokens, d_token)
        
        return result
    
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
            if mask.dim() == 2:
                # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
                attention_mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 4:
                attention_mask = mask
            else:
                raise ValueError(f"Unexpected mask dimension: {mask.dim()}")
            # (batch_size, 1, 1, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
            expanded_mask = attention_mask.expand(batch_size, self.n_heads, seq_len, seq_len)
            scores = scores.masked_fill(expanded_mask == 0, -1e9)
        
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
        
        # 出力層にマークを付ける
        self.score_head[-1]._is_output_layer = True

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

        # 馬同士の相互作用を計算（maskはそのまま渡す）
        interaction_features = self.interaction_attention(horse_features, mask)

        # 残差接続と正規化
        combined_features = self.interaction_norm(horse_features + self.interaction_dropout(interaction_features))

        # 各馬の強さスコアを計算
        scores = self.score_head(combined_features).squeeze(-1)  # (batch_size, num_horses)

        # マスクされた馬のスコアを-infに設定 (softmax時に0になる)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)

        return scores


class Horgues3Model(nn.Module):
    """統合された競馬予測モデル"""

    def __init__(self, numerical_features, categorical_features, d_token=192, n_layers=3, n_heads=8, d_ffn=None, dropout=0.1, max_horses=18):
        super().__init__()
        self.d_token = d_token
        self.max_horses = max_horses
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        # 辞書形式の特徴量用のトークナイザー
        self.tokenizer = FeatureTokenizer(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
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

        # 重み初期化を追加
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """重みの初期化"""
        if isinstance(module, nn.Linear):
            # He初期化（ReLU活性化関数用）とXavier初期化を使い分け
            if hasattr(module, '_is_output_layer') and module._is_output_layer:
                # 出力層は小さい値で初期化
                nn.init.xavier_uniform_(module.weight, gain=0.1)
            else:
                # 中間層はHe初期化
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            # 埋め込み層の初期化
            nn.init.normal_(module.weight, std=0.02)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)
                    
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
        elif isinstance(module, nn.Parameter):
            # CLSトークンなどのパラメータ
            nn.init.normal_(module, std=0.02)

    def forward(self, x_num: Dict[str, torch.Tensor], x_cat: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None):
        """
        Args:
            x_num: Dict with keys like 'horse_weight', 'weight_change' - (batch_size, num_horses)
            x_cat: Dict with keys like 'weather_code' - (batch_size, num_horses)
            mask: (batch_size, num_horses) - 有効な馬のマスク (1=有効, 0=無効/パディング)
        """
        batch_size, max_horses = next(iter(x_num.values())).shape

        # 全ての馬の特徴量を一度にトークン化（2次元対応）
        # (batch_size, num_horses) -> (batch_size, num_horses, num_tokens, d_token)
        tokens = self.tokenizer(x_num, x_cat)
        
        # 各馬の特徴量を個別にFT Transformerで処理
        # (batch_size, num_horses, num_tokens, d_token) -> (batch_size * num_horses, num_tokens, d_token)
        batch_size, num_horses, num_tokens, d_token = tokens.shape
        tokens_flat = tokens.view(batch_size * num_horses, num_tokens, d_token)
        
        # FT Transformerで特徴抽出（バッチ処理）
        features_flat = self.ft_transformer(tokens_flat)  # (batch_size * num_horses, d_token)
        
        # 元の形状に戻す
        horse_features = features_flat.view(batch_size, num_horses, d_token)  # (batch_size, num_horses, d_token)

        # レース内相互作用を考慮したスコア計算
        scores = self.race_interaction(horse_features, mask)

        return scores
