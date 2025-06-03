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
    """数値特徴量とカテゴリ特徴量をトークン化（共通利用）"""

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

    def forward(self, x_num: Optional[Dict[str, torch.Tensor]] = None, 
                x_cat: Optional[Dict[str, torch.Tensor]] = None):
        """
        Args:
            x_num: 数値特徴量 - 各値は任意の形状のテンソル
            x_cat: カテゴリ特徴量 - 各値は任意の形状のテンソル
        Returns:
            tokens: (..., num_tokens, d_token) - トークン化された特徴量
        """
        tokens = []

        # 数値特徴量のトークン化
        if x_num is not None:
            for name in self.numerical_features:
                if name in x_num:
                    feature_values = x_num[name]  # 任意の形状
                    
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
                    feature_values = x_cat[name]  # 任意の形状
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
            # mask: (batch_size, seq_len)
            attention_mask = mask.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            expanded_mask = attention_mask.expand(batch_size, self.n_heads, seq_len, seq_len)  # (batch_size, n_heads, seq_len, seq_len)
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
    """FT Transformer - 複数の特徴量を統合してCLSトークンを出力"""

    def __init__(self, d_token: int = 192, n_layers: int = 2, n_heads: int = 8, d_ffn: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token

        # [CLS]トークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            feature_tokens: (..., num_features, d_token) - 特徴量トークン
            mask: (..., num_features) - 特徴量マスク (1=有効, 0=無効/パディング)
        Returns:
            cls_output: (..., d_token) - CLSトークンの出力
        """
        # 入力の形状を取得
        *batch_dims, num_features, d_token = feature_tokens.shape
        batch_size = int(np.prod(batch_dims))
        
        # バッチ次元をまとめる
        feature_tokens_flat = feature_tokens.view(batch_size, num_features, d_token)
        
        # [CLS]トークンを先頭に追加
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_token)
        tokens = torch.cat([cls_tokens, feature_tokens_flat], dim=1)  # (batch_size, num_features + 1, d_token)

        # マスクも[CLS]トークン分を拡張
        if mask is not None:
            mask_flat = mask.view(batch_size, num_features)
            cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
            mask_with_cls = torch.cat([cls_mask, mask_flat], dim=1)  # (batch_size, num_features + 1)
        else:
            mask_with_cls = None

        # Transformerブロックを通す
        x = tokens
        for block in self.transformer_blocks:
            x = block(x, mask_with_cls)

        # [CLS]トークン（最初のトークン）を抽出
        cls_token = x[:, 0]
        cls_token = self.norm(cls_token)
        cls_token = self.dropout(cls_token)

        # 元の形状に戻す
        cls_output = cls_token.view(*batch_dims, d_token)

        return cls_output


class SequenceTransformer(nn.Module):
    """時系列データ処理用のTransformer (CLSトークンを出力)"""

    def __init__(self, d_token: int = 192, n_layers: int = 4, n_heads: int = 8, d_ffn: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token

        # [CLS]トークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            sequence_tokens: (..., seq_len, d_token) - 時系列トークン
            mask: (..., seq_len) - シーケンスマスク (1=有効, 0=無効/パディング)
        Returns:
            cls_output: (..., d_token) - CLSトークンの出力
        """
        # 入力の形状を取得
        *batch_dims, seq_len, d_token = sequence_tokens.shape
        batch_size = int(np.prod(batch_dims))
        
        # バッチ次元をまとめる
        sequence_tokens_flat = sequence_tokens.view(batch_size, seq_len, d_token)

        # [CLS]トークンを先頭に追加
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_token)
        tokens_with_cls = torch.cat([cls_tokens, sequence_tokens_flat], dim=1)  # (batch_size, seq_len + 1, d_token)

        # マスクも[CLS]トークン分を拡張
        if mask is not None:
            mask_flat = mask.view(batch_size, seq_len)
            cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
            mask_with_cls = torch.cat([cls_mask, mask_flat], dim=1)  # (batch_size, seq_len + 1)
        else:
            mask_with_cls = None

        # Transformerブロックを通す
        x = tokens_with_cls
        for block in self.transformer_blocks:
            x = block(x, mask_with_cls)

        # [CLS]トークン（最初のトークン）を抽出
        cls_token = x[:, 0]
        cls_token = self.norm(cls_token)
        cls_token = self.dropout(cls_token)

        # 元の形状に戻す
        cls_output = cls_token.view(*batch_dims, d_token)

        return cls_output


class RaceTransformer(nn.Module):
    """レース内の馬の相互作用を処理するTransformer (CLSトークンなし)"""

    def __init__(self, d_token: int = 192, n_layers: int = 1, n_heads: int = 8, d_ffn: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

        # 最終的な強さスコアを計算
        self.score_head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, 1)
        )

        # 出力層にマークを付ける
        self.score_head[-1]._is_output_layer = True

    def forward(self, horse_vectors: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            horse_vectors: (batch_size, num_horses, d_token) - 各馬のベクトル
            mask: (batch_size, num_horses) - 有効な馬のマスク (1=有効, 0=無効/パディング)

        Returns:
            scores: (batch_size, num_horses) - 相互作用を考慮した馬の強さスコア
        """
        # NaNをゼロで置換（マスクされた馬の特徴量）
        horse_vectors = torch.where(torch.isnan(horse_vectors), torch.zeros_like(horse_vectors), horse_vectors)

        # Transformerブロックを通す (相互作用を学習)
        x = horse_vectors
        for block in self.transformer_blocks:
            x = block(x, mask)

        # 正規化とドロップアウト
        x = self.norm(x)
        x = self.dropout(x)
        
        # 各馬の強さスコアを計算
        scores = self.score_head(x).squeeze(-1)  # (batch_size, num_horses)

        # マスクされた馬のスコアを-1e-9に設定 (softmax時に0になる)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), -1e9)

        return scores


class HorguesModel(nn.Module):
    """統合された競馬予測モデル"""

    def __init__(self, 
                 numerical_features: List[str],
                 categorical_features: Dict[str, int],
                 sequence_configs: Dict[str, Dict] = None,
                 d_token: int = 192,
                 ft_n_layers: int = 2,
                 ft_n_heads: int = 8,
                 ft_d_ffn: int = None,
                 seq_n_layers: int = 4,
                 seq_n_heads: int = 8,
                 seq_d_ffn: int = None,
                 race_n_layers: int = 1,
                 race_n_heads: int = 8,
                 race_d_ffn: int = None,
                 dropout: float = 0.1,
                 max_horses: int = 18):
        super().__init__()
        self.d_token = d_token
        self.max_horses = max_horses
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.sequence_configs = sequence_configs or {}

        # 共通のFeatureTokenizer
        self.tokenizer = FeatureTokenizer(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            d_token=d_token
        )

        # 時系列用のFTTransformer (特徴量統合用)
        self.sequence_ft_transformers = nn.ModuleDict()
        for seq_name, seq_config in self.sequence_configs.items():
            self.sequence_ft_transformers[seq_name] = FTTransformer(
                d_token=d_token,
                n_layers=seq_config.get('ft_n_layers', ft_n_layers),
                n_heads=seq_config.get('ft_n_heads', ft_n_heads),
                d_ffn=seq_config.get('ft_d_ffn', ft_d_ffn),
                dropout=dropout
            )

        # 時系列用のSequenceTransformer (時系列処理用)
        self.sequence_transformers = nn.ModuleDict()
        for seq_name, seq_config in self.sequence_configs.items():
            self.sequence_transformers[seq_name] = SequenceTransformer(
                d_token=d_token,
                n_layers=seq_config.get('seq_n_layers', seq_n_layers),
                n_heads=seq_config.get('seq_n_heads', seq_n_heads),
                d_ffn=seq_config.get('seq_d_ffn', seq_d_ffn),
                dropout=dropout
            )

        self.general_ft_transformer = FTTransformer(
            d_token=d_token,
            n_layers=ft_n_layers,
            n_heads=ft_n_heads,
            d_ffn=ft_d_ffn,
            dropout=dropout
        )

        # 最終統合用のFTTransformer
        self.final_ft_transformer = FTTransformer(
            d_token=d_token,
            n_layers=ft_n_layers,
            n_heads=ft_n_heads,
            d_ffn=ft_d_ffn,
            dropout=dropout
        )

        # Race Transformer (レース内相互作用用)
        self.race_transformer = RaceTransformer(
            d_token=d_token,
            n_layers=race_n_layers,
            n_heads=race_n_heads,
            d_ffn=race_d_ffn,
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

    def forward(self,
                x_num: Dict[str, torch.Tensor],
                x_cat: Dict[str, torch.Tensor],
                sequence_data: Dict[str, Dict] = None,
                mask: Optional[torch.Tensor] = None):
        """
        Args:
            x_num: Dict - 一般数値特徴量 (batch_size, num_horses)
            x_cat: Dict - 一般カテゴリ特徴量 (batch_size, num_horses)
            sequence_data: Dict - 時系列データ
                例: {
                    'jockey_history': {
                        'x_num': {...},  # (batch_size, num_horses, seq_len)
                        'x_cat': {...},  # (batch_size, num_horses, seq_len)
                        'mask': ...      # (batch_size, num_horses, seq_len)
                    }
                }
            mask: (batch_size, num_horses) - 有効な馬のマスク
        Returns:
            scores: (batch_size, num_horses) - 各馬の強さスコア
        """
        batch_size, num_horses = mask.shape if mask is not None else (list(x_num.values())[0].shape[:2] if x_num else list(x_cat.values())[0].shape[:2])
        
        # 各馬の最終特徴量を格納するリスト
        horse_features = []

        for horse_idx in range(num_horses):
            # 1. 時系列データの処理
            sequence_features = []
            
            for seq_name, seq_data in (sequence_data or {}).items():
                # 時系列データから該当馬のデータを抽出
                seq_x_num = {}
                seq_x_cat = {}
                seq_mask = None
                
                if 'x_num' in seq_data:
                    for key, val in seq_data['x_num'].items():
                        seq_x_num[key] = val[:, horse_idx]  # (batch_size, seq_len)
                
                if 'x_cat' in seq_data:
                    for key, val in seq_data['x_cat'].items():
                        seq_x_cat[key] = val[:, horse_idx]  # (batch_size, seq_len)
                
                if 'mask' in seq_data:
                    seq_mask = seq_data['mask'][:, horse_idx]  # (batch_size, seq_len)

                # FeatureTokenizer -> FTTransformer -> SequenceTransformer
                seq_tokens = self.tokenizer(seq_x_num or None, seq_x_cat or None)  # (batch_size, seq_len, num_features, d_token)
                
                # 各時系列ステップの特徴量を統合
                if seq_tokens.size(-2) > 0:  # 特徴量が存在する場合
                    batch_size_seq, seq_len, num_features, d_token = seq_tokens.shape
                    seq_tokens_reshaped = seq_tokens.view(batch_size_seq * seq_len, num_features, d_token)
                    
                    # FTTransformerで各時系列ステップの特徴量を統合
                    step_features = self.sequence_ft_transformers[seq_name](seq_tokens_reshaped)  # (batch_size * seq_len, d_token)
                    step_features = step_features.view(batch_size_seq, seq_len, d_token)  # (batch_size, seq_len, d_token)
                    
                    # SequenceTransformerで時系列を処理
                    seq_feature = self.sequence_transformers[seq_name](step_features, seq_mask)  # (batch_size, d_token)
                    sequence_features.append(seq_feature)

            # 2. 一般データの処理
            general_x_num = {}
            general_x_cat = {}
            
            for key, val in (x_num or {}).items():
                general_x_num[key] = val[:, horse_idx]  # (batch_size,)
            
            for key, val in (x_cat or {}).items():
                general_x_cat[key] = val[:, horse_idx]  # (batch_size,)

            # 一般データをトークン化
            general_tokens = self.tokenizer(general_x_num or None, general_x_cat or None)  # (batch_size, num_features, d_token)
            
            # 一般データの特徴量を統合
            if general_tokens.size(-2) > 0:  # 特徴量が存在する場合
                general_feature = self.general_ft_transformer(general_tokens)  # (batch_size, d_token)
                sequence_features.append(general_feature)

            # 3. 最終統合
            if sequence_features:
                # 全ての特徴量を統合
                all_features = torch.stack(sequence_features, dim=1)  # (batch_size, num_feature_types, d_token)
                horse_feature = self.final_ft_transformer(all_features)  # (batch_size, d_token)
            else:
                # 特徴量がない場合はゼロベクトル
                horse_feature = torch.zeros(batch_size, self.d_token, device=next(self.parameters()).device)
            
            horse_features.append(horse_feature)

        # 4. レース内相互作用の処理
        # 全馬のベクトルを結合
        horse_vectors = torch.stack(horse_features, dim=1)  # (batch_size, num_horses, d_token)
        
        # RaceTransformerで相互作用を考慮した強さスコアを計算
        scores = self.race_transformer(horse_vectors, mask)  # (batch_size, num_horses)

        return scores
