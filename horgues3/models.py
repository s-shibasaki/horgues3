import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Dict, Any
from itertools import permutations
import logging


logger = logging.getLogger(__name__)


class SoftBinning(nn.Module):
    """
    標準化済み数値特徴量をソフトビニングするモジュール
    
    Args:
        num_bins: ビンの数
        temperature: ソフトマックスの温度パラメータ（低いほどハード、高いほどソフト）
        init_range: ビン中心の初期化範囲 [-init_range, init_range]
    """
    
    def __init__(self, num_bins: int = 10, temperature: float = 1.0, init_range: float = 3.0):
        super().__init__()
        self.num_bins = num_bins
        self.temperature = temperature
        
        # 学習可能なビン中心（標準化されたデータを想定して初期化）
        self.bin_centers = nn.Parameter(
            torch.linspace(-init_range, init_range, num_bins)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 標準化済み数値特徴量 (...,) 任意の形状
        Returns:
            binned: ソフトビニング結果 (..., num_bins)
        """
        # 入力の形状を保存
        original_shape = x.shape
        
        # xを展開して計算しやすくする
        x_flat = x.view(-1, 1)  # (N, 1)
        bin_centers = self.bin_centers.view(1, -1)  # (1, num_bins)
        
        # 各ビン中心との距離を計算（負の二乗距離）
        distances = -(x_flat - bin_centers) ** 2  # (N, num_bins)
        
        # 温度でスケールしてソフトマックス
        soft_assignments = F.softmax(distances / self.temperature, dim=-1)  # (N, num_bins)
        
        # 元の形状に戻す
        output_shape = original_shape + (self.num_bins,)
        return soft_assignments.view(output_shape)




class SoftBinnedLinear(nn.Module):
    """
    数値特徴量をSoftBinningしてからLinear変換するモジュール
    
    Args:
        num_bins: ビンの数
        d_token: 出力次元
        temperature: ソフトマックスの温度パラメータ
        init_range: ビン中心の初期化範囲
        dropout: ドロップアウト率
    """
    
    def __init__(self, num_bins: int = 10, d_token: int = 192, temperature: float = 1.0, 
                 init_range: float = 3.0):
        super().__init__()
        self.soft_binning = SoftBinning(
            num_bins=num_bins,
            temperature=temperature,
            init_range=init_range
        )
        self.linear = nn.Linear(num_bins, d_token)

        self._init_weights()
        
    def _init_weights(self):
        """重みの初期化"""
        # Linearレイヤーの重み初期化 (Xavier uniform)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 標準化済み数値特徴量 (...,) 任意の形状
        Returns:
            output: (..., d_token) Linear変換後の特徴量
        """
        # SoftBinning適用
        binned = self.soft_binning(x)  # (..., num_bins)
        
        # Linear変換
        output = self.linear(binned)  # (..., d_token)
        
        return output


class FeatureTokenizer(nn.Module):
    """数値特徴量とカテゴリ特徴量をトークン化（共通利用）"""

    def __init__(self, numerical_features: List[str], categorical_features: Dict[str, int], feature_aliases: Dict[str, str],
                 d_token: int = 192, num_bins: int = 10, binning_temperature: float = 1.0,
                 binning_init_range: float = 3.0, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.feature_aliases = feature_aliases

        # 数値特徴量用のSoftBinnedLinear（特徴量ごとに個別）
        self.numerical_tokenizers = nn.ModuleDict()
        for feature in numerical_features:
            tokenizer_name = self.feature_aliases.get(feature, feature)
            if tokenizer_name not in self.numerical_tokenizers:
                self.numerical_tokenizers[tokenizer_name] = SoftBinnedLinear(
                    num_bins=num_bins,
                    d_token=d_token,
                    temperature=binning_temperature,
                    init_range=binning_init_range,
                )

        # カテゴリ特徴量用の埋め込み
        self.categorical_tokenizers = nn.ModuleDict()
        for feature, vocab_size in categorical_features.items():
            tokenizer_name = self.feature_aliases.get(feature, feature)
            if tokenizer_name not in self.categorical_tokenizers:
                self.categorical_tokenizers[tokenizer_name] = nn.Embedding(vocab_size + 1, d_token, padding_idx=0) 

        self.norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # カテゴリ埋め込みの初期化
        for embedding in self.categorical_tokenizers.values():
            nn.init.normal_(embedding.weight, mean=0, std=0.02)
            # padding_idxは0で固定
            if embedding.padding_idx is not None:
                nn.init.constant_(embedding.weight[embedding.padding_idx], 0)
        
        # LayerNormの初期化（通常はデフォルトで適切）
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

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
            for name, feature_values in x_num.items():
                tokenizer_name = self.feature_aliases.get(name, name)
                tokenizer = self.numerical_tokenizers[tokenizer_name]
                
                # NaN処理
                nan_mask = torch.isnan(feature_values)
                clean_values = torch.where(nan_mask, torch.zeros_like(feature_values), feature_values)
                
                # SoftBinnedLinearで変換
                token = tokenizer(clean_values)  # (..., d_token)

                # NaN位置のトークンをゼロにする
                nan_mask = nan_mask.unsqueeze(-1)  # (..., 1)
                token = torch.where(nan_mask, torch.zeros_like(token), token)
                tokens.append(token)

        # カテゴリ特徴量のトークン化
        if x_cat is not None:
            for name, feature_values in x_cat.items():
                tokenizer_name = self.feature_aliases.get(name, name)
                tokenizer = self.categorical_tokenizers[tokenizer_name]
                token = tokenizer(feature_values)  # (..., d_token)
                tokens.append(token)

        # トークンを結合
        result = torch.stack(tokens, dim=-2)  # (..., num_tokens, d_token)

        # 正規化とドロップアウト
        result = self.norm(result)
        result = self.dropout(result)
        
        return result
    

class AttentionHead(nn.Module):
    def __init__(self, d_token: int = 192, d_head: int = 8):
        super().__init__()
        self.q = nn.Linear(d_token, d_head)
        self.k = nn.Linear(d_token, d_head)
        self.v = nn.Linear(d_token, d_head)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # Attention用の線形層の初期化 (Xavier uniform)
        for linear in [self.q, self.k, self.v]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.constant_(linear.bias, 0)

    def forward(self, x, mask=None):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        dim_k = query.size(-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / np.sqrt(dim_k)

        # マスクを適用
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch_size, seq_len) -> (batch_size, seq_len, seq_len) にブロードキャスト
            scores = scores.masked_fill(~mask.bool(), -1e9)

        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, value)
    
class MultiHeadAttention(nn.Module):
    """マルチヘッドアテンション"""

    def __init__(self, d_token: int = 192, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_token % n_heads == 0

        self.d_token = d_token
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        self.heads = nn.ModuleList(
            [AttentionHead(d_token, self.d_head) for _ in range(n_heads)]
        )
        self.output_linear = nn.Linear(d_token, d_token)

    def _init_weights(self):
        """重みの初期化"""
        # 出力層の初期化
        nn.init.xavier_uniform_(self.output_linear.weight)
        if self.output_linear.bias is not None:
            nn.init.constant_(self.output_linear.bias, 0)

    def forward(self, x, mask=None):
        x = torch.cat([h(x, mask) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x

class FeedForward(nn.Module):
    """フィードフォワードネットワーク"""

    def __init__(self, d_token: int, d_ffn: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_token, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_token)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # He初期化 (GELU活性化関数に適している)
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        
        if self.linear1.bias is not None:
            nn.init.constant_(self.linear1.bias, 0)
        if self.linear2.bias is not None:
            nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x  # (batch_size, seq_len, d_token)

class TransformerBlock(nn.Module):
    """Transformerブロック"""

    def __init__(self, d_token: int, n_heads: int = 8, d_ffn: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ffn is None:
            d_ffn = d_token * 4

        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        self.attention = MultiHeadAttention(d_token, n_heads, dropout)
        self.feed_forward = FeedForward(d_token, d_ffn, dropout)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # LayerNormの初期化
        for norm in [self.norm1, self.norm2]:
            nn.init.constant_(norm.weight, 1.0)
            nn.init.constant_(norm.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        hidden_state = self.norm1(x)
        x = x + self.attention(hidden_state, mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class SequenceTransformer(nn.Module):
    """時系列データ処理用のTransformer (CLSトークンを出力)"""

    def __init__(self, d_token: int = 192, n_layers: int = 3, n_heads: int = 8, d_ffn: int = None, 
                 dropout: float = 0.1, max_seq_len: int = 10000):
        super().__init__()
        self.d_token = d_token
        self.max_seq_len = max_seq_len

        # [CLS]トークン
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))

        # 学習可能な位置エンコーディング
        # CLSトークン + 最大シーケンス長分を用意
        self.position_embeddings = nn.Parameter(torch.randn(1, max_seq_len + 1, d_token))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # CLSトークンの初期化
        nn.init.normal_(self.cls_token, mean=0, std=0.02)
        
        # 位置エンコーディングの初期化
        nn.init.normal_(self.position_embeddings, mean=0, std=0.02)
        
        # LayerNormの初期化
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)

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
        
        # シーケンス長のチェック
        if seq_len > self.max_seq_len:
            logger.warning(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}. Truncating.")
            sequence_tokens = sequence_tokens[..., :self.max_seq_len, :]
            if mask is not None:
                mask = mask[..., :self.max_seq_len]
            seq_len = self.max_seq_len
        
        # バッチ次元をまとめる
        sequence_tokens_flat = sequence_tokens.view(batch_size, seq_len, d_token)

        # [CLS]トークンを先頭に追加
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_token)
        tokens_with_cls = torch.cat([cls_tokens, sequence_tokens_flat], dim=1)  # (batch_size, seq_len + 1, d_token)

        # 位置エンコーディングを追加
        seq_len_with_cls = seq_len + 1
        pos_embeddings = self.position_embeddings[:, :seq_len_with_cls, :]  # (1, seq_len + 1, d_token)
        tokens_with_cls = tokens_with_cls + pos_embeddings

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

    def __init__(self, d_token: int = 192, n_layers: int = 3, n_heads: int = 8, d_ffn: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

        # 3つのスコア計算ヘッド（1着、2着、3着確率用）
        self.score_head = nn.Linear(d_token, 3)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # LayerNormの初期化
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)
        
        # スコアヘッドの初期化（出力の分散を小さくする）
        nn.init.xavier_uniform_(self.score_head.weight)
        if self.score_head.bias is not None:
            nn.init.constant_(self.score_head.bias, 0)

    def forward(self, horse_vectors: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            horse_vectors: (batch_size, num_horses, d_token) - 各馬のベクトル
            mask: (batch_size, num_horses) - 有効な馬のマスク (1=有効, 0=無効/パディング)

        Returns:
            scores: (batch_size, num_horses, 3) - 相互作用を考慮した馬の3つのスコア
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
        
        # 各馬の3つのスコアを計算
        scores = self.score_head(x)  # (batch_size, num_horses, 3)

        # マスクされた馬のスコアを-1e9に設定 (softmax時に0になる)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, 3)  # (batch_size, num_horses, 3)
            scores = scores.masked_fill(~mask_expanded.bool(), -1e9)

        return scores


class HorguesModel(nn.Module):
    """統合された競馬予測モデル"""

    def __init__(self,
                 sequence_names: List[str],
                 feature_aliases: Dict[str, str],
                 numerical_features: List[str],
                 categorical_features: Dict[str, int],

                 # 次元数 - 標準的な値に変更
                 d_token: int = 256,  # 中程度の表現力を持つ標準的な次元数

                 # SoftBinning 設定 - 標準的な値
                 num_bins: int = 10,  # より一般的なビン数
                 binning_temperature: float = 1.0,  # 標準的な温度パラメータ
                 binning_init_range: float = 3.0,  # 標準的な初期化範囲

                 # 特徴量統合Transformer - 標準的な軽量設定
                 ft_n_layers: int = 3,
                 ft_n_heads: int = 8,  # 標準的なヘッド数
                 ft_d_ffn: int = 1024,  # d_token * 4 の標準的な比率

                 # 時系列統合Transformer - 標準的な設定
                 seq_n_layers: int = 3,
                 seq_n_heads: int = 8,
                 seq_d_ffn: int = 1024,  # d_token * 4 の標準的な比率

                 # レース内相互作用Transformer - 標準的な設定
                 race_n_layers: int = 3,  # より軽量に
                 race_n_heads: int = 8,
                 race_d_ffn: int = 1024,  # d_token * 4 の標準的な比率

                 # 過学習防止 - 標準的なドロップアウト率
                 dropout: float = 0.1):
        super().__init__()
        self.sequence_names = sequence_names
        self.feature_aliases = feature_aliases
        self.d_token = d_token

        # dataset_params から必要な情報を抽出
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        # 共通のFeatureTokenizer（SoftBinning対応）
        self.tokenizer = FeatureTokenizer(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            feature_aliases=self.feature_aliases,
            d_token=d_token,
            num_bins=num_bins,
            binning_temperature=binning_temperature,
            binning_init_range=binning_init_range,
            dropout=dropout
        )

        # 時系列用のFTTransformer (特徴量統合用)
        self.sequence_ft_transformers = nn.ModuleDict()
        for seq_name in self.sequence_names:
            self.sequence_ft_transformers[seq_name] = SequenceTransformer(
                d_token=d_token,
                n_layers=ft_n_layers,
                n_heads=ft_n_heads,
                d_ffn=ft_d_ffn,
                dropout=dropout
            )

        # 時系列用のSequenceTransformer (時系列処理用)
        self.sequence_transformers = nn.ModuleDict()
        for seq_name in self.sequence_names:
            self.sequence_transformers[seq_name] = SequenceTransformer(
                d_token=d_token,
                n_layers=seq_n_layers,
                n_heads=seq_n_heads,
                d_ffn=seq_d_ffn,
                dropout=dropout
            )

        self.general_ft_transformer = SequenceTransformer(
            d_token=d_token,
            n_layers=ft_n_layers,
            n_heads=ft_n_heads,
            d_ffn=ft_d_ffn,
            dropout=dropout
        )

        # 最終統合用のFTTransformer
        self.final_ft_transformer = SequenceTransformer(
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
            scores: (batch_size, num_horses, 3) - 各馬の強さスコア
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
        scores = self.race_transformer(horse_vectors, mask)  # (batch_size, num_horses, 3)

        return scores


