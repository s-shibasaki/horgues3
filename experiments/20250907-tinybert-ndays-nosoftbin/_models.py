import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Dict, Any
from itertools import permutations
import logging


logger = logging.getLogger(__name__)


class FeatureTokenizer(nn.Module):
    """数値特徴量とカテゴリ特徴量をトークン化（共通利用）"""

    def __init__(self, numerical_features: List[str], categorical_features: Dict[str, int], feature_aliases: Dict[str, str],
                 d_token: int = 192, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.feature_aliases = feature_aliases

        # 数値特徴量用のLinear（特徴量ごとに個別）
        self.numerical_tokenizers = nn.ModuleDict()
        for feature in numerical_features:
            tokenizer_name = self.feature_aliases.get(feature, feature)
            if tokenizer_name not in self.numerical_tokenizers:
                self.numerical_tokenizers[tokenizer_name] = nn.Linear(1, d_token)

        # カテゴリ特徴量用の埋め込み
        self.categorical_tokenizers = nn.ModuleDict()
        for feature, vocab_size in categorical_features.items():
            tokenizer_name = self.feature_aliases.get(feature, feature)
            if tokenizer_name not in self.categorical_tokenizers:
                self.categorical_tokenizers[tokenizer_name] = nn.Embedding(vocab_size + 1, d_token, padding_idx=0) 

        # トークン統合用のMLPレイヤー
        n_tokens = len(numerical_features) + len(categorical_features)
        self.gelu = nn.GELU()
        self.linear = nn.Linear(n_tokens * d_token, d_token)
        self.norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # 数値特徴量用Linearの初期化
        for linear in self.numerical_tokenizers.values():
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.constant_(linear.bias, 0)
        
        # カテゴリ埋め込みの初期化
        for embedding in self.categorical_tokenizers.values():
            nn.init.normal_(embedding.weight, mean=0, std=0.02)
            # padding_idxは0で固定
            if embedding.padding_idx is not None:
                nn.init.constant_(embedding.weight[embedding.padding_idx], 0)
        
        # 統合用線形層の初期化
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
        
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
            output: (..., d_token) - 統合された特徴量
        """
        tokens = []

        # 数値特徴量のトークン化（定義された順序でループ）
        for name in self.numerical_features:
            tokenizer_name = self.feature_aliases.get(name, name)
            tokenizer = self.numerical_tokenizers[tokenizer_name]
            
            if x_num is not None and name in x_num:
                feature_values = x_num[name]
                
                # NaN処理
                nan_mask = torch.isnan(feature_values)
                clean_values = torch.where(nan_mask, torch.zeros_like(feature_values), feature_values)
                
                # Linearで変換（入力を1次元追加）
                clean_values_unsqueezed = clean_values.unsqueeze(-1)  # (..., 1)
                token = tokenizer(clean_values_unsqueezed)  # (..., d_token)

                # NaN位置のトークンをゼロにする
                nan_mask = nan_mask.unsqueeze(-1)  # (..., 1)
                token = torch.where(nan_mask, torch.zeros_like(token), token)
            else:
                # 特徴量が存在しない場合はゼロトークンを作成
                # 他の特徴量から形状を推定
                if x_num and len(x_num) > 0:
                    sample_tensor = next(iter(x_num.values()))
                elif x_cat and len(x_cat) > 0:
                    sample_tensor = next(iter(x_cat.values()))
                else:
                    # 何も特徴量がない場合は1次元のゼロテンソル
                    sample_tensor = torch.zeros(1, device=next(self.parameters()).device)
                
                # サンプルテンソルの形状でゼロトークンを作成
                token_shape = sample_tensor.shape + (self.d_token,)
                token = torch.zeros(token_shape, device=sample_tensor.device, dtype=torch.float32)
            
            tokens.append(token)

        # カテゴリ特徴量のトークン化（定義された順序でループ）
        for name in self.categorical_features.keys():
            tokenizer_name = self.feature_aliases.get(name, name)
            tokenizer = self.categorical_tokenizers[tokenizer_name]
            
            if x_cat is not None and name in x_cat:
                feature_values = x_cat[name]
                token = tokenizer(feature_values)  # (..., d_token)
            else:
                # 特徴量が存在しない場合はパディングIDでトークンを作成
                # 他の特徴量から形状を推定
                if x_cat and len(x_cat) > 0:
                    sample_tensor = next(iter(x_cat.values()))
                elif x_num and len(x_num) > 0:
                    sample_tensor = next(iter(x_num.values()))
                else:
                    # 何も特徴量がない場合は1次元のゼロテンソル
                    sample_tensor = torch.zeros(1, device=next(self.parameters()).device, dtype=torch.long)
                
                # サンプルテンソルの形状でパディングIDのテンソルを作成
                padding_shape = sample_tensor.shape
                padding_tensor = torch.zeros(padding_shape, device=sample_tensor.device, dtype=torch.long)
                token = tokenizer(padding_tensor)  # (..., d_token)
            
            tokens.append(token)

        # トークンをconcatenate
        concatenated = torch.cat(tokens, dim=-1)  # (..., n_tokens * d_token)

        # GELU -> Linear -> Norm -> Dropout
        output = self.gelu(concatenated)
        output = self.linear(output)  # (..., d_token)
        output = self.norm(output)
        output = self.dropout(output)
        
        return output
    

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
        self.linear = nn.Linear(d_token, d_token)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # スコア計算ヘッド
        self.score_head = nn.Linear(d_token, 3)

        self._init_weights()

    def _init_weights(self):
        """重みの初期化"""
        # LayerNormの初期化
        nn.init.constant_(self.norm.weight, 1.0)
        nn.init.constant_(self.norm.bias, 0.0)
        
        # Linearレイヤーの初期化
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
        
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
            scores: (batch_size, num_horses) - 相互作用を考慮した馬の強さスコア
        """
        # NaNをゼロで置換（マスクされた馬の特徴量）
        horse_vectors = torch.where(torch.isnan(horse_vectors), torch.zeros_like(horse_vectors), horse_vectors)

        # Transformerブロックを通す (相互作用を学習)
        x = horse_vectors
        for block in self.transformer_blocks:
            x = block(x, mask)

        # 正規化
        x = self.norm(x)
        
        # linear -> gelu -> dropout -> score_head の順序で処理
        x = self.linear(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        # 各馬の強さスコアを計算
        scores = self.score_head(x)  # (batch_size, num_horses, 3)

        return scores


class HorguesModel(nn.Module):
    """統合された競馬予測モデル"""

    def __init__(
            self,
            sequence_names: List[str],
            feature_aliases: Dict[str, str],
            numerical_features: List[str],
            categorical_features: Dict[str, int],

            # 次元数 - 標準的な値に変更
            d_token: int = 256,  # 中程度の表現力を持つ標準的な次元数

            # 時系列統合Transformer - 標準的な設定
            seq_n_layers: int = 2,
            seq_n_heads: int = 4,
            seq_d_ffn: int = 1024,  # d_token * 4 の標準的な比率

            # レース内相互作用Transformer - 標準的な設定
            race_n_layers: int = 2,  # より軽量に
            race_n_heads: int = 4,
            race_d_ffn: int = 1024,  # d_token * 4 の標準的な比率

            # 過学習防止 - 標準的なドロップアウト率
            dropout: float = 0.1
    ):
        super().__init__()
        self.sequence_names = sequence_names
        self.feature_aliases = feature_aliases
        self.d_token = d_token

        # dataset_params から必要な情報を抽出
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        # 共通のFeatureTokenizer（通常のLinear使用）
        self.tokenizer = FeatureTokenizer(
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            feature_aliases=self.feature_aliases,
            d_token=d_token,
            dropout=dropout
        )

        # 時系列用のSequenceTransformer (単一のTransformer)
        self.sequence_transformer = SequenceTransformer(
            d_token=d_token,
            n_layers=seq_n_layers,
            n_heads=seq_n_heads,
            d_ffn=seq_d_ffn,
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
            scores: (batch_size, num_horses) - 各馬の強さスコア
        """
        batch_size, num_horses = mask.shape if mask is not None else (list(x_num.values())[0].shape[:2] if x_num else list(x_cat.values())[0].shape[:2])
        
        # 各馬の最終特徴量を格納するリスト
        horse_features = []

        for horse_idx in range(num_horses):
            # 全てのシーケンスを結合するためのリスト
            all_sequence_tokens = []
            all_sequence_masks = []
            
            # 1. 一般特徴量を最初に追加
            general_x_num = {}
            general_x_cat = {}
            
            for key, val in (x_num or {}).items():
                general_x_num[key] = val[:, horse_idx]  # (batch_size,)
            
            for key, val in (x_cat or {}).items():
                general_x_cat[key] = val[:, horse_idx]  # (batch_size,)

            # 一般データをトークン化してunsqueezeで時系列次元を追加
            general_tokens = self.tokenizer(general_x_num or None, general_x_cat or None)  # (batch_size, d_token)
            general_tokens = general_tokens.unsqueeze(1)  # (batch_size, 1, d_token)
            all_sequence_tokens.append(general_tokens)
            
            # 一般特徴量のマスク（常に有効）
            general_mask = torch.ones(batch_size, 1, device=general_tokens.device)
            all_sequence_masks.append(general_mask)
            
            # 2. 時系列データを順次追加
            for seq_name in self.sequence_names:
                if sequence_data and seq_name in sequence_data:
                    seq_data = sequence_data[seq_name]
                    
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

                    # FeatureTokenizerでトークン化
                    seq_tokens = self.tokenizer(seq_x_num or None, seq_x_cat or None)  # (batch_size, seq_len, d_token) または (batch_size, d_token)
                    
                    # 時系列の場合
                    if seq_tokens.dim() == 3:  # (batch_size, seq_len, d_token)
                        all_sequence_tokens.append(seq_tokens)
                        if seq_mask is not None:
                            all_sequence_masks.append(seq_mask)
                        else:
                            # マスクがない場合は全て有効とする
                            seq_len = seq_tokens.shape[1]
                            default_mask = torch.ones(batch_size, seq_len, device=seq_tokens.device)
                            all_sequence_masks.append(default_mask)
                    else:  # (batch_size, d_token) - 非時系列データ
                        seq_tokens = seq_tokens.unsqueeze(1)  # (batch_size, 1, d_token)
                        all_sequence_tokens.append(seq_tokens)
                        default_mask = torch.ones(batch_size, 1, device=seq_tokens.device)
                        all_sequence_masks.append(default_mask)

            # 3. 全てのシーケンスを結合
            if all_sequence_tokens:
                combined_tokens = torch.cat(all_sequence_tokens, dim=1)  # (batch_size, total_seq_len, d_token)
                combined_mask = torch.cat(all_sequence_masks, dim=1)  # (batch_size, total_seq_len)
                
                # 単一のSequenceTransformerで処理
                horse_feature = self.sequence_transformer(combined_tokens, combined_mask)  # (batch_size, d_token)
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


