import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union, Dict
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
                 init_range: float = 3.0, dropout: float = 0.1):
        super().__init__()
        self.soft_binning = SoftBinning(
            num_bins=num_bins,
            temperature=temperature,
            init_range=init_range
        )
        self.linear = nn.Linear(num_bins, d_token)
        self.dropout = nn.Dropout(dropout)
        
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
        
        # ドロップアウト
        output = self.dropout(output)
        
        return output


class FeatureTokenizer(nn.Module):
    """数値特徴量とカテゴリ特徴量をトークン化（共通利用）"""

    def __init__(self, numerical_features: List[str], categorical_features: Dict[str, int], 
                 d_token: int = 192, num_bins: int = 10, binning_temperature: float = 1.0,
                 binning_init_range: float = 3.0, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        # 数値特徴量用のSoftBinnedLinear（特徴量ごとに個別）
        self.numerical_tokenizers = nn.ModuleDict({
            name: SoftBinnedLinear(
                num_bins=num_bins,
                d_token=d_token,
                temperature=binning_temperature,
                init_range=binning_init_range,
                dropout=dropout
            ) for name in numerical_features
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
                    
                    # SoftBinnedLinearで変換
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

    def __init__(self, d_token: int = 192, n_layers: int = 3, n_heads: int = 8, d_ffn: int = None, dropout: float = 0.1):
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

    def __init__(self, d_token: int = 192, n_layers: int = 3, n_heads: int = 8, d_ffn: int = None, dropout: float = 0.1):
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

    def __init__(self, d_token: int = 192, n_layers: int = 3, n_heads: int = 8, d_ffn: int = None, dropout: float = 0.1):
        super().__init__()
        self.d_token = d_token

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_token)
        self.dropout = nn.Dropout(dropout)

        # スコア計算ヘッドを改良（より安定した出力のため）
        self.score_head = nn.Sequential(
            nn.Linear(d_token, d_token // 2),
            nn.LayerNorm(d_token // 2),  # LayerNormを追加
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 2, d_token // 4),
            nn.LayerNorm(d_token // 4),  # LayerNormを追加
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token // 4, 1)
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
                 d_token: int = 96,
                 # SoftBinning関連のパラメータを追加
                 num_bins: int = 10,
                 binning_temperature: float = 1.0,
                 binning_init_range: float = 3.0,
                 # 既存のパラメータ
                 ft_n_layers: int = 2,
                 ft_n_heads: int = 4,
                 ft_d_ffn: int = None,
                 seq_n_layers: int = 2,
                 seq_n_heads: int = 4,
                 seq_d_ffn: int = None,
                 race_n_layers: int = 2,
                 race_n_heads: int = 4,
                 race_d_ffn: int = None,
                 dropout: float = 0.3,
                 max_horses: int = 18):
        super().__init__()
        self.d_token = d_token
        self.max_horses = max_horses
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.sequence_configs = sequence_configs or {}

        # 共通のFeatureTokenizer（SoftBinning対応）
        self.tokenizer = FeatureTokenizer(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            d_token=d_token,
            num_bins=num_bins,
            binning_temperature=binning_temperature,
            binning_init_range=binning_init_range,
            dropout=dropout
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
        """重みの初期化を改善"""
        if isinstance(module, nn.Linear):
            if hasattr(module, '_is_output_layer') and module._is_output_layer:
                # 出力層は非常に小さい値で初期化（安定化のため）
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # 中間層はXavier初期化
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
        elif isinstance(module, nn.Embedding):
            # 埋め込み層の初期化
            nn.init.normal_(module.weight, std=0.1)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0.0)
                    
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
            
        elif isinstance(module, nn.Parameter):
            # CLSトークンなどのパラメータ
            nn.init.normal_(module, std=0.1)

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


class PlackettLuceLoss(nn.Module):
    """
    Plackett-Luce損失関数 - 着順予測用
    
    Args:
        temperature: ソフトマックスの温度パラメータ
        top_k: 上位k位までの損失を計算（Noneの場合は全ての順位を使用）
        reduction: 'mean', 'sum', 'none'
    """
    
    def __init__(self, temperature: float = 1.0, top_k: Optional[int] = None, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.reduction = reduction
    
    def forward(self, scores: torch.Tensor, rankings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: 予測スコア (batch_size, num_horses)
            rankings: 真の着順 (batch_size, num_horses) - 0-indexed、-1は無効
            mask: 有効な馬のマスク (batch_size, num_horses)
            
        Returns:
            loss: Plackett-Luce損失
        """
        batch_size, num_horses = scores.shape
        
        # 温度でスケール
        scaled_scores = scores / self.temperature
        
        # マスクされた馬のスコアを非常に小さい値に設定
        masked_scores = torch.where(mask, scaled_scores, torch.full_like(scaled_scores, -1e9))
        
        losses = []
        
        for batch_idx in range(batch_size):
            batch_scores = masked_scores[batch_idx]
            batch_rankings = rankings[batch_idx]
            batch_mask = mask[batch_idx]
            
            # 有効な馬のみを取得
            valid_horses = torch.where(batch_mask)[0]
            if len(valid_horses) == 0:
                continue
                
            valid_scores = batch_scores[valid_horses]
            valid_rankings = batch_rankings[valid_horses]
            
            # 無効な着順(-1)を除外
            valid_rank_mask = valid_rankings >= 0
            if not valid_rank_mask.any():
                continue
                
            final_horses = valid_horses[valid_rank_mask]
            final_scores = valid_scores[valid_rank_mask]
            final_rankings = valid_rankings[valid_rank_mask]
            
            if len(final_horses) <= 1:
                continue
            
            # 着順でソート
            sorted_indices = torch.argsort(final_rankings)
            sorted_scores = final_scores[sorted_indices]
            
            # top_kが指定されている場合は上位k位まで
            if self.top_k is not None:
                k = min(self.top_k, len(sorted_scores))
                sorted_scores = sorted_scores[:k]
            
            # Plackett-Luce損失を計算
            batch_loss = 0.0
            remaining_scores = sorted_scores.clone()
            
            for i in range(len(sorted_scores) - 1):
                # i位の馬が選ばれる確率のログ
                log_prob = remaining_scores[i] - torch.logsumexp(remaining_scores[i:], dim=0)
                batch_loss -= log_prob
                
                # 次の順位のために残りのスコアを更新（実際にはもう使わないが概念的に）
                # remaining_scores = remaining_scores[1:]  # 実際は次のループで[i+1:]を使う
            
            losses.append(batch_loss)
        
        if not losses:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        loss_tensor = torch.stack(losses)
        
        if self.reduction == 'mean':
            return loss_tensor.mean()
        elif self.reduction == 'sum':
            return loss_tensor.sum()
        else:  # 'none'
            return loss_tensor


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
            
            valid_rank_mask = valid_rankings >= 0
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


class ListwiseLoss(nn.Module):
    """
    リストワイズランキング損失 - 全体の順序を考慮
    
    Args:
        temperature: ソフトマックスの温度パラメータ
        top_k: 上位k位までを考慮
        reduction: 'mean', 'sum', 'none'
    """
    
    def __init__(self, temperature: float = 1.0, top_k: Optional[int] = None, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
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
            
            valid_rank_mask = valid_rankings >= 0
            if not valid_rank_mask.any():
                continue
                
            final_scores = valid_scores[valid_rank_mask]
            final_rankings = valid_rankings[valid_rank_mask]
            
            if len(final_scores) <= 1:
                continue
            
            # top_kが指定されている場合
            if self.top_k is not None:
                top_k_mask = final_rankings < self.top_k
                if not top_k_mask.any():
                    continue
                final_scores = final_scores[top_k_mask]
                final_rankings = final_rankings[top_k_mask]
            
            # 理想的な順序での確率分布を計算
            ideal_order = torch.argsort(final_rankings)
            ideal_scores = final_scores[ideal_order]
            
            # 予測スコアによる確率分布
            pred_probs = F.softmax(final_scores, dim=0)
            ideal_probs = F.softmax(ideal_scores, dim=0)
            
            # KLダイバージェンス
            kl_loss = F.kl_div(torch.log(pred_probs + 1e-8), ideal_probs, reduction='sum')
            losses.append(kl_loss)
        
        if not losses:
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        
        loss_tensor = torch.stack(losses)
        
        if self.reduction == 'mean':
            return loss_tensor.mean()
        elif self.reduction == 'sum':
            return loss_tensor.sum()
        else:
            return loss_tensor


class PairwiseRankingLoss(nn.Module):
    """
    ペアワイズランキング損失 - 順位の相対関係を学習
    
    Args:
        margin: マージン（デフォルト1.0）
        top_k: 上位k位までのペアを考慮
        reduction: 'mean', 'sum', 'none'
    """
    
    def __init__(self, margin: float = 1.0, top_k: Optional[int] = None, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.top_k = top_k
        self.reduction = reduction
    
    def forward(self, scores: torch.Tensor, rankings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_horses = scores.shape
        
        losses = []
        
        for batch_idx in range(batch_size):
            batch_scores = scores[batch_idx]
            batch_rankings = rankings[batch_idx]
            batch_mask = mask[batch_idx]
            
            valid_horses = torch.where(batch_mask)[0]
            if len(valid_horses) == 0:
                continue
                
            valid_scores = batch_scores[valid_horses]
            valid_rankings = batch_rankings[valid_horses]
            
            valid_rank_mask = valid_rankings >= 0
            if not valid_rank_mask.any():
                continue
                
            final_scores = valid_scores[valid_rank_mask]
            final_rankings = valid_rankings[valid_rank_mask]
            
            if len(final_scores) <= 1:
                continue
            
            # top_kが指定されている場合
            if self.top_k is not None:
                top_k_mask = final_rankings < self.top_k
                if not top_k_mask.any():
                    continue
                final_scores = final_scores[top_k_mask]
                final_rankings = final_rankings[top_k_mask]
            
            # ペアワイズ損失を計算
            batch_loss = 0.0
            num_pairs = 0
            
            for i in range(len(final_scores)):
                for j in range(len(final_scores)):
                    if i != j:
                        # i位の馬がj位の馬より上位の場合
                        if final_rankings[i] < final_rankings[j]:
                            # i位の馬のスコアがj位の馬のスコアより高くなるべき
                            loss = torch.clamp(self.margin - (final_scores[i] - final_scores[j]), min=0)
                            batch_loss += loss
                            num_pairs += 1
            
            if num_pairs > 0:
                batch_loss /= num_pairs
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


class CombinedRankingLoss(nn.Module):
    """
    複数の損失関数を組み合わせた損失
    
    Args:
        losses: 損失関数のリストと重み [(loss_fn, weight), ...]
        reduction: 'mean', 'sum', 'none'
    """
    
    def __init__(self, losses, reduction: str = 'mean'):
        super().__init__()
        self.losses = nn.ModuleList([loss_fn for loss_fn, _ in losses])
        self.weights = [weight for _, weight in losses]
        self.reduction = reduction
    
    def forward(self, scores: torch.Tensor, rankings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            loss = loss_fn(scores, rankings, mask)
            total_loss += weight * loss
        
        return total_loss


class RankNetLoss(nn.Module):
    """
    RankNet損失 - ニューラル情報検索で使用される損失関数
    
    Args:
        temperature: シグモイドの温度パラメータ
        top_k: 上位k位までのペアを考慮
        reduction: 'mean', 'sum', 'none'
    """
    
    def __init__(self, temperature: float = 1.0, top_k: Optional[int] = None, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.top_k = top_k
        self.reduction = reduction
    
    def forward(self, scores: torch.Tensor, rankings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_horses = scores.shape
        scaled_scores = scores / self.temperature
        
        losses = []
        
        for batch_idx in range(batch_size):
            batch_scores = scaled_scores[batch_idx]
            batch_rankings = rankings[batch_idx]
            batch_mask = mask[batch_idx]
            
            valid_horses = torch.where(batch_mask)[0]
            if len(valid_horses) == 0:
                continue
                
            valid_scores = batch_scores[valid_horses]
            valid_rankings = batch_rankings[valid_horses]
            
            valid_rank_mask = valid_rankings >= 0
            if not valid_rank_mask.any():
                continue
                
            final_scores = valid_scores[valid_rank_mask]
            final_rankings = valid_rankings[valid_rank_mask]
            
            if len(final_scores) <= 1:
                continue
            
            # top_kが指定されている場合
            if self.top_k is not None:
                top_k_mask = final_rankings < self.top_k
                if not top_k_mask.any():
                    continue
                final_scores = final_scores[top_k_mask]
                final_rankings = final_rankings[top_k_mask]
            
            # RankNet損失を計算
            batch_loss = 0.0
            num_pairs = 0
            
            for i in range(len(final_scores)):
                for j in range(len(final_scores)):
                    if i != j and final_rankings[i] != final_rankings[j]:
                        # 順位の差を正規化したラベル
                        if final_rankings[i] < final_rankings[j]:
                            # i位の馬がj位の馬より上位
                            p_ij = 1.0
                        else:
                            # j位の馬がi位の馬より上位
                            p_ij = 0.0
                        
                        # 予測確率
                        s_ij = torch.sigmoid(final_scores[i] - final_scores[j])
                        
                        # クロスエントロピー損失
                        loss = -p_ij * torch.log(s_ij + 1e-8) - (1 - p_ij) * torch.log(1 - s_ij + 1e-8)
                        batch_loss += loss
                        num_pairs += 1
            
            if num_pairs > 0:
                batch_loss /= num_pairs
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