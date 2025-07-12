import torch
import torch.nn as nn
from typing import Dict
import itertools

class PluckettLuceKeibaBetting(nn.Module):
    def __init__(self, num_horses: int = 18):
        super().__init__()
        self.num_horses = num_horses
        
        # 1～3着の全順列組み合わせを生成
        self.all_permutations = list(itertools.permutations(range(num_horses), 3))
        self.num_permutations = len(self.all_permutations)
        self.register_buffer('perm_indices', torch.tensor(self.all_permutations, dtype=torch.long))
        
        # 各馬券種のマッピングインデックスを準備
        self._prepare_mapping_indices()
    
    def _prepare_mapping_indices(self):
        """各馬券種のマッピング用インデックスを準備"""
        
        # 単勝: 1着馬のインデックス
        tansho_indices = torch.zeros(self.num_horses, self.num_permutations, dtype=torch.bool)
        for horse in range(self.num_horses):
            for perm_idx, (first, _, _) in enumerate(self.all_permutations):
                if first == horse:
                    tansho_indices[horse, perm_idx] = True
        self.register_buffer('tansho_indices', tansho_indices)
        
        # 複勝: 1～3着に入る馬のインデックス（8頭以上の場合）
        fukusho_indices_3place = torch.zeros(self.num_horses, self.num_permutations, dtype=torch.bool)
        for horse in range(self.num_horses):
            for perm_idx, (first, second, third) in enumerate(self.all_permutations):
                if horse in [first, second, third]:
                    fukusho_indices_3place[horse, perm_idx] = True
        self.register_buffer('fukusho_indices_3place', fukusho_indices_3place)
        
        # 複勝: 1～2着に入る馬のインデックス（7頭以下の場合）
        fukusho_indices_2place = torch.zeros(self.num_horses, self.num_permutations, dtype=torch.bool)
        for horse in range(self.num_horses):
            for perm_idx, (first, second, _) in enumerate(self.all_permutations):
                if horse in [first, second]:
                    fukusho_indices_2place[horse, perm_idx] = True
        self.register_buffer('fukusho_indices_2place', fukusho_indices_2place)
        
        # 馬連: 1～2着の組み合わせ（順序無関係）
        umaren_combinations = list(itertools.combinations(range(self.num_horses), 2))
        self.umaren_combinations = umaren_combinations
        umaren_indices = torch.zeros(len(umaren_combinations), self.num_permutations, dtype=torch.bool)
        for comb_idx, (horse1, horse2) in enumerate(umaren_combinations):
            for perm_idx, (first, second, _) in enumerate(self.all_permutations):
                if {first, second} == {horse1, horse2}:
                    umaren_indices[comb_idx, perm_idx] = True
        self.register_buffer('umaren_indices', umaren_indices)
        
        # ワイド: 1～3着のうち2頭の組み合わせ（順序無関係）
        wide_combinations = list(itertools.combinations(range(self.num_horses), 2))
        self.wide_combinations = wide_combinations
        wide_indices = torch.zeros(len(wide_combinations), self.num_permutations, dtype=torch.bool)
        for comb_idx, (horse1, horse2) in enumerate(wide_combinations):
            for perm_idx, (first, second, third) in enumerate(self.all_permutations):
                positions = [first, second, third]
                if horse1 in positions and horse2 in positions:
                    wide_indices[comb_idx, perm_idx] = True
        self.register_buffer('wide_indices', wide_indices)
        
        # 馬単: 1～2着の順列
        umatan_permutations = list(itertools.permutations(range(self.num_horses), 2))
        self.umatan_permutations = umatan_permutations
        umatan_indices = torch.zeros(len(umatan_permutations), self.num_permutations, dtype=torch.bool)
        for perm_idx_small, (horse1, horse2) in enumerate(umatan_permutations):
            for perm_idx, (first, second, _) in enumerate(self.all_permutations):
                if first == horse1 and second == horse2:
                    umatan_indices[perm_idx_small, perm_idx] = True
        self.register_buffer('umatan_indices', umatan_indices)
        
        # 三連複: 1～3着の組み合わせ（順序無関係）
        sanrenpuku_combinations = list(itertools.combinations(range(self.num_horses), 3))
        self.sanrenpuku_combinations = sanrenpuku_combinations
        sanrenpuku_indices = torch.zeros(len(sanrenpuku_combinations), self.num_permutations, dtype=torch.bool)
        for comb_idx, (horse1, horse2, horse3) in enumerate(sanrenpuku_combinations):
            for perm_idx, (first, second, third) in enumerate(self.all_permutations):
                if {first, second, third} == {horse1, horse2, horse3}:
                    sanrenpuku_indices[comb_idx, perm_idx] = True
        self.register_buffer('sanrenpuku_indices', sanrenpuku_indices)
        
        # 三連単: 1～3着の順列
        sanrentan_permutations = list(itertools.permutations(range(self.num_horses), 3))
        self.sanrentan_permutations = sanrentan_permutations
        sanrentan_indices = torch.zeros(len(sanrentan_permutations), self.num_permutations, dtype=torch.bool)
        for perm_idx_small, (horse1, horse2, horse3) in enumerate(sanrentan_permutations):
            for perm_idx, (first, second, third) in enumerate(self.all_permutations):
                if first == horse1 and second == horse2 and third == horse3:
                    sanrentan_indices[perm_idx_small, perm_idx] = True
        self.register_buffer('sanrentan_indices', sanrentan_indices)
    
    def forward(self, scores: torch.Tensor, num_horses_running: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            scores: (batch_size, num_horses) - 各馬の強さスコア
            num_horses_running: (batch_size,) - 各レースの出走頭数
        
        Returns:
            Dict[str, torch.Tensor] - 各馬券種の確率
        """
        # 1～3着の全順列の確率を計算
        perm_probs = self._calculate_permutation_probabilities(scores)  # (batch_size, num_permutations)
        
        # 各馬券種の確率を計算（forループを除去してテンソル演算で一括処理）
        probabilities = {}
        
        # 単勝: (batch_size, num_horses)
        # perm_probs: (batch_size, num_permutations), tansho_indices: (num_horses, num_permutations)
        # -> (batch_size, num_horses)
        probabilities['tansho'] = torch.matmul(perm_probs, self.tansho_indices.float().T)
        
        # 複勝: 出走頭数に応じて1～2着または1～3着を適用
        batch_size = scores.shape[0]
        fukusho_probs = torch.zeros(batch_size, self.num_horses, device=scores.device, dtype=scores.dtype)
        
        # 7頭以下のレース
        mask_7_or_less = num_horses_running <= 7
        if mask_7_or_less.any():
            fukusho_2place = torch.matmul(perm_probs[mask_7_or_less], self.fukusho_indices_2place.float().T)
            fukusho_probs[mask_7_or_less] = fukusho_2place
        
        # 8頭以上のレース
        mask_8_or_more = num_horses_running >= 8
        if mask_8_or_more.any():
            fukusho_3place = torch.matmul(perm_probs[mask_8_or_more], self.fukusho_indices_3place.float().T)
            fukusho_probs[mask_8_or_more] = fukusho_3place
        
        probabilities['fukusho'] = fukusho_probs
        
        # 馬連: (batch_size, num_combinations)
        probabilities['umaren'] = torch.matmul(perm_probs, self.umaren_indices.float().T)
        
        # ワイド: (batch_size, num_combinations)
        probabilities['wide'] = torch.matmul(perm_probs, self.wide_indices.float().T)
        
        # 馬単: (batch_size, num_permutations)
        probabilities['umatan'] = torch.matmul(perm_probs, self.umatan_indices.float().T)
        
        # 三連複: (batch_size, num_combinations)
        probabilities['sanrenpuku'] = torch.matmul(perm_probs, self.sanrenpuku_indices.float().T)
        
        # 三連単: (batch_size, num_permutations)
        probabilities['sanrentan'] = torch.matmul(perm_probs, self.sanrentan_indices.float().T)
        
        return probabilities
        
    def _calculate_permutation_probabilities(self, scores: torch.Tensor) -> torch.Tensor:
        """1～3着の全順列の確率を計算"""

        # 数値安定性のためのクリッピング 
        scores = torch.clamp(scores, min=-10, max=10)
        exp_scores = scores.exp()

        # 各順列の1着、2着、3着のインデックス
        first_indices = self.perm_indices[:, 0]   # (num_permutations,)
        second_indices = self.perm_indices[:, 1]  # (num_permutations,)
        third_indices = self.perm_indices[:, 2]   # (num_permutations,)
        
        # 1着の確率を計算
        first_scores = exp_scores[:, first_indices]  # (batch_size, num_permutations)
        first_denominator = exp_scores.sum(dim=1, keepdim=True)
        first_probs = first_scores / first_denominator

        # 2着の確率を計算（1着馬を除外）
        second_scores = exp_scores[:, second_indices]
        first_horse_scores = exp_scores[:, first_indices]
        second_denominator = first_denominator - first_horse_scores
        second_probs = second_scores / second_denominator

        # 3着の確率を計算（1着、2着馬を除外）
        third_scores = exp_scores[:, third_indices]
        second_horse_scores = exp_scores[:, second_indices]
        third_denominator = second_denominator - second_horse_scores
        third_probs = third_scores / third_denominator

        # 全体の順列確率（1着 × 2着 × 3着）: (batch_size, num_permutations)
        perm_probs = first_probs * second_probs * third_probs

        return perm_probs

# 使用例
if __name__ == "__main__":
    # モデルの初期化
    model = PluckettLuceKeibaBetting(num_horses=18)
    
    # サンプル入力
    batch_size = 2
    scores = torch.randn(batch_size, 18)  # ランダムなスコア
    num_horses_running = torch.tensor([7, 12], dtype=torch.long)  # 7頭と12頭のレース
    
    # 確率計算
    probabilities = model(scores, num_horses_running)
    
    # 結果の確認
    for bet_type, probs in probabilities.items():
        print(f"{bet_type}: {probs.shape}")
        print(f"  確率の合計: {torch.sum(probs, dim=1)}")  # 各バッチで合計が1になることを確認
