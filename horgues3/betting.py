import numpy as np
from itertools import permutations, combinations

def calculate_betting_probabilities(horse_strengths, mask=None):
    """
    馬の強さスコアから各種馬券の確率を計算する関数
    
    Args:
        horse_strengths: 馬の強さスコア (num_races, num_horses)
        mask: 有効な馬のマスク (num_races, num_horses)、1=有効、0=無効/パディング
        
    Returns:
        probabilities: 各種馬券の確率を格納した辞書
    """
    num_races, num_horses = horse_strengths.shape
    
    # マスクが指定されていない場合はすべて有効とみなす
    if mask is None:
        mask = np.ones_like(horse_strengths, dtype=bool)
    else:
        mask = mask.astype(bool)
    
    # 無効な馬のスコアを非常に小さい値に設定
    masked_strengths = np.where(mask, horse_strengths, -np.inf)
    
    # 各馬券タイプの確率を格納する辞書
    probabilities = {}
    
    # 指数化した強さスコア
    exp_strengths = np.exp(masked_strengths)
    
    # ==== 単勝（1位の馬を当てる）====
    probabilities['tansho'] = exp_strengths / np.sum(exp_strengths, axis=1, keepdims=True)
    
    # ==== 順位ごとの確率行列を効率的に計算 ====
    # P(1位,2位,3位,...) の同時分布を計算
    
    # 1位の確率 (num_races, num_horses)
    first_probs = probabilities['tansho']
    
    # 2位の条件付き確率と結合確率を計算
    second_cond_probs = np.zeros((num_races, num_horses, num_horses))
    first_second_probs = np.zeros((num_races, num_horses, num_horses))
    
    # 各1位候補馬について2位の確率を計算
    for i in range(num_horses):
        # i番目の馬を除いたマスクとスコア
        remaining_mask = mask.copy()
        remaining_mask[:, i] = False
        remaining_strengths = np.where(remaining_mask, horse_strengths, -np.inf)
        remaining_exp = np.exp(remaining_strengths)
        remaining_sum = np.sum(remaining_exp, axis=1, keepdims=True)
        
        # 2位の条件付き確率 P(2位=j | 1位=i)
        second_cond_probs[:, i] = remaining_exp / np.maximum(remaining_sum, 1e-10)
        
        # 結合確率 P(1位=i, 2位=j)
        for j in range(num_horses):
            if i != j:
                first_second_probs[:, i, j] = first_probs[:, i] * second_cond_probs[:, i, j]
    
    # 3位の条件付き確率と結合確率を計算
    third_cond_probs = np.zeros((num_races, num_horses, num_horses, num_horses))
    first_second_third_probs = np.zeros((num_races, num_horses, num_horses, num_horses))
    
    # 順序付き組み合わせに対応するインデックス配列を作成
    ordered_triplets = np.array(list(permutations(range(num_horses), 3)))
    num_triplets = len(ordered_triplets)
    
    # 各1位,2位候補馬について3位の確率を計算
    for i in range(num_horses):
        for j in range(num_horses):
            if i == j:
                continue
                
            # i番目とj番目の馬を除いたマスクとスコア
            remaining_mask = mask.copy()
            remaining_mask[:, i] = False
            remaining_mask[:, j] = False
            remaining_strengths = np.where(remaining_mask, horse_strengths, -np.inf)
            remaining_exp = np.exp(remaining_strengths)
            remaining_sum = np.sum(remaining_exp, axis=1, keepdims=True)
            
            # 3位の条件付き確率 P(3位=k | 1位=i, 2位=j)
            third_cond_probs[:, i, j] = remaining_exp / np.maximum(remaining_sum, 1e-10)
            
            # 結合確率 P(1位=i, 2位=j, 3位=k)
            for k in range(num_horses):
                if k != i and k != j:
                    first_second_third_probs[:, i, j, k] = first_second_probs[:, i, j] * third_cond_probs[:, i, j, k]
    
    # ==== 各馬券タイプの確率を計算 ====
    
    # 複勝（1〜3位の馬を当てる）
    place_probs = np.zeros((num_races, num_horses))
    for i in range(num_horses):
        # i番目の馬が1位の確率
        place_probs[:, i] += first_probs[:, i]
        
        # i番目の馬が2位の確率
        for j in range(num_horses):
            if j != i:
                place_probs[:, i] += first_second_probs[:, j, i]
                
        # i番目の馬が3位の確率
        for j in range(num_horses):
            for k in range(num_horses):
                if i != j and i != k and j != k:
                    place_probs[:, i] += first_second_third_probs[:, j, k, i]
    
    probabilities['fukusho'] = place_probs
    
    # 馬連（1位と2位の組み合わせを順不同で当てる）
    umaren_probs = np.zeros((num_races, num_horses * (num_horses - 1) // 2))
    umaren_idx = 0
    for i in range(num_horses):
        for j in range(i+1, num_horses):
            umaren_probs[:, umaren_idx] = (first_second_probs[:, i, j] + first_second_probs[:, j, i])
            umaren_idx += 1
    
    probabilities['umaren'] = umaren_probs
    
    # 馬単（1位と2位の組み合わせを順序通りに当てる）
    umatan_probs = np.zeros((num_races, num_horses * (num_horses - 1)))
    umatan_idx = 0
    for i in range(num_horses):
        for j in range(num_horses):
            if i != j:
                umatan_probs[:, umatan_idx] = first_second_probs[:, i, j]
                umatan_idx += 1
    
    probabilities['umatan'] = umatan_probs
    
    # ワイド（1〜3位の2頭の組み合わせを順不同で当てる）
    wide_probs = np.zeros((num_races, num_horses * (num_horses - 1) // 2))
    wide_idx = 0
    for i in range(num_horses):
        for j in range(i+1, num_horses):
            # (i,j)が1-2位、1-3位、2-3位のいずれかのケース
            prob = np.zeros(num_races)
            
            # 1-2位のケース
            prob += first_second_probs[:, i, j] + first_second_probs[:, j, i]
            
            # 1-3位のケース
            for k in range(num_horses):
                if k != i and k != j:
                    prob += first_second_third_probs[:, i, k, j] + first_second_third_probs[:, j, k, i]
            
            # 2-3位のケース
            for k in range(num_horses):
                if k != i and k != j:
                    prob += first_second_third_probs[:, k, i, j] + first_second_third_probs[:, k, j, i]
                    
            wide_probs[:, wide_idx] = prob
            wide_idx += 1
    
    probabilities['wide'] = wide_probs
    
    # 3連複（1〜3位の3頭の組み合わせを順不同で当てる）
    sanrenfuku_probs = np.zeros((num_races, num_horses * (num_horses - 1) * (num_horses - 2) // 6))
    sanrenfuku_idx = 0
    for i in range(num_horses):
        for j in range(i+1, num_horses):
            for k in range(j+1, num_horses):
                # (i,j,k)の6通りの順列すべての確率を合計
                prob = np.zeros(num_races)
                prob += first_second_third_probs[:, i, j, k]
                prob += first_second_third_probs[:, i, k, j]
                prob += first_second_third_probs[:, j, i, k]
                prob += first_second_third_probs[:, j, k, i]
                prob += first_second_third_probs[:, k, i, j]
                prob += first_second_third_probs[:, k, j, i]
                
                sanrenfuku_probs[:, sanrenfuku_idx] = prob
                sanrenfuku_idx += 1
    
    probabilities['sanrenfuku'] = sanrenfuku_probs
    
    # 3連単（1〜3位の3頭の組み合わせを順序通りに当てる）
    sanrentan_probs = np.zeros((num_races, num_horses * (num_horses - 1) * (num_horses - 2)))
    sanrentan_idx = 0
    for i in range(num_horses):
        for j in range(num_horses):
            if j == i:
                continue
            for k in range(num_horses):
                if k == i or k == j:
                    continue
                
                sanrentan_probs[:, sanrentan_idx] = first_second_third_probs[:, i, j, k]
                sanrentan_idx += 1
    
    probabilities['sanrentan'] = sanrentan_probs
    
    return probabilities