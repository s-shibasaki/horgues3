import numpy as np
from itertools import permutations, combinations

def calculate_betting_probabilities(horse_strengths, mask=None, temperature=1.0):
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
    
    # 温度パラメータでスケール
    scaled_strengths = masked_strengths / temperature
    
    # 各馬券タイプの確率を格納する辞書
    probabilities = {}
    
    # 指数化した強さスコア
    exp_strengths = np.exp(scaled_strengths)
    
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


def get_betting_combinations(max_horses=18):
    """各馬券種の組み合わせインデックスを返す"""
    combinations = {}
    
    # 単勝・複勝（馬番のみ）
    combinations['tansho'] = list(range(max_horses))
    combinations['fukusho'] = list(range(max_horses))
    
    # 馬連・ワイド（2頭の組み合わせ、順不同）
    umaren_pairs = []
    for i in range(max_horses):
        for j in range(i+1, max_horses):
            umaren_pairs.append((i, j))
    combinations['umaren'] = umaren_pairs
    combinations['wide'] = umaren_pairs
    
    # 馬単（2頭の組み合わせ、順序あり）
    umatan_pairs = []
    for i in range(max_horses):
        for j in range(max_horses):
            if i != j:
                umatan_pairs.append((i, j))
    combinations['umatan'] = umatan_pairs
    
    # 3連複（3頭の組み合わせ、順不同）
    sanrenfuku_triplets = []
    for i in range(max_horses):
        for j in range(i+1, max_horses):
            for k in range(j+1, max_horses):
                sanrenfuku_triplets.append((i, j, k))
    combinations['sanrenfuku'] = sanrenfuku_triplets
    
    # 3連単（3頭の組み合わせ、順序あり）
    sanrentan_triplets = []
    for i in range(max_horses):
        for j in range(max_horses):
            if j == i:
                continue
            for k in range(max_horses):
                if k == i or k == j:
                    continue
                sanrentan_triplets.append((i, j, k))
    combinations['sanrentan'] = sanrentan_triplets
    
    return combinations


def extract_winning_tickets(rankings, mask=None, max_horses=18):
    """
    着順データから各馬券種の的中馬券を抽出する
    
    Args:
        rankings: 着順データ (num_races, num_horses) - 1-indexed、0は無効
        mask: 有効な馬のマスク (num_races, num_horses) - 1=有効、0=無効/パディング
        max_horses: 最大馬数
        
    Returns:
        winning_tickets: 各馬券種の的中馬券を格納した辞書
            各値は (num_races, num_combinations) のboolean配列
    """
    num_races, num_horses = rankings.shape
    
    # マスクが指定されていない場合はすべて有効とみなす
    if mask is None:
        mask = np.ones_like(rankings, dtype=bool)
    else:
        mask = mask.astype(bool)
    
    # 組み合わせ情報を取得
    combinations = get_betting_combinations(max_horses)
    
    # 結果を格納する辞書
    winning_tickets = {}
    
    for race_idx in range(num_races):
        race_rankings = rankings[race_idx]
        race_mask = mask[race_idx]
        
        # 有効な馬のみを抽出
        valid_horses = np.where(race_mask)[0]
        if len(valid_horses) == 0:
            continue
            
        valid_rankings = race_rankings[valid_horses]
        
        # 無効な着順を除外
        valid_rank_mask = valid_rankings > 0
        if not valid_rank_mask.any():
            continue
            
        final_horses = valid_horses[valid_rank_mask]
        final_rankings = valid_rankings[valid_rank_mask]
        
        # 着順から順位を特定（0-indexed）
        horse_to_rank = {}
        for horse_idx, rank in zip(final_horses, final_rankings):
            horse_to_rank[horse_idx] = rank - 1  # 0-indexedに変換
        
        # 各着順の馬を特定
        rank_to_horse = {}
        for horse_idx, rank in horse_to_rank.items():
            rank_to_horse[rank] = horse_idx
        
        # レースの的中馬券を初期化（初回のみ）
        if race_idx == 0:
            for bet_type in ['tansho', 'fukusho', 'umaren', 'umatan', 'wide', 'sanrenfuku', 'sanrentan']:
                num_combinations = len(combinations[bet_type])
                winning_tickets[bet_type] = np.zeros((num_races, num_combinations), dtype=bool)
    
    # 各レースの的中馬券を計算
    for race_idx in range(num_races):
        race_rankings = rankings[race_idx]
        race_mask = mask[race_idx]
        
        # 有効な馬のみを抽出
        valid_horses = np.where(race_mask)[0]
        if len(valid_horses) == 0:
            continue
            
        valid_rankings = race_rankings[valid_horses]
        valid_rank_mask = valid_rankings > 0
        if not valid_rank_mask.any():
            continue
            
        final_horses = valid_horses[valid_rank_mask]
        final_rankings = valid_rankings[valid_rank_mask]
        
        # 着順マッピングを作成
        horse_to_rank = {}
        for horse_idx, rank in zip(final_horses, final_rankings):
            horse_to_rank[horse_idx] = rank - 1
        
        rank_to_horse = {}
        for horse_idx, rank in horse_to_rank.items():
            rank_to_horse[rank] = horse_idx
        
        # 単勝（1位の馬）
        if 0 in rank_to_horse:
            winner = rank_to_horse[0]
            winning_tickets['tansho'][race_idx, winner] = True
        
        # 複勝（1〜3位の馬）
        for rank in range(min(3, len(rank_to_horse))):
            if rank in rank_to_horse:
                horse = rank_to_horse[rank]
                winning_tickets['fukusho'][race_idx, horse] = True
        
        # 2頭系馬券（馬連・馬単・ワイド）
        if len(rank_to_horse) >= 2:
            # 1-2位の組み合わせ
            if 0 in rank_to_horse and 1 in rank_to_horse:
                first = rank_to_horse[0]
                second = rank_to_horse[1]
                
                # 馬連（順不同）
                combo = tuple(sorted([first, second]))
                combo_idx = combinations['umaren'].index(combo)
                winning_tickets['umaren'][race_idx, combo_idx] = True
                
                # 馬単（順序あり）
                combo = (first, second)
                combo_idx = combinations['umatan'].index(combo)
                winning_tickets['umatan'][race_idx, combo_idx] = True
            
            # ワイド（1-3位の2頭の組み合わせ）
            top3_horses = []
            for rank in range(min(3, len(rank_to_horse))):
                if rank in rank_to_horse:
                    top3_horses.append(rank_to_horse[rank])
            
            if len(top3_horses) >= 2:
                for i in range(len(top3_horses)):
                    for j in range(i + 1, len(top3_horses)):
                        combo = tuple(sorted([top3_horses[i], top3_horses[j]]))
                        combo_idx = combinations['wide'].index(combo)
                        winning_tickets['wide'][race_idx, combo_idx] = True
        
        # 3頭系馬券（3連複・3連単）
        if len(rank_to_horse) >= 3:
            if 0 in rank_to_horse and 1 in rank_to_horse and 2 in rank_to_horse:
                first = rank_to_horse[0]
                second = rank_to_horse[1]
                third = rank_to_horse[2]
                
                # 3連複（順不同）
                combo = tuple(sorted([first, second, third]))
                combo_idx = combinations['sanrenfuku'].index(combo)
                winning_tickets['sanrenfuku'][race_idx, combo_idx] = True
                
                # 3連単（順序あり）
                combo = (first, second, third)
                combo_idx = combinations['sanrentan'].index(combo)
                winning_tickets['sanrentan'][race_idx, combo_idx] = True
    
    return winning_tickets


def format_betting_results(race_ids, probabilities, masks=None):
    """
    馬券確率の結果をフォーマットして出力（複数レース対応）
    
    Args:
        race_ids: レースIDのリスト (num_races,)
        probabilities: 各種馬券の確率を格納した辞書 (num_races, num_combinations)
        masks: 有効な馬のマスク (num_races, num_horses)
        
    Returns:
        results: フォーマットされた結果の文字列
    """
    num_races = len(race_ids)
    combinations = get_betting_combinations()
    
    all_results = []
    
    for race_idx in range(num_races):
        race_id = race_ids[race_idx]
        
        # 有効馬数を計算
        if masks is not None:
            num_horses = int(masks[race_idx].sum())
        else:
            num_horses = 18  # デフォルト値
        
        results = []
        results.append(f"\n=== Race ID: {race_id} (有効馬数: {num_horses}) ===")

        # 各馬券種の上位確率を表示
        for bet_type in ['tansho', 'fukusho', 'umaren', 'umatan', 'wide', 'sanrenfuku', 'sanrentan']:
            if bet_type not in probabilities:
                continue
                
            probs = probabilities[bet_type][race_idx]  # 該当レースの確率を取得
            combos = combinations[bet_type]
            
            # 確率でソート（降順）
            sorted_indices = np.argsort(probs)[::-1]
            
            # 出力枚数を決定
            num_output = min(10, len(sorted_indices))
            
            # 上位の馬券を表示
            bet_results = []
            for i in range(num_output):
                idx = sorted_indices[i]
                prob = probs[idx]
                
                # 確率が0に近い場合はスキップ
                if prob < 1e-8:
                    continue
                
                if bet_type in ['tansho', 'fukusho']:
                    # 単勝・複勝：馬番のみ
                    horse_num = combos[idx] + 1  # 0-indexedから1-indexedに変換
                    
                    # 有効な馬かチェック
                    if masks is not None and not masks[race_idx][combos[idx]]:
                        continue
                        
                    bet_results.append(f"{horse_num}番({prob:.4f})")
                    
                elif bet_type in ['umaren', 'umatan', 'wide']:
                    # 2頭の組み合わせ
                    horse1, horse2 = combos[idx]
                    horse1_num, horse2_num = horse1 + 1, horse2 + 1
                    
                    # 有効な馬かチェック
                    if masks is not None and (not masks[race_idx][horse1] or not masks[race_idx][horse2]):
                        continue
                    
                    if bet_type == 'umatan':
                        bet_results.append(f"{horse1_num}-{horse2_num}({prob:.4f})")
                    else:
                        bet_results.append(f"{horse1_num}-{horse2_num}({prob:.4f})")
                        
                else:
                    # 3頭の組み合わせ
                    horse1, horse2, horse3 = combos[idx]
                    horse1_num, horse2_num, horse3_num = horse1 + 1, horse2 + 1, horse3 + 1
                    
                    # 有効な馬かチェック
                    if masks is not None and (not masks[race_idx][horse1] or not masks[race_idx][horse2] or not masks[race_idx][horse3]):
                        continue
                    
                    if bet_type == 'sanrentan':
                        bet_results.append(f"{horse1_num}-{horse2_num}-{horse3_num}({prob:.4f})")
                    else:
                        bet_results.append(f"{horse1_num}-{horse2_num}-{horse3_num}({prob:.4f})")
            
            if bet_results:  # 有効な結果がある場合のみ表示
                results.append(f"{bet_type.upper()}: {' '.join(bet_results)}")
        
        all_results.extend(results)
    
    return '\n'.join(all_results)