import tkinter as tk
from tkinter import ttk, font, messagebox
import random
import threading
from collections import defaultdict
import multiprocessing # 引入多进程模块
import sys # 引入 sys 模块，用于处理打包后的环境
import math # 引入 math 模块以使用缓动函数
import webbrowser # 新增: 用于打开网页链接
# from hand_strength_data import HAND_TIERS # 导入起手牌强度数据 (假设文件在同一目录)
# from strength_chart_window import StrengthChartWindow # 导入起手牌强度图表窗口 (假设文件在同一目录)
import os

# 注意：由于我无法访问你的本地 hand_strength_data 和 strength_chart_window 文件
# 为了让这段代码能直接运行，我将暂时注释掉这两个导入。
# 如果你在本地运行，请取消下面两行的注释，并确保文件存在。
try:
    from hand_strength_data import HAND_TIERS
    from strength_chart_window import StrengthChartWindow
except ImportError:
    HAND_TIERS = {} # 占位符
    StrengthChartWindow = None # 占位符
    print("提示: 未找到辅助模块 (hand_strength_data/strength_chart_window)，相关功能将禁用。")

# 建议安装 'treys' 库: pip install treys
try:
    from treys import Card, Evaluator, Deck
except ImportError:
    print("错误: 未找到 'treys' 库。请使用 'pip install treys' 命令进行安装。")
    exit()

# ##################################################################
# ############### 新增的并行计算工作函数 ###########################
# ##################################################################
# 定义常量
RANK_CLASS_TO_STRING = {
    1: "同花顺 (Straight Flush)", 
    2: "四条 (Four of a Kind)", 
    3: "葫芦 (Full House)",
    4: "同花 (Flush)", 
    5: "顺子 (Straight)", 
    6: "三条 (Three of a Kind)",
    7: "两对 (Two Pair)", 
    8: "一对 (One Pair)", 
    9: "高牌 (High Card)"
}

# --- v8/v9 核心: 模块级牌力缓存 + 全局进程池 ---
_RANK_CLASS_CACHE = [0] * 7463
_CACHE_INITIALIZED = False
_GLOBAL_POOL = None  # 全局进程池，复用以减少启动开销
_POOL_SIZE = None    # 记录进程数

# 【核心优化】：进程级 Evaluator 缓存
# 避免在每个任务块中重复初始化 Evaluator (这非常耗时)
_WORKER_EVALUATOR = None

def _get_worker_evaluator():
    """获取当前进程的 Evaluator 单例，避免重复初始化 LookupTable。"""
    global _WORKER_EVALUATOR
    if _WORKER_EVALUATOR is None:
        _WORKER_EVALUATOR = Evaluator()
    return _WORKER_EVALUATOR

def _initialize_rank_cache(evaluator):
    """初始化牌力缓存表，将函数调用转为 O(1) 数组索引。"""
    global _CACHE_INITIALIZED
    # 修复BUG：检查索引 1 (Straight Flush) 是否已填充，而不是 0
    # Treys 的 rank 是 1-7462，索引 0 永远是 0
    if _CACHE_INITIALIZED and _RANK_CLASS_CACHE[1] != 0:
        return
        
    get_rc = evaluator.get_rank_class
    try:
        # 修复BUG：从 1 开始遍历，避免调用 get_rc(0) 导致崩溃
        for r in range(1, 7463):
            _RANK_CLASS_CACHE[r] = get_rc(r)
            
        # 只有当数据真正写入成功后（检查索引 1），才标记为已初始化
        if _RANK_CLASS_CACHE[1] != 0:
            _CACHE_INITIALIZED = True
    except Exception as e:
        print(f"Cache init warning: {e}")
        # 初始化失败不设置 _CACHE_INITIALIZED，以便下次重试

def _get_or_create_pool(num_cores):
    """获取或创建全局进程池，避免重复创建的开销。"""
    global _GLOBAL_POOL, _POOL_SIZE
    if _GLOBAL_POOL is None or _POOL_SIZE != num_cores:
        # 如果有旧的进程池，先关闭
        if _GLOBAL_POOL is not None:
            _GLOBAL_POOL.close()
            _GLOBAL_POOL.join()
        # 这里的 None 表示使用 os.cpu_count()，或者你可以传入具体的 num_cores
        _GLOBAL_POOL = multiprocessing.Pool(processes=num_cores)
        _POOL_SIZE = num_cores
    return _GLOBAL_POOL

def _run_simulation_chunk(args: tuple[str, list[list[str]], str, list[list[str]], list[str], int, list[str], bool]) -> tuple[int, int, int, dict[str, int], int]:
    """
    执行扑克模拟任务块 (v9 Hybrid Enumeration Edition)。
    【保留优化 + 修复死锁版 + Evaluator 复用优化】
    """
    # 解包参数
    p1_type, p1_hands, p2_type, p2_hands, board, num_sims_chunk, master_deck_cards, calculate_p1_strength = args

    # 【核心优化点】：使用进程单例，而不是每次都创建新的 Evaluator()
    evaluator = _get_worker_evaluator()
    
    # 确保缓存已正确初始化
    if calculate_p1_strength:
        if not _CACHE_INITIALIZED or _RANK_CLASS_CACHE[1] == 0:
            _initialize_rank_cache(evaluator)
    
    # --- 局部变量缓存 (优化调用速度) ---
    _evaluate = evaluator.evaluate
    _sample = random.sample
    _choice = random.choice
    _repeat = lambda obj, times: (obj for _ in range(times))
    from itertools import combinations as _combinations

    _rank_cache = _RANK_CLASS_CACHE 
    
    # 安全查表函数
    def _get_rc_safe(score):
        rc = _rank_cache[score]
        if rc == 0:
            return evaluator.get_rank_class(score)
        return rc
    
    # --- 1. 基础数据构建 ---
    board_list: list[str] = list(board)
    board_len: int = len(board_list)
    cards_needed: int = 5 - board_len
    
    # 构建基础牌库
    board_set: set[str] = set(board_list)
    base_deck_set: set[str] = set(master_deck_cards) - board_set
    base_deck_list: list[str] = list(base_deck_set)
    base_deck_len: int = len(base_deck_list)
    
    # 结果统计容器
    p1_wins: int = 0
    p2_wins: int = 0
    p1_rank_counts_list: list[int] = [0] * 10 
    valid_sims: int = 0

    # --- 2. 预处理与手牌过滤 ---
    
    def filter_hands(hands: list[list[str]]) -> list[list[str]]:
        if not hands: return []
        return [h for h in hands if set(h).isdisjoint(board_set)]

    real_p1_hands = filter_hands(p1_hands) if p1_type != 'random' else None
    real_p2_hands = filter_hands(p2_hands) if p2_type != 'random' else None
    
    if (p1_type != 'random' and not real_p1_hands) or (p2_type != 'random' and not real_p2_hands):
        return 0, 0, 0, {}, 0

    p1_is_fixed: bool = (p1_type != 'random' and len(real_p1_hands) == 1)
    p2_is_fixed: bool = (p2_type != 'random' and len(real_p2_hands) == 1)

    # ==============================================================================
    # 路径 A: Fixed vs Fixed (确定性对决 - 混合枚举优化)
    # ==============================================================================
    if p1_is_fixed and p2_is_fixed:
        p1_hand = real_p1_hands[0]
        p2_hand = real_p2_hands[0]
        
        if not set(p1_hand).isdisjoint(set(p2_hand)):
            return 0, 0, 0, {}, 0
            
        # --- A1: River (无需发牌) ---
        if cards_needed == 0:
            p1_score = _evaluate(board_list, p1_hand)
            p2_score = _evaluate(board_list, p2_hand)
            
            if calculate_p1_strength:
                p1_rank_counts_list[_get_rc_safe(p1_score)] = num_sims_chunk
            
            if p1_score < p2_score: p1_wins = num_sims_chunk
            elif p2_score < p1_score: p2_wins = num_sims_chunk
            valid_sims = num_sims_chunk
            
        # --- A2: 需要发公共牌 (混合枚举核心逻辑) ---
        else:
            used_cards = set(p1_hand) | set(p2_hand)
            deck_rem = [c for c in base_deck_list if c not in used_cards]
            n_rem = len(deck_rem)
            
            try:
                possible_combos = math.comb(n_rem, cards_needed)
            except AttributeError: 
                def nCr(n, r):
                    f = math.factorial
                    return f(n) // f(r) // f(n-r)
                possible_combos = nCr(n_rem, cards_needed)

            # --- 算法分支判定 ---
            if possible_combos <= num_sims_chunk:
                # >>> 穷举模式 (Enumeration Mode) <<<
                combo_iter = _combinations(deck_rem, cards_needed)
                
                if calculate_p1_strength:
                    for board_ext in combo_iter:
                        run_board = board_list + list(board_ext)
                        p1_s = _evaluate(run_board, p1_hand)
                        p2_s = _evaluate(run_board, p2_hand)
                        
                        p1_rank_counts_list[_get_rc_safe(p1_s)] += 1
                        if p1_s < p2_s: p1_wins += 1
                        elif p2_s < p1_s: p2_wins += 1
                else:
                    for board_ext in combo_iter:
                        run_board = board_list + list(board_ext)
                        p1_s = _evaluate(run_board, p1_hand)
                        p2_s = _evaluate(run_board, p2_hand)
                        
                        if p1_s < p2_s: p1_wins += 1
                        elif p2_s < p1_s: p2_wins += 1
                
                valid_sims = possible_combos
            
            else:
                # >>> 模拟模式 (Simulation Mode) <<<
                loop_iter = _repeat(None, num_sims_chunk)
                
                if calculate_p1_strength:
                    for _ in loop_iter:
                        board_ext = _sample(deck_rem, cards_needed)
                        run_board = board_list + board_ext
                        p1_s = _evaluate(run_board, p1_hand)
                        p2_s = _evaluate(run_board, p2_hand)
                        p1_rank_counts_list[_get_rc_safe(p1_s)] += 1
                        if p1_s < p2_s: p1_wins += 1
                        elif p2_s < p1_s: p2_wins += 1
                else:
                    for _ in loop_iter:
                        board_ext = _sample(deck_rem, cards_needed)
                        run_board = board_list + board_ext
                        p1_s = _evaluate(run_board, p1_hand)
                        p2_s = _evaluate(run_board, p2_hand)
                        if p1_s < p2_s: p1_wins += 1
                        elif p2_s < p1_s: p2_wins += 1
                
                valid_sims = num_sims_chunk

    # ==============================================================================
    # 路径 B: Random vs Random (极速模式)
    # ==============================================================================
    elif p1_type == 'random' and p2_type == 'random':
        total_draw = 4 + cards_needed
        if base_deck_len >= total_draw:
            loop_iter = _repeat(None, num_sims_chunk)
            
            # --- B1: River ---
            if cards_needed == 0:
                if calculate_p1_strength:
                    for _ in loop_iter:
                        c1, c2, c3, c4 = _sample(base_deck_list, 4)
                        p1_s = _evaluate(board_list, [c1, c2])
                        p2_s = _evaluate(board_list, [c3, c4])
                        p1_rank_counts_list[_get_rc_safe(p1_s)] += 1
                        if p1_s < p2_s: p1_wins += 1
                        elif p2_s < p1_s: p2_wins += 1
                else:
                    for _ in loop_iter:
                        c1, c2, c3, c4 = _sample(base_deck_list, 4)
                        p1_s = _evaluate(board_list, [c1, c2])
                        p2_s = _evaluate(board_list, [c3, c4])
                        if p1_s < p2_s: p1_wins += 1
                        elif p2_s < p1_s: p2_wins += 1
            
            # --- B2: Preflop/Flop/Turn ---
            else:
                if board_len == 0:
                    if calculate_p1_strength:
                        for _ in loop_iter:
                            draw = _sample(base_deck_list, total_draw)
                            run_board = draw[4:] 
                            p1_s = _evaluate(run_board, draw[0:2])
                            p2_s = _evaluate(run_board, draw[2:4])
                            p1_rank_counts_list[_get_rc_safe(p1_s)] += 1
                            if p1_s < p2_s: p1_wins += 1
                            elif p2_s < p1_s: p2_wins += 1
                    else:
                        for _ in loop_iter:
                            draw = _sample(base_deck_list, total_draw)
                            run_board = draw[4:]
                            p1_s = _evaluate(run_board, draw[0:2])
                            p2_s = _evaluate(run_board, draw[2:4])
                            if p1_s < p2_s: p1_wins += 1
                            elif p2_s < p1_s: p2_wins += 1
                else:
                    if calculate_p1_strength:
                        for _ in loop_iter:
                            draw = _sample(base_deck_list, total_draw)
                            run_board = board_list + draw[4:]
                            p1_s = _evaluate(run_board, draw[0:2])
                            p2_s = _evaluate(run_board, draw[2:4])
                            p1_rank_counts_list[_get_rc_safe(p1_s)] += 1
                            if p1_s < p2_s: p1_wins += 1
                            elif p2_s < p1_s: p2_wins += 1
                    else:
                        for _ in loop_iter:
                            draw = _sample(base_deck_list, total_draw)
                            run_board = board_list + draw[4:]
                            p1_s = _evaluate(run_board, draw[0:2])
                            p2_s = _evaluate(run_board, draw[2:4])
                            if p1_s < p2_s: p1_wins += 1
                            elif p2_s < p1_s: p2_wins += 1
            
            valid_sims = num_sims_chunk

    # ==============================================================================
    # 路径 C: Fixed P1 vs Random P2
    # ==============================================================================
    elif p1_is_fixed and p2_type == 'random':
        p1_hand = real_p1_hands[0]
        p1_set = set(p1_hand)
        p2_deck_list = [c for c in base_deck_list if c not in p1_set]
        
        p1_static_score = -1
        p1_static_rank = 0
        if cards_needed == 0:
            p1_static_score = _evaluate(board_list, p1_hand)
            if calculate_p1_strength:
                p1_static_rank = _get_rc_safe(p1_static_score)
        
        loop_iter = _repeat(None, num_sims_chunk)
        
        if cards_needed == 0:
            for _ in loop_iter:
                p2_hand = _sample(p2_deck_list, 2)
                p2_s = _evaluate(board_list, p2_hand)
                if calculate_p1_strength: p1_rank_counts_list[p1_static_rank] += 1
                if p1_static_score < p2_s: p1_wins += 1
                elif p2_s < p1_static_score: p2_wins += 1
        else:
            total_draw = 2 + cards_needed
            if calculate_p1_strength:
                for _ in loop_iter:
                    draw = _sample(p2_deck_list, total_draw)
                    run_board = board_list + draw[2:]
                    p1_s = _evaluate(run_board, p1_hand)
                    p2_s = _evaluate(run_board, draw[0:2])
                    p1_rank_counts_list[_get_rc_safe(p1_s)] += 1
                    if p1_s < p2_s: p1_wins += 1
                    elif p2_s < p1_s: p2_wins += 1
            else:
                for _ in loop_iter:
                    draw = _sample(p2_deck_list, total_draw)
                    run_board = board_list + draw[2:]
                    p1_s = _evaluate(run_board, p1_hand)
                    p2_s = _evaluate(run_board, draw[0:2])
                    if p1_s < p2_s: p1_wins += 1
                    elif p2_s < p1_s: p2_wins += 1

        valid_sims = num_sims_chunk

    # ==============================================================================
    # 路径 D: Random P1 vs Fixed P2
    # ==============================================================================
    elif p1_type == 'random' and p2_is_fixed:
        p2_hand = real_p2_hands[0]
        p2_set = set(p2_hand)
        p1_deck_list = [c for c in base_deck_list if c not in p2_set]
        
        p2_static_score = -1
        if cards_needed == 0:
            p2_static_score = _evaluate(board_list, p2_hand)

        loop_iter = _repeat(None, num_sims_chunk)

        if cards_needed == 0:
            if calculate_p1_strength:
                for _ in loop_iter:
                    p1_hand = _sample(p1_deck_list, 2)
                    p1_s = _evaluate(board_list, p1_hand)
                    p1_rank_counts_list[_get_rc_safe(p1_s)] += 1
                    if p1_s < p2_static_score: p1_wins += 1
                    elif p2_static_score < p1_s: p2_wins += 1
            else:
                for _ in loop_iter:
                    p1_hand = _sample(p1_deck_list, 2)
                    p1_s = _evaluate(board_list, p1_hand)
                    if p1_s < p2_static_score: p1_wins += 1
                    elif p2_static_score < p1_s: p2_wins += 1
        else:
            total_draw = 2 + cards_needed
            if calculate_p1_strength:
                for _ in loop_iter:
                    draw = _sample(p1_deck_list, total_draw)
                    run_board = board_list + draw[2:]
                    p1_s = _evaluate(run_board, draw[0:2])
                    p2_s = _evaluate(run_board, p2_hand)
                    p1_rank_counts_list[_get_rc_safe(p1_s)] += 1
                    if p1_s < p2_s: p1_wins += 1
                    elif p2_s < p1_s: p2_wins += 1
            else:
                for _ in loop_iter:
                    draw = _sample(p1_deck_list, total_draw)
                    run_board = board_list + draw[2:]
                    p1_s = _evaluate(run_board, draw[0:2])
                    p2_s = _evaluate(run_board, p2_hand)
                    if p1_s < p2_s: p1_wins += 1
                    elif p2_s < p1_s: p2_wins += 1
        
        valid_sims = num_sims_chunk

    # ==============================================================================
    # 路径 E: Range vs Range
    # ==============================================================================
    else:
        card_to_mask = {card: (1 << i) for i, card in enumerate(base_deck_list)}
        
        # 预计算位掩码 (优化保留)
        def prepare_weighted_hands(hands: list[list[str]]) -> list[tuple[list[str], int]]:
            weighted = []
            for h in hands:
                mask = 0
                valid = True
                for c in h:
                    if c not in card_to_mask:
                        valid = False; break
                    mask |= card_to_mask[c]
                if valid: weighted.append((h, mask))
            return weighted

        p1_data = prepare_weighted_hands(real_p1_hands) if p1_type != 'random' else None
        p2_data = prepare_weighted_hands(real_p2_hands) if p2_type != 'random' else None
        
        loop_iter = _repeat(None, num_sims_chunk)

        # --- E1: Range vs Range ---
        if p1_type != 'random' and p2_type != 'random':
            if calculate_p1_strength:
                for _ in loop_iter:
                    valid_matchup = False
                    for _ in range(20): # 尝试 20 次选取 P1
                        p1_h, p1_m = _choice(p1_data)
                        
                        # 尝试为这个 P1 找一个合法的 P2
                        p2_found = False
                        for _ in range(20): # 尝试 20 次选取 P2
                            p2_h, p2_m = _choice(p2_data)
                            if not (p1_m & p2_m): # 位运算快速冲突检测
                                p2_found = True
                                break
                        
                        if p2_found:
                            valid_matchup = True
                            break
                    
                    if not valid_matchup: continue 
                    
                    if cards_needed > 0:
                        used_set = set(p1_h) | set(p2_h)
                        deck_rem = [c for c in base_deck_list if c not in used_set]
                        if len(deck_rem) < cards_needed: continue
                        board_ext = _sample(deck_rem, cards_needed)
                        run_board = board_list + board_ext
                    else:
                        run_board = board_list

                    p1_s = _evaluate(run_board, p1_h)
                    p2_s = _evaluate(run_board, p2_h)
                    p1_rank_counts_list[_get_rc_safe(p1_s)] += 1
                    if p1_s < p2_s: p1_wins += 1
                    elif p2_s < p1_s: p2_wins += 1
                    valid_sims += 1
            else:
                # 不计算牌力版本
                for _ in loop_iter:
                    valid_matchup = False
                    for _ in range(20):
                        p1_h, p1_m = _choice(p1_data)
                        p2_found = False
                        for _ in range(20):
                            p2_h, p2_m = _choice(p2_data)
                            if not (p1_m & p2_m):
                                p2_found = True
                                break
                        if p2_found:
                            valid_matchup = True
                            break
                    
                    if not valid_matchup: continue

                    if cards_needed > 0:
                        used_set = set(p1_h) | set(p2_h)
                        deck_rem = [c for c in base_deck_list if c not in used_set]
                        if len(deck_rem) < cards_needed: continue
                        board_ext = _sample(deck_rem, cards_needed)
                        run_board = board_list + board_ext
                    else:
                        run_board = board_list

                    p1_s = _evaluate(run_board, p1_h)
                    p2_s = _evaluate(run_board, p2_h)
                    if p1_s < p2_s: p1_wins += 1
                    elif p2_s < p1_s: p2_wins += 1
                    valid_sims += 1
        
        # --- E2: Range vs Random ---
        else:
            for _ in loop_iter:
                valid_matchup = False
                
                # 带重试的选取逻辑
                for _ in range(50):
                    # 1. 选 P1
                    if p1_type == 'random':
                        p1_hand = _sample(base_deck_list, 2)
                    else:
                        p1_hand = _choice(real_p1_hands) 
                    p1_set = set(p1_hand)

                    # 2. 选 P2
                    p2_found = False
                    if p2_type == 'random':
                        for _ in range(20):
                            p2_hand = _sample(base_deck_list, 2)
                            if p2_hand[0] not in p1_set and p2_hand[1] not in p1_set: 
                                p2_found = True; break
                    else:
                        for _ in range(20):
                            p2_hand = _choice(real_p2_hands)
                            if set(p2_hand).isdisjoint(p1_set): 
                                p2_found = True; break
                    
                    if p2_found:
                        valid_matchup = True
                        break

                if not valid_matchup: continue
                
                p2_set = set(p2_hand)
                
                if cards_needed > 0:
                    deck_rem_set = base_deck_set - p1_set - p2_set
                    if len(deck_rem_set) < cards_needed: continue
                    board_ext = _sample(list(deck_rem_set), cards_needed)
                    run_board = board_list + board_ext
                else:
                    run_board = board_list
                
                p1_s = _evaluate(run_board, p1_hand)
                p2_s = _evaluate(run_board, p2_hand)

                if calculate_p1_strength: p1_rank_counts_list[_get_rc_safe(p1_s)] += 1
                if p1_s < p2_s: p1_wins += 1
                elif p2_s < p1_s: p2_wins += 1
                valid_sims += 1

    # --- 3. 结果格式化 ---
    ties: int = valid_sims - p1_wins - p2_wins
    
    p1_hand_strength_counts: dict[str, int] = {}
    for rank_val in range(1, 10):
        count = p1_rank_counts_list[rank_val]
        if count: 
            p1_hand_strength_counts[RANK_CLASS_TO_STRING.get(rank_val)] = count

    return p1_wins, p2_wins, ties, p1_hand_strength_counts, valid_sims


def _run_simulation_chunk_3way(args) -> tuple[float, float, float, dict, int]:
    """
    三人模式模拟任务块。返回 (p1_wins, p2_wins, p3_wins, p1_strength_counts, valid_sims)。
    平局时按人头均分胜率（如两人并列第一各 0.5，三人并列各 1/3）。
    """
    (p1_type, p1_hands, p2_type, p2_hands, p3_type, p3_hands, board, num_sims_chunk,
     master_deck_cards, calculate_p1_strength) = args

    evaluator = _get_worker_evaluator()
    if calculate_p1_strength:
        if not _CACHE_INITIALIZED or _RANK_CLASS_CACHE[1] == 0:
            _initialize_rank_cache(evaluator)

    _evaluate = evaluator.evaluate
    _sample = random.sample
    _choice = random.choice
    _rank_cache = _RANK_CLASS_CACHE

    def _get_rc_safe(score):
        rc = _rank_cache[score]
        if rc == 0:
            return evaluator.get_rank_class(score)
        return rc

    board_list = list(board)
    board_len = len(board_list)
    cards_needed = 5 - board_len
    board_set = set(board_list)
    base_deck_set = set(master_deck_cards) - board_set
    base_deck_list = list(base_deck_set)

    p1_wins_f, p2_wins_f, p3_wins_f = 0.0, 0.0, 0.0
    p1_rank_counts_list = [0] * 10
    valid_sims = 0

    def filter_hands(hands, bset):
        if not hands:
            return []
        return [h for h in hands if set(h).isdisjoint(bset)]

    real_p1 = filter_hands(p1_hands, board_set) if p1_type != 'random' else None
    real_p2 = filter_hands(p2_hands, board_set) if p2_type != 'random' else None
    real_p3 = filter_hands(p3_hands, board_set) if p3_type != 'random' else None

    if (p1_type != 'random' and not real_p1) or (p2_type != 'random' and not real_p2) or (p3_type != 'random' and not real_p3):
        return 0.0, 0.0, 0.0, {}, 0

    def pick_three_hands():
        used = set()
        # P1
        if p1_type == 'random':
            if len(base_deck_list) < 6:
                return None
            draw = _sample(base_deck_list, 6)
            h1 = draw[0:2]
            used.update(h1)
        elif len(real_p1) == 1:
            h1 = list(real_p1[0])
            if not set(h1).isdisjoint(used):
                return None
            used.update(h1)
        else:
            for _ in range(30):
                h1 = list(_choice(real_p1))
                if set(h1).isdisjoint(used):
                    used.update(h1)
                    break
            else:
                return None
        # P2
        deck2 = [c for c in base_deck_list if c not in used]
        if p2_type == 'random':
            if len(deck2) < 2:
                return None
            h2 = _sample(deck2, 2)
            used.update(h2)
        elif len(real_p2) == 1:
            h2 = list(real_p2[0])
            if not set(h2).isdisjoint(used):
                return None
            used.update(h2)
        else:
            valid_p2 = [h for h in real_p2 if set(h).isdisjoint(used)]
            if not valid_p2:
                return None
            h2 = list(_choice(valid_p2))
            used.update(h2)
        # P3
        deck3 = [c for c in base_deck_list if c not in used]
        if p3_type == 'random':
            if len(deck3) < 2:
                return None
            h3 = _sample(deck3, 2)
        elif len(real_p3) == 1:
            h3 = list(real_p3[0])
            if not set(h3).isdisjoint(used):
                return None
        else:
            valid_p3 = [h for h in real_p3 if set(h).isdisjoint(used)]
            if not valid_p3:
                return None
            h3 = list(_choice(valid_p3))
        return (h1, h2, h3)

    for _ in range(num_sims_chunk):
        hands = pick_three_hands()
        if hands is None:
            continue
        h1, h2, h3 = hands
        if cards_needed > 0:
            used = set(h1) | set(h2) | set(h3) | set(board_list)
            deck_rem = [c for c in base_deck_list if c not in used]
            if len(deck_rem) < cards_needed:
                continue
            board_ext = _sample(deck_rem, cards_needed)
            run_board = board_list + board_ext
        else:
            run_board = board_list

        s1 = _evaluate(run_board, h1)
        s2 = _evaluate(run_board, h2)
        s3 = _evaluate(run_board, h3)
        best = min(s1, s2, s3)
        winners = [i for i, s in enumerate([s1, s2, s3]) if s == best]
        frac = 1.0 / len(winners)
        p1_wins_f += frac if 0 in winners else 0.0
        p2_wins_f += frac if 1 in winners else 0.0
        p3_wins_f += frac if 2 in winners else 0.0
        if calculate_p1_strength:
            p1_rank_counts_list[_get_rc_safe(s1)] += 1
        valid_sims += 1

    strength_dict = {}
    if calculate_p1_strength:
        for rank_val in range(1, 10):
            c = p1_rank_counts_list[rank_val]
            if c:
                strength_dict[RANK_CLASS_TO_STRING.get(rank_val)] = c
    return p1_wins_f, p2_wins_f, p3_wins_f, strength_dict, valid_sims


def _run_simulation_chunk_4way(args) -> tuple[float, float, float, float, dict, int]:
    """
    四人模式模拟任务块。返回 (p1_wins, p2_wins, p3_wins, p4_wins, p1_strength_counts, valid_sims)。
    平局时按人头均分胜率。
    """
    (p1_type, p1_hands, p2_type, p2_hands, p3_type, p3_hands, p4_type, p4_hands, board, num_sims_chunk,
     master_deck_cards, calculate_p1_strength) = args

    evaluator = _get_worker_evaluator()
    if calculate_p1_strength:
        if not _CACHE_INITIALIZED or _RANK_CLASS_CACHE[1] == 0:
            _initialize_rank_cache(evaluator)

    _evaluate = evaluator.evaluate
    _sample = random.sample
    _choice = random.choice
    _rank_cache = _RANK_CLASS_CACHE

    def _get_rc_safe(score):
        rc = _rank_cache[score]
        if rc == 0:
            return evaluator.get_rank_class(score)
        return rc

    board_list = list(board)
    cards_needed = 5 - len(board_list)
    board_set = set(board_list)
    base_deck_set = set(master_deck_cards) - board_set
    base_deck_list = list(base_deck_set)

    p1_wins_f, p2_wins_f, p3_wins_f, p4_wins_f = 0.0, 0.0, 0.0, 0.0
    p1_rank_counts_list = [0] * 10
    valid_sims = 0

    def filter_hands(hands, bset):
        if not hands:
            return []
        return [h for h in hands if set(h).isdisjoint(bset)]

    real_p1 = filter_hands(p1_hands, board_set) if p1_type != 'random' else None
    real_p2 = filter_hands(p2_hands, board_set) if p2_type != 'random' else None
    real_p3 = filter_hands(p3_hands, board_set) if p3_type != 'random' else None
    real_p4 = filter_hands(p4_hands, board_set) if p4_type != 'random' else None

    if (p1_type != 'random' and not real_p1) or (p2_type != 'random' and not real_p2) or (p3_type != 'random' and not real_p3) or (p4_type != 'random' and not real_p4):
        return 0.0, 0.0, 0.0, 0.0, {}, 0

    def pick_four_hands():
        used = set()
        if p1_type == 'random':
            if len(base_deck_list) < 8:
                return None
            draw = _sample(base_deck_list, 8)
            h1 = draw[0:2]
            used.update(h1)
        elif len(real_p1) == 1:
            h1 = list(real_p1[0])
            if not set(h1).isdisjoint(used):
                return None
            used.update(h1)
        else:
            for _ in range(30):
                h1 = list(_choice(real_p1))
                if set(h1).isdisjoint(used):
                    used.update(h1)
                    break
            else:
                return None
        deck2 = [c for c in base_deck_list if c not in used]
        if p2_type == 'random':
            if len(deck2) < 2:
                return None
            h2 = _sample(deck2, 2)
            used.update(h2)
        elif len(real_p2) == 1:
            h2 = list(real_p2[0])
            if not set(h2).isdisjoint(used):
                return None
            used.update(h2)
        else:
            valid_p2 = [h for h in real_p2 if set(h).isdisjoint(used)]
            if not valid_p2:
                return None
            h2 = list(_choice(valid_p2))
            used.update(h2)
        deck3 = [c for c in base_deck_list if c not in used]
        if p3_type == 'random':
            if len(deck3) < 2:
                return None
            h3 = _sample(deck3, 2)
            used.update(h3)
        elif len(real_p3) == 1:
            h3 = list(real_p3[0])
            if not set(h3).isdisjoint(used):
                return None
            used.update(h3)
        else:
            valid_p3 = [h for h in real_p3 if set(h).isdisjoint(used)]
            if not valid_p3:
                return None
            h3 = list(_choice(valid_p3))
            used.update(h3)
        deck4 = [c for c in base_deck_list if c not in used]
        if p4_type == 'random':
            if len(deck4) < 2:
                return None
            h4 = _sample(deck4, 2)
        elif len(real_p4) == 1:
            h4 = list(real_p4[0])
            if not set(h4).isdisjoint(used):
                return None
        else:
            valid_p4 = [h for h in real_p4 if set(h).isdisjoint(used)]
            if not valid_p4:
                return None
            h4 = list(_choice(valid_p4))
        return (h1, h2, h3, h4)

    for _ in range(num_sims_chunk):
        hands = pick_four_hands()
        if hands is None:
            continue
        h1, h2, h3, h4 = hands
        if cards_needed > 0:
            used = set(h1) | set(h2) | set(h3) | set(h4) | set(board_list)
            deck_rem = [c for c in base_deck_list if c not in used]
            if len(deck_rem) < cards_needed:
                continue
            board_ext = _sample(deck_rem, cards_needed)
            run_board = board_list + board_ext
        else:
            run_board = board_list

        s1 = _evaluate(run_board, h1)
        s2 = _evaluate(run_board, h2)
        s3 = _evaluate(run_board, h3)
        s4 = _evaluate(run_board, h4)
        best = min(s1, s2, s3, s4)
        winners = [i for i, s in enumerate([s1, s2, s3, s4]) if s == best]
        frac = 1.0 / len(winners)
        p1_wins_f += frac if 0 in winners else 0.0
        p2_wins_f += frac if 1 in winners else 0.0
        p3_wins_f += frac if 2 in winners else 0.0
        p4_wins_f += frac if 3 in winners else 0.0
        if calculate_p1_strength:
            p1_rank_counts_list[_get_rc_safe(s1)] += 1
        valid_sims += 1

    strength_dict = {}
    if calculate_p1_strength:
        for rank_val in range(1, 10):
            c = p1_rank_counts_list[rank_val]
            if c:
                strength_dict[RANK_CLASS_TO_STRING.get(rank_val)] = c
    return p1_wins_f, p2_wins_f, p3_wins_f, p4_wins_f, strength_dict, valid_sims


def _run_simulation_chunk_5way(args) -> tuple[float, float, float, float, float, dict, int]:
    """
    五人模式模拟任务块。返回 (p1_wins, p2_wins, p3_wins, p4_wins, p5_wins, p1_strength_counts, valid_sims)。
    平局时按人头均分胜率。
    """
    (p1_type, p1_hands, p2_type, p2_hands, p3_type, p3_hands, p4_type, p4_hands, p5_type, p5_hands,
     board, num_sims_chunk, master_deck_cards, calculate_p1_strength) = args

    evaluator = _get_worker_evaluator()
    if calculate_p1_strength:
        if not _CACHE_INITIALIZED or _RANK_CLASS_CACHE[1] == 0:
            _initialize_rank_cache(evaluator)

    _evaluate = evaluator.evaluate
    _sample = random.sample
    _choice = random.choice
    _rank_cache = _RANK_CLASS_CACHE

    def _get_rc_safe(score):
        rc = _rank_cache[score]
        if rc == 0:
            return evaluator.get_rank_class(score)
        return rc

    board_list = list(board)
    cards_needed = 5 - len(board_list)
    board_set = set(board_list)
    base_deck_set = set(master_deck_cards) - board_set
    base_deck_list = list(base_deck_set)

    p1_wins_f, p2_wins_f, p3_wins_f, p4_wins_f, p5_wins_f = 0.0, 0.0, 0.0, 0.0, 0.0
    p1_rank_counts_list = [0] * 10
    valid_sims = 0

    def filter_hands(hands, bset):
        if not hands:
            return []
        return [h for h in hands if set(h).isdisjoint(bset)]

    real_p1 = filter_hands(p1_hands, board_set) if p1_type != 'random' else None
    real_p2 = filter_hands(p2_hands, board_set) if p2_type != 'random' else None
    real_p3 = filter_hands(p3_hands, board_set) if p3_type != 'random' else None
    real_p4 = filter_hands(p4_hands, board_set) if p4_type != 'random' else None
    real_p5 = filter_hands(p5_hands, board_set) if p5_type != 'random' else None

    if (p1_type != 'random' and not real_p1) or (p2_type != 'random' and not real_p2) or (p3_type != 'random' and not real_p3) or (p4_type != 'random' and not real_p4) or (p5_type != 'random' and not real_p5):
        return 0.0, 0.0, 0.0, 0.0, 0.0, {}, 0

    def pick_five_hands():
        used = set()
        hands = []
        for p_type, real_list in [(p1_type, real_p1), (p2_type, real_p2), (p3_type, real_p3), (p4_type, real_p4), (p5_type, real_p5)]:
            deck_cur = [c for c in base_deck_list if c not in used]
            if p_type == 'random':
                if len(deck_cur) < 2:
                    return None
                h = _sample(deck_cur, 2)
            elif len(real_list) == 1:
                h = list(real_list[0])
                if not set(h).isdisjoint(used):
                    return None
            else:
                valid = [h for h in real_list if set(h).isdisjoint(used)]
                if not valid:
                    return None
                h = list(_choice(valid))
            hands.append(h)
            used.update(h)
        return tuple(hands)

    for _ in range(num_sims_chunk):
        hands = pick_five_hands()
        if hands is None:
            continue
        h1, h2, h3, h4, h5 = hands
        if cards_needed > 0:
            used = set(h1) | set(h2) | set(h3) | set(h4) | set(h5) | set(board_list)
            deck_rem = [c for c in base_deck_list if c not in used]
            if len(deck_rem) < cards_needed:
                continue
            board_ext = _sample(deck_rem, cards_needed)
            run_board = board_list + board_ext
        else:
            run_board = board_list

        s1 = _evaluate(run_board, h1)
        s2 = _evaluate(run_board, h2)
        s3 = _evaluate(run_board, h3)
        s4 = _evaluate(run_board, h4)
        s5 = _evaluate(run_board, h5)
        best = min(s1, s2, s3, s4, s5)
        winners = [i for i, s in enumerate([s1, s2, s3, s4, s5]) if s == best]
        frac = 1.0 / len(winners)
        p1_wins_f += frac if 0 in winners else 0.0
        p2_wins_f += frac if 1 in winners else 0.0
        p3_wins_f += frac if 2 in winners else 0.0
        p4_wins_f += frac if 3 in winners else 0.0
        p5_wins_f += frac if 4 in winners else 0.0
        if calculate_p1_strength:
            p1_rank_counts_list[_get_rc_safe(s1)] += 1
        valid_sims += 1

    strength_dict = {}
    if calculate_p1_strength:
        for rank_val in range(1, 10):
            c = p1_rank_counts_list[rank_val]
            if c:
                strength_dict[RANK_CLASS_TO_STRING.get(rank_val)] = c
    return p1_wins_f, p2_wins_f, p3_wins_f, p4_wins_f, p5_wins_f, strength_dict, valid_sims


# --- 核心扑克逻辑模块 ---
class PokerLogic:
    def __init__(self):
        self.evaluator = Evaluator()
        self.rank_class_to_string_map = {
            1: "同花顺 (Straight Flush)", 2: "四条 (Four of a Kind)", 3: "葫芦 (Full House)",
            4: "同花 (Flush)", 5: "顺子 (Straight)", 6: "三条 (Three of a Kind)",
            7: "两对 (Two Pair)", 8: "一对 (One Pair)", 9: "高牌 (High Card)"
        }
        self.master_deck = Deck()

    def _parse_hand_range(self, range_str_list):
        hand_pairs = []
        for r_str in filter(None, [s.strip().upper() for s in range_str_list]):
            try:
                if len(r_str) == 2 and r_str[0] == r_str[1]:
                    rank = r_str[0]
                    suits = 'shdc'
                    for i in range(len(suits)):
                        for j in range(i + 1, len(suits)):
                            card1 = Card.new(f"{rank}{suits[i]}")
                            card2 = Card.new(f"{rank}{suits[j]}")
                            hand_pairs.append([card1, card2])
                elif len(r_str) == 3:
                    r1, r2, s = r_str[0], r_str[1], r_str[2].lower()
                    suits = 'shdc'
                    if s == 's':
                        for suit in suits:
                            card1 = Card.new(f"{r1}{suit}")
                            card2 = Card.new(f"{r2}{suit}")
                            hand_pairs.append([card1, card2])
                    elif s == 'o':
                        for s1 in suits:
                            for s2 in suits:
                                if s1 != s2:
                                    card1 = Card.new(f"{r1}{s1}")
                                    card2 = Card.new(f"{r2}{s2}")
                                    hand_pairs.append([card1, card2])
                elif len(r_str) == 2:
                    r1, r2 = r_str[0], r_str[1]
                    suits = 'shdc'
                    for s1 in suits:
                        for s2 in suits:
                            card1 = Card.new(f"{r1}{s1}")
                            card2 = Card.new(f"{r2}{s2}")
                            hand_pairs.append([card1, card2])
            except (ValueError, KeyError) as e:
                print(f"警告: 忽略无效的范围字符串 '{r_str}': {e}")
                continue
        return hand_pairs

    def _split_hand_str(self, s):
        cards_raw = [s[i:i+2] for i in range(0, len(s), 2)] # ["AS", "AH"]
        formatted_cards = []
        for card in cards_raw:
            if len(card) == 2:
                formatted_cards.append(card[0].upper() + card[1].lower())
            else:
                formatted_cards.append(card) 
        return formatted_cards

    def _determine_input_type(self, p_input):
        clean_input = [s.strip().upper() for s in p_input if s.strip()]
        if not clean_input or clean_input == ['']:
            return 'random', None

        all_hand_pairs = []
        specific_hands_in_list = set()
        range_strings = []

        for item in clean_input:
            if len(item) == 4:
                try:
                    r1 = item[0].upper()
                    s1 = item[1].lower()
                    r2 = item[2].upper()
                    s2 = item[3].lower()
                    c1 = Card.new(r1 + s1)
                    c2 = Card.new(r2 + s2)
                    
                    if c1 == c2:
                        raise ValueError(f"手牌 '{item}' 包含重复的牌。")
                        
                    specific_hands_in_list.add(item)
                except (ValueError, KeyError):
                    range_strings.append(item)
            else:
                range_strings.append(item)
        
        # 1. 解析所有特定的手牌
        for hand_str in specific_hands_in_list:
            try:
                cards = self._split_hand_str(hand_str)
                hand = [Card.new(c) for c in cards]
                if len(set(hand)) != 2:
                    raise ValueError(f"手牌 '{hand_str}' 包含重复的牌。")
                all_hand_pairs.append(hand)
            except (ValueError, KeyError) as e:
                raise ValueError(f"解析特定手牌 '{hand_str}' 时出错: {e}")
                
        # 2. 解析所有范围字符串
        if range_strings:
            try:
                all_hand_pairs.extend(self._parse_hand_range(range_strings))
            except Exception as e:
                print(f"解析范围时出错: {e}")

        if not all_hand_pairs:
            if specific_hands_in_list or range_strings:
                 raise ValueError("无法从输入解析出任何有效手牌。")
            else:
                 return 'random', None

        deduped_set = {tuple(sorted(h)) for h in all_hand_pairs}
        final_hands_list = [list(t) for t in deduped_set]
        
        if not final_hands_list: 
            raise ValueError("无法解析手牌范围或特定手牌。")
            
        if len(final_hands_list) == 1 and not range_strings:
             return 'hand', final_hands_list
        else:
            return 'range', final_hands_list
            
    def run_analysis(self, p1_input_raw, p2_input_raw, board_str, num_simulations=50000, progress_callback=None):
        
        def check_hand_list(hand_list, seen_cards_set, player_name):
            for hand in hand_list:
                hand_set = set(hand)
                if not hand_set.isdisjoint(seen_cards_set):
                    conflicting_card = (hand_set & seen_cards_set).pop()
                    card_str = Card.int_to_str(conflicting_card)
                    hand_str = [Card.int_to_str(c) for c in hand]
                    raise ValueError(f"{player_name} 内部冲突: 牌 {card_str} (在手牌 {hand_str} 中) 已被使用。")
            for hand in hand_list:
                seen_cards_set.update(hand)

        def filter_range(hand_list, seen_cards_set, player_name):
            valid_hands = []
            for hand in hand_list:
                if set(hand).isdisjoint(seen_cards_set):
                    valid_hands.append(hand)
            if not valid_hands and hand_list:
                raise ValueError(f"{player_name} 的范围与已选卡牌完全冲突。")
            return valid_hands

        try:
            # --- 步骤 1: 解析所有输入 ---
            board = [Card.new(c) for c in self._split_hand_str(board_str)] if board_str else []
            if len(set(board)) != len(board): 
                raise ValueError("公共牌中包含重复的牌。")
            
            p1_type, p1_hands = self._determine_input_type(p1_input_raw)
            p2_type, p2_hands = self._determine_input_type(p2_input_raw)
            
            calculate_p1_strength = True

            # --- 步骤 2: 健壮的冲突检测 ---
            seen_cards = set(board)

            if p1_type == 'hand': check_hand_list(p1_hands, seen_cards, "玩家1")
            if p2_type == 'hand': check_hand_list(p2_hands, seen_cards, "玩家2")

            if p1_type == 'range': p1_hands = filter_range(p1_hands, seen_cards, "玩家1")
            if p2_type == 'range': p2_hands = filter_range(p2_hands, seen_cards, "玩家2")

        except ValueError as e:
            raise ValueError(f"输入解析错误: {e}")

        # --- 步骤 3: 并行计算设置 ---
        try:
            num_cores = max(1, multiprocessing.cpu_count() - 1)
        except NotImplementedError:
            num_cores = 1
            
        TARGET_SMOOTHNESS_TASKS = 150  # 任务分块数

        if num_simulations <= 0:
            num_tasks, chunk_size, remainder = 0, 0, 0
        else:
            num_tasks = min(TARGET_SMOOTHNESS_TASKS, num_simulations)
            chunk_size = num_simulations // num_tasks
            remainder = num_simulations % num_tasks
        
        tasks = []
        master_deck_cards = list(self.master_deck.cards)
        base_args = (p1_type, p1_hands, p2_type, p2_hands, board, master_deck_cards, calculate_p1_strength)

        for _ in range(num_tasks):
            if chunk_size > 0:
                tasks.append(base_args[:5] + (chunk_size,) + base_args[5:])
        if remainder > 0:
            tasks.append(base_args[:5] + (remainder,) + base_args[5:])

        total_p1_wins, total_p2_wins, total_ties = 0, 0, 0
        total_p1_strength_counts = defaultdict(int)
        total_valid_sims = 0
        
        # --- 步骤 4: 执行 (优化：使用全局进程池) ---
        pool = _get_or_create_pool(num_cores)
        
        completed_sims = 0
        for result_chunk in pool.imap_unordered(_run_simulation_chunk, tasks):
            p1_wins, p2_wins, ties, p1_strength_counts, valid_sims = result_chunk
            
            total_p1_wins += p1_wins
            total_p2_wins += p2_wins
            total_ties += ties
            for hand, count in p1_strength_counts.items():
                total_p1_strength_counts[hand] += count
            total_valid_sims += valid_sims
            
            if progress_callback:
                completed_sims += valid_sims
                progress_callback(completed_sims)
        
        if progress_callback: 
            progress_callback(num_simulations)
            
        if total_valid_sims == 0 and num_simulations > 0: 
            raise ValueError("无法完成任何有效模拟。请检查输入设置（例如手牌和公共牌冲突）。")
        
        # --- 步骤 5: 汇总结果 ---
        equity_results = {'p1_win': 0, 'p2_win': 0, 'tie': 0}
        if total_valid_sims > 0:
            equity_results = {
                'p1_win': (total_p1_wins / total_valid_sims) * 100,
                'p2_win': (total_p2_wins / total_valid_sims) * 100,
                'tie': (total_ties / total_valid_sims) * 100
            }
            
        strength_results = {}
        if calculate_p1_strength:
            total_strength_hands = sum(total_p1_strength_counts.values())
            if total_strength_hands > 0:
                for rank in range(1, 10):
                    hand_type = self.rank_class_to_string_map[rank]
                    prob = (total_p1_strength_counts.get(hand_type, 0) / total_strength_hands) * 100
                    strength_results[hand_type] = prob
        
        return equity_results, strength_results, calculate_p1_strength

    def run_analysis_3way(self, p1_input_raw, p2_input_raw, p3_input_raw, board_str, num_simulations=50000, progress_callback=None):
        """三人模式胜率分析。返回 (equity_results, strength_results, show_strength)。"""
        def check_hand_list(hand_list, seen_cards_set, player_name):
            for hand in hand_list:
                hand_set = set(hand)
                if not hand_set.isdisjoint(seen_cards_set):
                    conflicting_card = (hand_set & seen_cards_set).pop()
                    card_str = Card.int_to_str(conflicting_card)
                    hand_str = [Card.int_to_str(c) for c in hand]
                    raise ValueError(f"{player_name} 内部冲突: 牌 {card_str} (在手牌 {hand_str} 中) 已被使用。")
            for hand in hand_list:
                seen_cards_set.update(hand)

        def filter_range(hand_list, seen_cards_set, player_name):
            valid_hands = [h for h in hand_list if set(h).isdisjoint(seen_cards_set)]
            if not valid_hands and hand_list:
                raise ValueError(f"{player_name} 的范围与已选卡牌完全冲突。")
            return valid_hands

        try:
            board = [Card.new(c) for c in self._split_hand_str(board_str)] if board_str else []
            if len(set(board)) != len(board):
                raise ValueError("公共牌中包含重复的牌。")

            p1_type, p1_hands = self._determine_input_type(p1_input_raw)
            p2_type, p2_hands = self._determine_input_type(p2_input_raw)
            p3_type, p3_hands = self._determine_input_type(p3_input_raw)
            calculate_p1_strength = True

            seen_cards = set(board)
            if p1_type == 'hand':
                check_hand_list(p1_hands, seen_cards, "玩家1")
            if p2_type == 'hand':
                check_hand_list(p2_hands, seen_cards, "玩家2")
            if p3_type == 'hand':
                check_hand_list(p3_hands, seen_cards, "玩家3")
            if p1_type == 'range':
                p1_hands = filter_range(p1_hands, seen_cards, "玩家1")
            if p2_type == 'range':
                p2_hands = filter_range(p2_hands, seen_cards, "玩家2")
            if p3_type == 'range':
                p3_hands = filter_range(p3_hands, seen_cards, "玩家3")
        except ValueError as e:
            raise ValueError(f"输入解析错误: {e}")

        try:
            num_cores = max(1, multiprocessing.cpu_count() - 1)
        except NotImplementedError:
            num_cores = 1
        TARGET_SMOOTHNESS_TASKS = 150
        if num_simulations <= 0:
            num_tasks, chunk_size, remainder = 0, 0, 0
        else:
            num_tasks = min(TARGET_SMOOTHNESS_TASKS, num_simulations)
            chunk_size = num_simulations // num_tasks
            remainder = num_simulations % num_tasks

        master_deck_cards = list(self.master_deck.cards)
        base_args = (p1_type, p1_hands, p2_type, p2_hands, p3_type, p3_hands, board, master_deck_cards, calculate_p1_strength)
        tasks = []
        for _ in range(num_tasks):
            if chunk_size > 0:
                tasks.append(base_args[:7] + (chunk_size,) + base_args[7:])
        if remainder > 0:
            tasks.append(base_args[:7] + (remainder,) + base_args[7:])

        total_p1_wins = total_p2_wins = total_p3_wins = 0.0
        total_p1_strength_counts = defaultdict(int)
        total_valid_sims = 0
        pool = _get_or_create_pool(num_cores)
        completed_sims = 0
        for result_chunk in pool.imap_unordered(_run_simulation_chunk_3way, tasks):
            p1w, p2w, p3w, p1_strength, valid_sims = result_chunk
            total_p1_wins += p1w
            total_p2_wins += p2w
            total_p3_wins += p3w
            for hand_type, count in p1_strength.items():
                total_p1_strength_counts[hand_type] += count
            total_valid_sims += valid_sims
            if progress_callback:
                completed_sims += valid_sims
                progress_callback(completed_sims)
        if progress_callback:
            progress_callback(num_simulations)
        if total_valid_sims == 0 and num_simulations > 0:
            raise ValueError("无法完成任何有效模拟。请检查输入设置（例如手牌和公共牌冲突）。")

        equity_results = {'p1_win': 0, 'p2_win': 0, 'p3_win': 0}
        if total_valid_sims > 0:
            equity_results = {
                'p1_win': (total_p1_wins / total_valid_sims) * 100,
                'p2_win': (total_p2_wins / total_valid_sims) * 100,
                'p3_win': (total_p3_wins / total_valid_sims) * 100
            }
        strength_results = {}
        if calculate_p1_strength:
            total_strength_hands = sum(total_p1_strength_counts.values())
            if total_strength_hands > 0:
                for rank in range(1, 10):
                    hand_type = self.rank_class_to_string_map[rank]
                    prob = (total_p1_strength_counts.get(hand_type, 0) / total_strength_hands) * 100
                    strength_results[hand_type] = prob
        return equity_results, strength_results, calculate_p1_strength

    def run_analysis_4way(self, p1_input_raw, p2_input_raw, p3_input_raw, p4_input_raw, board_str, num_simulations=50000, progress_callback=None):
        """四人模式胜率分析。返回 (equity_results, strength_results, show_strength)。"""
        def check_hand_list(hand_list, seen_cards_set, player_name):
            for hand in hand_list:
                hand_set = set(hand)
                if not hand_set.isdisjoint(seen_cards_set):
                    conflicting_card = (hand_set & seen_cards_set).pop()
                    card_str = Card.int_to_str(conflicting_card)
                    hand_str = [Card.int_to_str(c) for c in hand]
                    raise ValueError(f"{player_name} 内部冲突: 牌 {card_str} (在手牌 {hand_str} 中) 已被使用。")
            for hand in hand_list:
                seen_cards_set.update(hand)

        def filter_range(hand_list, seen_cards_set, player_name):
            valid_hands = [h for h in hand_list if set(h).isdisjoint(seen_cards_set)]
            if not valid_hands and hand_list:
                raise ValueError(f"{player_name} 的范围与已选卡牌完全冲突。")
            return valid_hands

        try:
            board = [Card.new(c) for c in self._split_hand_str(board_str)] if board_str else []
            if len(set(board)) != len(board):
                raise ValueError("公共牌中包含重复的牌。")

            p1_type, p1_hands = self._determine_input_type(p1_input_raw)
            p2_type, p2_hands = self._determine_input_type(p2_input_raw)
            p3_type, p3_hands = self._determine_input_type(p3_input_raw)
            p4_type, p4_hands = self._determine_input_type(p4_input_raw)
            calculate_p1_strength = True

            seen_cards = set(board)
            if p1_type == 'hand':
                check_hand_list(p1_hands, seen_cards, "玩家1")
            if p2_type == 'hand':
                check_hand_list(p2_hands, seen_cards, "玩家2")
            if p3_type == 'hand':
                check_hand_list(p3_hands, seen_cards, "玩家3")
            if p4_type == 'hand':
                check_hand_list(p4_hands, seen_cards, "玩家4")
            if p1_type == 'range':
                p1_hands = filter_range(p1_hands, seen_cards, "玩家1")
            if p2_type == 'range':
                p2_hands = filter_range(p2_hands, seen_cards, "玩家2")
            if p3_type == 'range':
                p3_hands = filter_range(p3_hands, seen_cards, "玩家3")
            if p4_type == 'range':
                p4_hands = filter_range(p4_hands, seen_cards, "玩家4")
        except ValueError as e:
            raise ValueError(f"输入解析错误: {e}")

        try:
            num_cores = max(1, multiprocessing.cpu_count() - 1)
        except NotImplementedError:
            num_cores = 1
        TARGET_SMOOTHNESS_TASKS = 150
        if num_simulations <= 0:
            num_tasks, chunk_size, remainder = 0, 0, 0
        else:
            num_tasks = min(TARGET_SMOOTHNESS_TASKS, num_simulations)
            chunk_size = num_simulations // num_tasks
            remainder = num_simulations % num_tasks

        master_deck_cards = list(self.master_deck.cards)
        base_args = (p1_type, p1_hands, p2_type, p2_hands, p3_type, p3_hands, p4_type, p4_hands, board, master_deck_cards, calculate_p1_strength)
        tasks = []
        for _ in range(num_tasks):
            if chunk_size > 0:
                tasks.append(base_args[:9] + (chunk_size,) + base_args[9:])
        if remainder > 0:
            tasks.append(base_args[:9] + (remainder,) + base_args[9:])

        total_p1_wins = total_p2_wins = total_p3_wins = total_p4_wins = 0.0
        total_p1_strength_counts = defaultdict(int)
        total_valid_sims = 0
        pool = _get_or_create_pool(num_cores)
        completed_sims = 0
        for result_chunk in pool.imap_unordered(_run_simulation_chunk_4way, tasks):
            p1w, p2w, p3w, p4w, p1_strength, valid_sims = result_chunk
            total_p1_wins += p1w
            total_p2_wins += p2w
            total_p3_wins += p3w
            total_p4_wins += p4w
            for hand_type, count in p1_strength.items():
                total_p1_strength_counts[hand_type] += count
            total_valid_sims += valid_sims
            if progress_callback:
                completed_sims += valid_sims
                progress_callback(completed_sims)
        if progress_callback:
            progress_callback(num_simulations)
        if total_valid_sims == 0 and num_simulations > 0:
            raise ValueError("无法完成任何有效模拟。请检查输入设置（例如手牌和公共牌冲突）。")

        equity_results = {'p1_win': 0, 'p2_win': 0, 'p3_win': 0, 'p4_win': 0}
        if total_valid_sims > 0:
            equity_results = {
                'p1_win': (total_p1_wins / total_valid_sims) * 100,
                'p2_win': (total_p2_wins / total_valid_sims) * 100,
                'p3_win': (total_p3_wins / total_valid_sims) * 100,
                'p4_win': (total_p4_wins / total_valid_sims) * 100
            }
        strength_results = {}
        if calculate_p1_strength:
            total_strength_hands = sum(total_p1_strength_counts.values())
            if total_strength_hands > 0:
                for rank in range(1, 10):
                    hand_type = self.rank_class_to_string_map[rank]
                    prob = (total_p1_strength_counts.get(hand_type, 0) / total_strength_hands) * 100
                    strength_results[hand_type] = prob
        return equity_results, strength_results, calculate_p1_strength

    def run_analysis_5way(self, p1_input_raw, p2_input_raw, p3_input_raw, p4_input_raw, p5_input_raw, board_str, num_simulations=50000, progress_callback=None):
        """五人模式胜率分析。返回 (equity_results, strength_results, show_strength)。"""
        def check_hand_list(hand_list, seen_cards_set, player_name):
            for hand in hand_list:
                hand_set = set(hand)
                if not hand_set.isdisjoint(seen_cards_set):
                    conflicting_card = (hand_set & seen_cards_set).pop()
                    card_str = Card.int_to_str(conflicting_card)
                    hand_str = [Card.int_to_str(c) for c in hand]
                    raise ValueError(f"{player_name} 内部冲突: 牌 {card_str} (在手牌 {hand_str} 中) 已被使用。")
            for hand in hand_list:
                seen_cards_set.update(hand)

        def filter_range(hand_list, seen_cards_set, player_name):
            valid_hands = [h for h in hand_list if set(h).isdisjoint(seen_cards_set)]
            if not valid_hands and hand_list:
                raise ValueError(f"{player_name} 的范围与已选卡牌完全冲突。")
            return valid_hands

        try:
            board = [Card.new(c) for c in self._split_hand_str(board_str)] if board_str else []
            if len(set(board)) != len(board):
                raise ValueError("公共牌中包含重复的牌。")

            p1_type, p1_hands = self._determine_input_type(p1_input_raw)
            p2_type, p2_hands = self._determine_input_type(p2_input_raw)
            p3_type, p3_hands = self._determine_input_type(p3_input_raw)
            p4_type, p4_hands = self._determine_input_type(p4_input_raw)
            p5_type, p5_hands = self._determine_input_type(p5_input_raw)
            calculate_p1_strength = True

            seen_cards = set(board)
            for i, (pt, ph, name) in enumerate([(p1_type, p1_hands, "玩家1"), (p2_type, p2_hands, "玩家2"), (p3_type, p3_hands, "玩家3"), (p4_type, p4_hands, "玩家4"), (p5_type, p5_hands, "玩家5")]):
                if pt == 'hand':
                    check_hand_list(ph, seen_cards, name)
            if p1_type == 'range': p1_hands = filter_range(p1_hands, seen_cards, "玩家1")
            if p2_type == 'range': p2_hands = filter_range(p2_hands, seen_cards, "玩家2")
            if p3_type == 'range': p3_hands = filter_range(p3_hands, seen_cards, "玩家3")
            if p4_type == 'range': p4_hands = filter_range(p4_hands, seen_cards, "玩家4")
            if p5_type == 'range': p5_hands = filter_range(p5_hands, seen_cards, "玩家5")
        except ValueError as e:
            raise ValueError(f"输入解析错误: {e}")

        try:
            num_cores = max(1, multiprocessing.cpu_count() - 1)
        except NotImplementedError:
            num_cores = 1
        TARGET_SMOOTHNESS_TASKS = 150
        if num_simulations <= 0:
            num_tasks, chunk_size, remainder = 0, 0, 0
        else:
            num_tasks = min(TARGET_SMOOTHNESS_TASKS, num_simulations)
            chunk_size = num_simulations // num_tasks
            remainder = num_simulations % num_tasks

        master_deck_cards = list(self.master_deck.cards)
        base_args = (p1_type, p1_hands, p2_type, p2_hands, p3_type, p3_hands, p4_type, p4_hands, p5_type, p5_hands, board, master_deck_cards, calculate_p1_strength)
        tasks = []
        for _ in range(num_tasks):
            if chunk_size > 0:
                tasks.append(base_args[:11] + (chunk_size,) + base_args[11:])
        if remainder > 0:
            tasks.append(base_args[:11] + (remainder,) + base_args[11:])

        total_p1_wins = total_p2_wins = total_p3_wins = total_p4_wins = total_p5_wins = 0.0
        total_p1_strength_counts = defaultdict(int)
        total_valid_sims = 0
        pool = _get_or_create_pool(num_cores)
        completed_sims = 0
        for result_chunk in pool.imap_unordered(_run_simulation_chunk_5way, tasks):
            p1w, p2w, p3w, p4w, p5w, p1_strength, valid_sims = result_chunk
            total_p1_wins += p1w
            total_p2_wins += p2w
            total_p3_wins += p3w
            total_p4_wins += p4w
            total_p5_wins += p5w
            for hand_type, count in p1_strength.items():
                total_p1_strength_counts[hand_type] += count
            total_valid_sims += valid_sims
            if progress_callback:
                completed_sims += valid_sims
                progress_callback(completed_sims)
        if progress_callback:
            progress_callback(num_simulations)
        if total_valid_sims == 0 and num_simulations > 0:
            raise ValueError("无法完成任何有效模拟。请检查输入设置（例如手牌和公共牌冲突）。")

        equity_results = {'p1_win': 0, 'p2_win': 0, 'p3_win': 0, 'p4_win': 0, 'p5_win': 0}
        if total_valid_sims > 0:
            equity_results = {
                'p1_win': (total_p1_wins / total_valid_sims) * 100,
                'p2_win': (total_p2_wins / total_valid_sims) * 100,
                'p3_win': (total_p3_wins / total_valid_sims) * 100,
                'p4_win': (total_p4_wins / total_valid_sims) * 100,
                'p5_win': (total_p5_wins / total_valid_sims) * 100
            }
        strength_results = {}
        if calculate_p1_strength:
            total_strength_hands = sum(total_p1_strength_counts.values())
            if total_strength_hands > 0:
                for rank in range(1, 10):
                    hand_type = self.rank_class_to_string_map[rank]
                    prob = (total_p1_strength_counts.get(hand_type, 0) / total_strength_hands) * 100
                    strength_results[hand_type] = prob
        return equity_results, strength_results, calculate_p1_strength


# ##################################################################
# ############### 花色选择器弹出窗口 ################################
# ##################################################################
class SuitSelectorWindow(tk.Toplevel):
    def __init__(self, master, hand_text, callback):
        super().__init__(master)
        self.withdraw()
        
        self.hand_text = hand_text
        self.callback = callback
        self.hand_type = None
        self.rank1 = hand_text[0]
        self.rank2 = hand_text[1]

        try:
            self.iconbitmap(os.path.join(os.path.dirname(__file__), "TexasPoker.ico"))
        except tk.TclError:
            pass
        
        self.FONT_STATUS = ("Microsoft YaHei", 16, "bold")
        self.FONT_SYMBOL = ("Microsoft YaHei", 24, "bold")
        self.FONT_LABEL = ("Microsoft YaHei", 12)
        self.FONT_PROMPT = ("Microsoft YaHei", 14)
        self.FONT_SELECTED = ("Microsoft YaHei", 14, "italic")
        
        self.BG_COLOR = '#2e2e2e'
        self.BTN_BG_NORMAL = '#4a4a4a'
        self.BTN_FG_NORMAL = 'white'
        self.BTN_BG_HOVER = '#6a6a6a'
        self.BTN_BG_DISABLED = '#3a3a3a'
        self.BTN_FG_DISABLED = '#5c5c5c'
        self.BTN_BG_SELECTED = '#007bff'
        self.BTN_BORDER_SELECTED = '#80bfff'
        self.LABEL_FG = '#d0d0d0'

        if hand_text.endswith('s'):
            self.hand_type = 'suited'
            self.title(f"选择 {hand_text} 花色")
            self.selection = [] 
        else:
            self.hand_type = 'pair' if self.rank1 == self.rank2 else 'offsuit'
            self.title(f"选择 {hand_text} 花色")
            self.selection = [None, None]

        self.configure(bg=self.BG_COLOR)
        self.transient(master)
        self.resizable(False, False)
        
        master_x = master.winfo_x()
        master_y = master.winfo_y()
        master_w = master.winfo_width()
        master_h = master.winfo_height()
        win_w, win_h = 580, 350 
        pos_x = master_x + (master_w // 2) - (win_w // 2)
        pos_y = master_y + (master_h // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

        main_frame = ttk.Frame(self, padding="20") 
        main_frame.pack(fill='both', expand=True)
        
        self.status_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.status_var, font=self.FONT_STATUS, anchor='center').pack(pady=(0, 20), fill='x')
        
        self.selection_frame = ttk.Frame(main_frame)
        self.selection_frame.pack(fill='x', expand=True, pady=5)

        self.buttons_r1 = {}
        self.buttons_r2 = {}
        
        self.suits_map_display = {'s': ('♠', 'black'), 'h': ('♥', 'red'), 'd': ('♦', 'blue'), 'c': ('♣', 'green')}
        self.button_key_list = ['s', 'h', 'd', 'c', 'unknown'] 
        
        def create_suit_button(parent, suit_key, command_func, state='normal'):
            if suit_key == 'unknown':
                disp, color = '?', self.LABEL_FG
            else:
                disp, color = self.suits_map_display[suit_key]

            btn = tk.Label(parent, text=disp, font=self.FONT_SYMBOL, 
                           fg=color, bg=self.BTN_BG_NORMAL,
                           width=3, height=1, relief='raised', borderwidth=2)
            btn.suit_key = suit_key
            
            def on_click(e):
                if btn['state'] == 'normal':
                    command_func(btn.suit_key) 
            
            def on_enter(e):
                if btn['state'] == 'normal':
                    btn.config(bg=self.BTN_BG_HOVER)
            
            def on_leave(e):
                if btn['state'] == 'normal':
                    is_selected = False
                    if self.hand_type in ['offsuit', 'pair'] and self.selection:
                        is_selected = (btn.suit_key == self.selection[0]) or (btn.suit_key == self.selection[1])
                    
                    if not is_selected:
                        btn.config(bg=self.BTN_BG_NORMAL)
            
            btn.bind("<Button-1>", on_click)
            btn.bind("<Button-3>", on_click)
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            
            if state == 'disabled':
                btn.config(state='disabled', fg=self.BTN_FG_DISABLED, bg=self.BTN_BG_DISABLED, relief='flat')
            
            return btn

        if self.hand_type == 'offsuit' or self.hand_type == 'pair':
            r1_cmd = self._on_r1_click if self.hand_type == 'offsuit' else self._on_pair_r1_click
            r2_cmd = self._on_r2_click if self.hand_type == 'offsuit' else self._on_pair_r2_click
            
            frame_r1 = ttk.Frame(self.selection_frame)
            frame_r1.pack(pady=5, fill='x')
            ttk.Label(frame_r1, text=f"牌 1 ({self.rank1}):", font=self.FONT_LABEL, foreground=self.LABEL_FG, width=10).pack(side='left', padx=(10, 10))
            for key in self.button_key_list:
                btn = create_suit_button(frame_r1, key, r1_cmd) 
                btn.pack(side='left', padx=6, pady=4)
                self.buttons_r1[key] = btn

            frame_r2 = ttk.Frame(self.selection_frame)
            frame_r2.pack(pady=5, fill='x')
            ttk.Label(frame_r2, text=f"牌 2 ({self.rank2}):", font=self.FONT_LABEL, foreground=self.LABEL_FG, width=10).pack(side='left', padx=(10, 10))
            for key in self.button_key_list:
                btn = create_suit_button(frame_r2, key, r2_cmd, state='disabled') 
                btn.pack(side='left', padx=6, pady=4)
                self.buttons_r2[key] = btn
        
        elif self.hand_type == 'suited':
            prompt = f"为 {self.rank1}{self.rank2}s 选择花色:"
            ttk.Label(self.selection_frame, text=prompt, font=self.FONT_PROMPT, foreground=self.LABEL_FG, anchor='center').pack(pady=10, fill='x')
            
            button_frame = ttk.Frame(self.selection_frame)
            button_frame.pack(pady=10) 
            for key in self.button_key_list:
                btn = create_suit_button(button_frame, key, self._on_suited_click) 
                btn.pack(side='left', padx=8, pady=5)
                self.buttons_r1[key] = btn 

        self.selection_label_var = tk.StringVar(value="已选: -")
        ttk.Label(main_frame, textvariable=self.selection_label_var, font=self.FONT_SELECTED, foreground='#ccc', anchor='center').pack(pady=(15, 0), fill='x')

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side='bottom', fill='x', pady=(20, 0))
        
        ttk.Button(control_frame, text="重置", command=self._reset).pack(fill='x', expand=True, padx=10, ipady=10)
                   
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._update_status_label() 
        
        self.deiconify()
        self.grab_set()

    def _update_status_label(self):
        suits_map = self.suits_map_display 

        if self.hand_type == 'offsuit' or self.hand_type == 'pair':
            s1 = self.selection[0]
            if s1 is None:
                self.status_var.set(f"请为 {self.rank1} 选择花色")
                self.selection_label_var.set("已选: -")
            else:
                s1_disp = '?' if s1 == 'unknown' else suits_map[s1][0]
                self.status_var.set(f"请为 {self.rank2} 选择花色")
                self.selection_label_var.set(f"已选: {self.rank1}{s1_disp}")
        
        elif self.hand_type == 'suited':
            self.status_var.set(f"请为 {self.hand_text} 选择 1 种花色")
            self.selection_label_var.set("已选: -")

    def _enable_r2_buttons(self, r1_key_char):
        for key, btn in self.buttons_r2.items():
            is_disabled = (r1_key_char != 'unknown' and key == r1_key_char)
            if is_disabled:
                btn.config(state='disabled', fg=self.BTN_FG_DISABLED, bg=self.BTN_BG_DISABLED, relief='flat')
            else:
                color = self.LABEL_FG if key == 'unknown' else self.suits_map_display[key][1]
                btn.config(state='normal', fg=color, bg=self.BTN_BG_NORMAL, relief='raised')
                btn.bind("<Enter>", btn.bindtags()[0], "+")
                btn.bind("<Leave>", btn.bindtags()[0], "+")

    def _disable_all_r1_buttons(self, selected_key):
        for btn in self.buttons_r1.values():
            btn.config(state='disabled', fg=self.BTN_FG_DISABLED, bg=self.BTN_BG_DISABLED, relief='flat')
            btn.unbind("<Enter>")
            btn.unbind("<Leave>")
        self.buttons_r1[selected_key].config(bg=self.BTN_BG_SELECTED, relief='solid', borderwidth=2, highlightbackground=self.BTN_BORDER_SELECTED) 

    def _disable_all_r2_buttons(self, selected_key):
        for btn in self.buttons_r2.values():
            btn.config(state='disabled', fg=self.BTN_FG_DISABLED, bg=self.BTN_BG_DISABLED, relief='flat')
            btn.unbind("<Enter>")
            btn.unbind("<Leave>")
        self.buttons_r2[selected_key].config(bg=self.BTN_BG_SELECTED, relief='solid', borderwidth=2, highlightbackground=self.BTN_BORDER_SELECTED) 

    def _on_pair_r1_click(self, key_char):
        self.selection[0] = key_char
        self._disable_all_r1_buttons(key_char)
        self._enable_r2_buttons(key_char)
        self._update_status_label()

    def _on_pair_r2_click(self, key_char):
        self.selection[1] = key_char
        self._disable_all_r2_buttons(key_char)
        
        s1, s2 = self.selection[0], self.selection[1]
        final_submission = None
        all_suits = ['s', 'h', 'd', 'c']

        if s1 != 'unknown' and s2 != 'unknown':
            final_submission = f"{self.rank1}{s1}{self.rank1}{s2}".upper()
        elif s1 == 'unknown' and s2 == 'unknown':
            final_submission = self.hand_text
        else:
            final_hands_list = []
            if s1 == 'unknown' and s2 != 'unknown':
                for suit1 in all_suits:
                    if suit1 == s2: continue 
                    final_hands_list.append(f"{self.rank1}{suit1}{self.rank1}{s2}".upper())
            elif s1 != 'unknown' and s2 == 'unknown':
                for suit2 in all_suits:
                    if suit2 == s1: continue
                    final_hands_list.append(f"{self.rank1}{s1}{self.rank1}{suit2}".upper())
            final_submission = ", ".join(final_hands_list)

        self.callback(final_submission)
        self.destroy()

    def _on_r1_click(self, key_char):
        self.selection[0] = key_char
        self._disable_all_r1_buttons(key_char)
        self._enable_r2_buttons(key_char)
        self._update_status_label()

    def _on_r2_click(self, key_char):
        self.selection[1] = key_char
        self._disable_all_r2_buttons(key_char)
        s1, s2 = self.selection[0], self.selection[1]
        final_submission = None
        all_suits = ['s', 'h', 'd', 'c']

        if s1 != 'unknown' and s2 != 'unknown':
            final_submission = f"{self.rank1}{s1}{self.rank2}{s2}".upper()
        elif s1 == 'unknown' and s2 == 'unknown':
            final_submission = self.hand_text
        else:
            final_hands_list = []
            if s1 == 'unknown' and s2 != 'unknown':
                for suit1 in all_suits:
                    if suit1 == s2: continue
                    final_hands_list.append(f"{self.rank1}{suit1}{self.rank2}{s2}".upper())
            elif s1 != 'unknown' and s2 == 'unknown':
                for suit2 in all_suits:
                    if suit2 == s1: continue
                    final_hands_list.append(f"{self.rank1}{s1}{self.rank2}{suit2}".upper())
            final_submission = ", ".join(final_hands_list)

        self.callback(final_submission)
        self.destroy()

    def _on_suited_click(self, key_char):
        if key_char == 'unknown':
            self.callback(self.hand_text)
            self.destroy()
            return

        self.selection.append(key_char)
        for btn in self.buttons_r1.values():
            btn.config(state='disabled', fg=self.BTN_FG_DISABLED, bg=self.BTN_BG_DISABLED, relief='flat')
            btn.unbind("<Enter>")
            btn.unbind("<Leave>")
        self.buttons_r1[key_char].config(bg=self.BTN_BG_SELECTED, relief='solid', borderwidth=2, highlightbackground=self.BTN_BORDER_SELECTED) 

        s_disp = self.suits_map_display[key_char][0]
        self.selection_label_var.set(f"已选: {self.rank1}{s_disp}{self.rank2}{s_disp}")

        final_hand = f"{self.rank1}{key_char}{self.rank2}{key_char}"
        self.callback(final_hand.upper())
        self.destroy()

    def _on_cancel(self):
        self.callback(None)
        self.destroy()

    def _reset(self):
        def reset_button_row(buttons_dict, state='normal'):
            for key, btn in buttons_dict.items():
                if key == 'unknown':
                    color = self.LABEL_FG
                else:
                    color = self.suits_map_display[key][1]
                btn.config(state=state, fg=color, bg=self.BTN_BG_NORMAL, relief='raised')
                if state == 'normal':
                    btn.bind("<Enter>", btn.bindtags()[0], "+")
                    btn.bind("<Leave>", btn.bindtags()[0], "+")
                else:
                    btn.unbind("<Enter>")
                    btn.unbind("<Leave>")

        if self.hand_type == 'offsuit' or self.hand_type == 'pair':
            self.selection = [None, None]
            reset_button_row(self.buttons_r1, state='normal')
            reset_button_row(self.buttons_r2, state='disabled')
        elif self.hand_type == 'suited':
            self.selection = []
            reset_button_row(self.buttons_r1, state='normal')
        self._update_status_label()


# ############### 渐变进度条类 #####################################
class GradientProgressBar(tk.Canvas):
    def __init__(self, parent, color_list, max_val=100, width=200, height=20, bg_color='#2e2e2e', **kwargs):
        super().__init__(parent, width=width, height=height, bg=bg_color, highlightthickness=0, borderwidth=0, **kwargs)
        self.color_list = color_list
        self.max_val = max_val
        self.current_val = 0
        self.bg_color = bg_color
        self.bind('<Configure>', self._on_resize)
        self._width = 1
        self._height = 1
        self._mask_id = None
        
    def _hex_to_rgb(self, hex_col):
        hex_col = hex_col.lstrip('#')
        return tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(self, rgb):
        return '#{:02x}{:02x}{:02x}'.format(int(max(0, min(255, rgb[0]))), int(max(0, min(255, rgb[1]))), int(max(0, min(255, rgb[2]))))

    def _interpolate(self, color1, color2, t):
        c1 = self._hex_to_rgb(color1)
        c2 = self._hex_to_rgb(color2)
        new_rgb = tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(3))
        return self._rgb_to_hex(new_rgb)
    
    def set_colors(self, new_color_list):
        self.color_list = new_color_list
        self._draw_gradient()

    def _draw_gradient(self):
        self.delete("all")
        self._width = self.winfo_width()
        self._height = self.winfo_height()
        if self._width <= 1: return
        num_colors = len(self.color_list)
        if num_colors < 2: return
        
        limit_segments = 120 
        step = max(1, self._width // limit_segments)
        for x in range(0, self._width, step):
            center_x = x + step / 2
            t_global = center_x / self._width
            idx = int(t_global * (num_colors - 1))
            idx = min(idx, num_colors - 2)
            t_local = (t_global * (num_colors - 1)) - idx
            color = self._interpolate(self.color_list[idx], self.color_list[idx+1], t_local)
            x1 = x
            x2 = min(x + step, self._width)
            self.create_rectangle(x1, 0, x2, self._height, fill=color, outline="", tags="gradient")
            
        self._mask_id = self.create_rectangle(0, 0, self._width, self._height, fill=self.bg_color, outline="", tags="mask")
        self._update_mask_position()

    def _on_resize(self, event):
        self._draw_gradient()

    def _update_mask_position(self):
        if not self._mask_id: return
        if self.max_val <= 0: pct = 0
        else: pct = min(1.0, max(0.0, self.current_val / self.max_val))
        x_pos = int(pct * self._width)
        self.coords(self._mask_id, x_pos, -5, self._width + 5, self._height + 5)
        
    def set_value(self, value):
        self.current_val = value
        self._update_mask_position()
        
    def set_max(self, max_val):
        self.max_val = max_val
        self._update_mask_position()


# --- 启动页：选择 2～6 人模式 ---
class StartupWindow(tk.Tk):
    """精美的启动页，用于选择 2/3/4/5 人对战模式。"""
    def __init__(self):
        super().__init__()
        self.selected_players = None  # 2～5 或 None（未选/关闭）
        self._icon_path = None
        self._glow_items = []  # 存储需要动画的元素
        self._glow_phase = 0   # 动画相位
        
        try:
            base_path = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(base_path, "TexasPoker.ico")
            if os.path.exists(icon_path):
                self._icon_path = icon_path
                self.iconbitmap(default=icon_path)
        except (tk.TclError, Exception):
            pass

        self.title("Texas Hold'em Equity Calculator")
        self.resizable(False, False)
        w, h = 760, 580
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{int((screen_w - w) / 2)}+{int((screen_h - h) / 2) - 50}")
        self.configure(bg='#0a0c10')
        self.option_add('*Font', ('Microsoft YaHei', 10))
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # 主画布
        self.main_canvas = tk.Canvas(self, width=w, height=h, bg='#0a0c10', highlightthickness=0)
        self.main_canvas.pack(fill='both', expand=True)
        c = self.main_canvas  # 简写

        # ========== 背景装饰 ==========
        # 深色渐变背景（模拟）
        for i in range(h):
            # 从顶部深蓝到底部更深的黑
            ratio = i / h
            r = int(10 + ratio * 2)
            g = int(12 + ratio * 4)
            b = int(16 + ratio * 6)
            c.create_line(0, i, w, i, fill=f'#{r:02x}{g:02x}{b:02x}')
        
        # 装饰性网格线（营造科技感）
        for x in range(0, w, 60):
            c.create_line(x, 0, x, h, fill='#151922', width=1)
        for y in range(0, h, 60):
            c.create_line(0, y, w, y, fill='#151922', width=1)

        # ========== 顶部发光条 ==========
        # 主发光条（带渐变效果）
        glow_colors = ['#0d3d30', '#0f5040', '#11a080', '#00ffaa', '#11a080', '#0f5040', '#0d3d30']
        bar_height = 4
        segment_w = w // len(glow_colors)
        for i, color in enumerate(glow_colors):
            c.create_rectangle(i * segment_w, 0, (i + 1) * segment_w + 1, bar_height, fill=color, outline='')
        
        # 顶部区域背景
        c.create_rectangle(0, bar_height, w, 70, fill='#0d1117', outline='')
        c.create_line(0, 70, w, 70, fill='#1a2332', width=1)

        # ========== 装饰性扑克牌符号 ==========
        suits = ['♠', '♥', '♦', '♣']
        suit_colors = ['#2a3444', '#3a2030', '#3a3020', '#2a3444']
        positions = [(80, 150), (w-80, 150), (100, h-100), (w-100, h-100)]
        for (px, py), suit, color in zip(positions, suits, suit_colors):
            c.create_text(px, py, text=suit, fill=color, font=('Arial', 48, 'bold'))
        
        # ========== Logo 区域 ==========
        # Logo 背景光晕
        for r in range(80, 0, -5):
            alpha_hex = hex(int(15 - r * 0.15))[2:].zfill(2)
            c.create_oval(w/2 - r, 115 - r/2, w/2 + r, 115 + r/2, 
                         fill='', outline=f'#00{alpha_hex}55', width=2)
        
        # 主标题 "TEXAS HOLD'EM"
        c.create_text(w/2, 100, text="TEXAS HOLD'EM", 
                     fill='#00ffaa', font=('Arial Black', 32, 'bold'))
        # 标题阴影/辉光效果
        c.create_text(w/2 + 2, 102, text="TEXAS HOLD'EM", 
                     fill='#004433', font=('Arial Black', 32, 'bold'))
        # 重绘前景标题
        title_id = c.create_text(w/2, 100, text="TEXAS HOLD'EM", 
                                fill='#00ffaa', font=('Arial Black', 32, 'bold'))
        self._glow_items.append(('title', title_id))
        
        # 副标题
        c.create_text(w/2, 145, text="EQUITY CALCULATOR", 
                     fill='#5a6a7a', font=('Arial', 14, 'bold'))
        
        # 中文标题
        c.create_text(w/2, 178, text="— 德州扑克胜率分析器 —", 
                     fill='#3a4a5a', font=('Microsoft YaHei', 11))

        # ========== 分隔线装饰 ==========
        line_y = 210
        # 中间亮线
        c.create_line(150, line_y, w-150, line_y, fill='#00aa77', width=1)
        # 两端菱形装饰
        diamond_size = 6
        for dx in [150, w-150]:
            c.create_polygon(dx, line_y - diamond_size, dx + diamond_size, line_y,
                            dx, line_y + diamond_size, dx - diamond_size, line_y,
                            fill='#00aa77', outline='')
        
        # "选择对战模式" 文字
        c.create_text(w/2, 245, text="SELECT PLAYERS", 
                     fill='#6a7a8a', font=('Arial', 11, 'bold'))
        c.create_text(w/2, 265, text="选择分析人数", 
                     fill='#4a5a6a', font=('Microsoft YaHei', 10))

        # ========== 模式选择按钮 ==========
        btn_configs = [
            {'n': 2, 'color': '#00d4aa', 'hover': '#00ffcc', 'glow': '#004433', 'label': '双人对决', 'icon': '👥'},
            {'n': 3, 'color': '#ff7043', 'hover': '#ff8a65', 'glow': '#4d2213', 'label': '三人混战', 'icon': '👥'},
            {'n': 4, 'color': '#7c4dff', 'hover': '#9e7bff', 'glow': '#2a1a4d', 'label': '四人对局', 'icon': '👥'},
            {'n': 5, 'color': '#ff4081', 'hover': '#ff79a8', 'glow': '#4d1228', 'label': '五人激战', 'icon': '👥'},
        ]
        
        self._mode_buttons = []
        btn_width, btn_height = 140, 110
        total_width = len(btn_configs) * btn_width + (len(btn_configs) - 1) * 20
        start_x = (w - total_width) / 2
        btn_y = 320
        
        for i, cfg in enumerate(btn_configs):
            x = start_x + i * (btn_width + 20)
            btn_data = self._create_mode_button(c, x, btn_y, btn_width, btn_height, cfg)
            self._mode_buttons.append(btn_data)

        # ========== 底部信息栏 ==========
        footer_y = h - 70
        c.create_rectangle(0, footer_y, w, h, fill='#0d1117', outline='')
        c.create_line(0, footer_y, w, footer_y, fill='#1a2332', width=1)
        
        # 底部装饰点
        for dx in range(30, w, 60):
            c.create_oval(dx-2, footer_y + 6, dx+2, footer_y + 10, fill='#1a2a3a', outline='')
        
        # 功能说明
        features = ["手牌分析", "范围计算", "公共牌模拟", "多人对战"]
        feature_x = w / (len(features) + 1)
        for i, feat in enumerate(features):
            fx = feature_x * (i + 1)
            c.create_text(fx, h - 42, text="●", fill='#00aa77', font=('Arial', 8))
            c.create_text(fx, h - 24, text=feat, fill='#5a6a7a', font=('Microsoft YaHei', 9))
        
        # 版本号
        c.create_text(w - 40, h - 15, text="v2.0", fill='#3a4a5a', font=('Arial', 9))

        # 启动发光动画
        self._animate_glow()

    def _create_mode_button(self, canvas, x, y, w, h, cfg):
        """创建带悬停效果的模式选择按钮"""
        c = canvas
        n = cfg['n']
        color = cfg['color']
        hover_color = cfg['hover']
        glow_color = cfg['glow']
        
        # 按钮背景（带圆角效果模拟）
        corner = 12
        # 主背景
        bg_items = []
        # 外发光
        for offset in range(4, 0, -1):
            glow_id = c.create_rectangle(x - offset, y - offset, x + w + offset, y + h + offset,
                                        fill='', outline=glow_color, width=1)
            bg_items.append(glow_id)
        
        # 主按钮背景
        main_bg = c.create_rectangle(x, y, x + w, y + h, fill='#151c25', outline=color, width=2)
        bg_items.append(main_bg)
        
        # 顶部高亮条
        top_bar = c.create_rectangle(x + 2, y + 2, x + w - 2, y + 6, fill=color, outline='')
        bg_items.append(top_bar)
        
        # 人数大字
        num_text = c.create_text(x + w/2, y + 42, text=str(n), 
                                fill=color, font=('Arial Black', 36, 'bold'))
        
        # "人" 字
        ren_text = c.create_text(x + w/2 + 30, y + 50, text="人", 
                                fill='#5a6a7a', font=('Microsoft YaHei', 12))
        
        # 标签文字
        label_text = c.create_text(x + w/2, y + 85, text=cfg['label'], 
                                  fill='#6a7a8a', font=('Microsoft YaHei', 10))
        
        all_items = bg_items + [num_text, ren_text, label_text]
        
        # 绑定点击事件
        def on_click(event):
            self._on_select(n)
        
        def on_enter(event):
            c.itemconfig(main_bg, fill='#1a2530', outline=hover_color)
            c.itemconfig(num_text, fill=hover_color)
            c.itemconfig(top_bar, fill=hover_color)
            c.config(cursor='hand2')
        
        def on_leave(event):
            c.itemconfig(main_bg, fill='#151c25', outline=color)
            c.itemconfig(num_text, fill=color)
            c.itemconfig(top_bar, fill=color)
            c.config(cursor='')
        
        # 为所有元素绑定事件
        for item in all_items:
            c.tag_bind(item, '<Button-1>', on_click)
            c.tag_bind(item, '<Enter>', on_enter)
            c.tag_bind(item, '<Leave>', on_leave)
        
        return {'items': all_items, 'n': n, 'color': color, 'hover': hover_color}

    def _animate_glow(self):
        """标题发光动画"""
        if not self.winfo_exists():
            return
        
        self._glow_phase = (self._glow_phase + 1) % 60
        
        # 计算颜色明度变化（正弦波）
        brightness = 0.7 + 0.3 * math.sin(self._glow_phase * math.pi / 30)
        
        # 更新标题颜色
        for item_type, item_id in self._glow_items:
            if item_type == 'title':
                # 绿色通道变化
                g = int(255 * brightness)
                b = int(170 * brightness)
                color = f'#00{g:02x}{b:02x}'
                try:
                    self.main_canvas.itemconfig(item_id, fill=color)
                except tk.TclError:
                    return
        
        # 继续动画
        self.after(50, self._animate_glow)

    def _on_close(self):
        self.selected_players = None
        self.quit()

    def _on_select(self, num_players):
        if num_players in (2, 3, 4, 5):
            self.selected_players = num_players
            self.quit()
        else:
            messagebox.showinfo("提示", "请选择 2～5 人进行对战分析。")


# --- GUI 应用 ---
class PokerApp(tk.Tk):
    def __init__(self, poker_logic, num_players=2):
        super().__init__()
        self.withdraw()
        self.wm_attributes('-alpha', 0.0)

        self.poker_logic = poker_logic
        self.num_players = num_players
        title_suffix = " (三人模式)" if num_players == 3 else " (四人模式)" if num_players == 4 else " (五人模式)" if num_players == 5 else ""
        self.title("德州扑克分析工具" + title_suffix)
        window_width = 1370
        # 四人/五人模式调大窗口高度，便于显示更多内容
        window_height = 993 if num_players == 4 else 993 if num_players == 5 else 960
        
        # 获取图标路径（支持开发环境和 PyInstaller 打包环境）
        self._icon_path = None
        try:
            if getattr(sys, 'frozen', False):
                # 打包后的环境
                base_path = sys._MEIPASS
            else:
                # 开发环境
                base_path = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(base_path, "TexasPoker.ico")
            if os.path.exists(icon_path):
                self._icon_path = icon_path
                # 设置窗口图标
                self.iconbitmap(default=icon_path)
                self.iconbitmap(icon_path)
        except (tk.TclError, Exception) as e:
            print(f"图标加载警告: {e}")

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        position_x = int(screen_width / 2 - window_width / 2)
        position_y = int(screen_height / 2 - window_height / 2) - 33
        self.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.configure(bg='#2e2e2e')
        self.protocol("WM_DELETE_WINDOW", self._on_app_close)
        
        self.board_cards = []
        self.analysis_thread = None
        self.analysis_result = None
        self.progress_var = tk.DoubleVar()
        # 四位玩家胜率条与按钮区分色：蓝、红、绿、紫
        self.P1_COLOR = '#007bff'   # 玩家1 蓝
        self.P2_COLOR = '#dc3545'   # 玩家2 红
        self.P3_COLOR = '#28a745'   # 玩家3 绿
        self.P4_COLOR = '#6f42c1'   # 玩家4 紫
        self.P5_COLOR = '#fd7e14'   # 玩家5 橙
        self.BOTH_COLOR = '#8a2be2'
        self.DEFAULT_BG = '#4f4f4f'
        self.PAIR_BG = '#8fbc8f'
        self.SUITED_BG = '#4a7a96'
        
        self.pb_themes = [
            ['#4158D0', '#C850C0', '#FFCC70'], ['#0093E9', '#80D0C7'], ['#8EC5FC', '#E0C3FC'],
            ['#FA8BFF', '#2BD2FF', '#2BFF88'], ['#FF9A9E', '#FECFEF'], ['#FBAB7E', '#F7CE68'],
            ['#85FFBD', '#FFFB7F'], ['#FF3CAC', '#784BA0', '#2B86C5'], ['#FF9966', '#FF5E62'],
            ['#56ab2f', '#a8e063'], ['#F12711', '#F5AF19'], ['#FC466B', '#3F5EFB'],
            ['#C6FFDD', '#FBD786', '#f7797d'], ['#12c2e9', '#c471ed', '#f64f59'], ['#b92b27', '#1565C0'],
            ['#00F260', '#0575E6'], ['#8E2DE2', '#4A00E0'], ['#00c3ff', '#ffff1c'],
            ['#ee0979', '#ff6a00'], ['#DA22FF', '#9733EE'], ['#1D976C', '#93F9B9'],
            ['#E55D87', '#5FC3E4']
        ]
        
        self.strength_chart_window = None
        self._configure_styles()
        self._create_widgets()
        self.bind_all('<KeyPress>', self._handle_window_movement)
        self.bind_all('<Return>', self._on_enter_key_pressed)
        
        self.deiconify()
        # 窗口显示后再次确保图标正确设置
        if self._icon_path:
            self.after(100, lambda: self._apply_icon())
        self.after(10, self._start_fade_in)

    def _apply_icon(self):
        """在窗口显示后重新应用图标"""
        try:
            if self._icon_path and os.path.exists(self._icon_path):
                self.iconbitmap(default=self._icon_path)
                self.iconbitmap(self._icon_path)
        except (tk.TclError, Exception):
            pass

    def _on_app_close(self):
        global _GLOBAL_POOL
        if _GLOBAL_POOL is not None:
            try:
                _GLOBAL_POOL.terminate()
                _GLOBAL_POOL.join()
            except Exception:
                pass
        self.destroy()

    def _handle_window_movement(self, event):
        if self.strength_chart_window and self.strength_chart_window.winfo_exists():
            return
        focused_widget = self.focus_get()
        if isinstance(focused_widget, ttk.Entry):
            return
        move_step = 15
        x = self.winfo_x()
        y = self.winfo_y()
        key = event.keysym.lower()
        if key == 'w' or key == 'up': y -= move_step
        elif key == 's' or key == 'down': y += move_step
        elif key == 'a' or key == 'left': x -= move_step
        elif key == 'd' or key == 'right': x += move_step
        else: return
        self.geometry(f"+{x}+{y}")

    def _on_enter_key_pressed(self, event):
        """处理 Enter 键按下事件，触发开始分析"""
        # 如果分析按钮处于可用状态，则触发分析
        if str(self.calc_button['state']) != 'disabled':
            self.run_analysis_thread()

    def _open_strength_chart(self):
        if StrengthChartWindow is None:
            messagebox.showinfo("功能未启用", "强度图表模块未加载。")
            return
        if self.strength_chart_window and self.strength_chart_window.winfo_exists():
            self.strength_chart_window.lift()
            self.strength_chart_window.focus_force()
            return
        self.open_chart_button.config(state='disabled')
        self.strength_chart_window = StrengthChartWindow(self)

    def _specific_to_range_category(self, hand_str):
        try:
            if len(hand_str) != 4: return None
            # 支持图案格式，先转换为字母格式
            suits_reverse = {'♠': 's', '♥': 'h', '♦': 'd', '♣': 'c'}
            converted = hand_str
            for symbol, letter in suits_reverse.items():
                converted = converted.replace(symbol, letter)
            
            r1, s1 = converted[0].upper(), converted[1].lower()
            r2, s2 = converted[2].upper(), converted[3].lower()
            ranks_order = 'AKQJT98765432'
            if r1 not in ranks_order or r2 not in ranks_order or s1 not in 'shdc' or s2 not in 'shdc':
                return None
            if ranks_order.index(r1) > ranks_order.index(r2):
                r1, r2 = r2, r1 
            if r1 == r2: return f"{r1}{r2}"
            elif s1 == s2: return f"{r1}{r2}s"
            else: return f"{r1}{r2}o"
        except Exception:
            return None

    def _start_fade_in(self):
        self.animation_total_duration = 300
        self.animation_step_delay = 15
        try: total_steps = self.animation_total_duration / self.animation_step_delay
        except ZeroDivisionError: total_steps = 0
        if total_steps == 0:
            self.wm_attributes('-alpha', 1.0)
            return
        self.progress_increment = 1.0 / total_steps
        self.current_progress = 0.0
        self._fade_in_step()

    def _fade_in_step(self):
        self.current_progress += self.progress_increment
        if self.current_progress >= 1.0:
            self.wm_attributes('-alpha', 1.0)
        else:
            eased_alpha = math.sin(self.current_progress * (math.pi / 2))
            try: self.wm_attributes('-alpha', eased_alpha)
            except tk.TclError: return
            self.after(self.animation_step_delay, self._fade_in_step)

    def _configure_styles(self):
        self.style.configure('.', background='#2e2e2e', foreground='white')
        self.style.configure('TFrame', background='#2e2e2e')
        self.style.configure('TLabel', background='#2e2e2e', foreground='white', font=('Microsoft YaHei', 10))
        self.style.configure('TLabelframe', background='#2e2e2e', bordercolor='#888')
        self.style.configure('TLabelframe.Label', background='#2e2e2e', foreground='white', font=('Microsoft YaHei', 11, 'bold'))
        self.style.configure('TEntry', fieldbackground='#4a4a4a', foreground='white', insertbackground='white')
        self.style.configure('Treeview', fieldbackground='#3c3c3c', background='#3c3c3c', foreground='white', rowheight=25)
        self.style.configure('Treeview.Heading', font=('Microsoft YaHei', 11, 'bold'), background='#4a4a4a', foreground='white')
        self.style.map('Treeview.Heading', background=[('active', '#6a6a6a')])
        self.style.configure("p1.Horizontal.TProgressbar", background=self.P1_COLOR)
        self.style.configure("p2.Horizontal.TProgressbar", background=self.P2_COLOR)
        self.style.configure("p3.Horizontal.TProgressbar", background=self.P3_COLOR)
        self.style.configure("p4.Horizontal.TProgressbar", background=self.P4_COLOR)
        self.style.configure("p5.Horizontal.TProgressbar", background=self.P5_COLOR)
        self.style.configure("tie.Horizontal.TProgressbar", background='#6c757d')
        self.style.configure('TButton', background='#4a4a4a', foreground='white', font=('Microsoft YaHei', 10, 'bold'), borderwidth=1)
        self.style.map('TButton', background=[('active', '#6a6a6a'), ('disabled', '#3a3a3a')])
        self.style.layout('TButton', [('Button.border', {'children': [('Button.padding', {'children': [('Button.label', {'side': 'left', 'expand': 1})]})]})])
 
    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill='both', expand=True)
        left_pane = ttk.Frame(main_frame)
        left_pane.pack(side='left', fill='both', expand=True, padx=(0, 10))
        right_pane = ttk.Frame(main_frame)
        right_pane.pack(side='left', fill='both', expand=True)
        self._create_control_pane(left_pane)
        self._create_analysis_pane(right_pane)

    def _create_control_pane(self, parent_pane):
        parent_pane.columnconfigure(0, weight=1)
        parent_pane.rowconfigure(2, weight=1)
        
        player_setup_frame = ttk.LabelFrame(parent_pane, text="玩家设置")
        player_setup_frame.pack(fill='x', pady=5)
        player_setup_frame.columnconfigure(1, weight=1)

        self.p1_hand_var = tk.StringVar()
        self.p2_hand_var = tk.StringVar()
        self.p3_hand_var = None
        self.p4_hand_var = None
        self.p5_hand_var = None

        if self.num_players == 4:
            # 四人模式：手牌/范围栏两组切换，节省高度
            self._player_setup_group_var = tk.IntVar(value=1)  # 1=玩家1、2  2=玩家3、4
            group_switch_row = ttk.Frame(player_setup_frame)
            group_switch_row.grid(row=0, column=0, columnspan=3, sticky='w', padx=5, pady=(5, 2))
            self._btn_players_12 = tk.Button(group_switch_row, text=" 玩家1、2 ", relief='flat', bg=self.P1_COLOR, fg='white', font=('Microsoft YaHei', 9, 'bold'), cursor='hand2', command=lambda: self._switch_player_setup_group(1), takefocus=0)
            self._btn_players_12.pack(side='left', padx=2, ipady=3)
            self._btn_players_34 = tk.Button(group_switch_row, text=" 玩家3、4 ", relief='flat', bg='#4a4a4a', fg='white', font=('Microsoft YaHei', 9, 'bold'), cursor='hand2', command=lambda: self._switch_player_setup_group(2), takefocus=0)
            self._btn_players_34.pack(side='left', padx=2, ipady=3)

            self._player_setup_frame_12 = ttk.Frame(player_setup_frame)
            self._player_setup_frame_12.grid(row=1, column=0, columnspan=3, sticky='ew', pady=0)
            self._player_setup_frame_12.columnconfigure(1, weight=1)
            ttk.Label(self._player_setup_frame_12, text="玩家1 (手牌/范围):").grid(row=0, column=0, padx=10, pady=6, sticky='w')
            ttk.Entry(self._player_setup_frame_12, textvariable=self.p1_hand_var, font=('Segoe UI Symbol', 12)).grid(row=0, column=1, padx=10, pady=6, sticky='ew')
            ttk.Button(self._player_setup_frame_12, text="重置", command=self._reset_player1, width=8).grid(row=0, column=2, padx=5, pady=6)
            ttk.Label(self._player_setup_frame_12, text="玩家2 (手牌/范围):").grid(row=1, column=0, padx=10, pady=6, sticky='w')
            ttk.Entry(self._player_setup_frame_12, textvariable=self.p2_hand_var, font=('Segoe UI Symbol', 12)).grid(row=1, column=1, padx=10, pady=6, sticky='ew')
            ttk.Button(self._player_setup_frame_12, text="重置", command=self._reset_player2, width=8).grid(row=1, column=2, padx=5, pady=6)

            self.p3_hand_var = tk.StringVar()
            self.p4_hand_var = tk.StringVar()
            self._player_setup_frame_34 = ttk.Frame(player_setup_frame)
            self._player_setup_frame_34.grid(row=1, column=0, columnspan=3, sticky='ew', pady=0)
            self._player_setup_frame_34.columnconfigure(1, weight=1)
            ttk.Label(self._player_setup_frame_34, text="玩家3 (手牌/范围):").grid(row=0, column=0, padx=10, pady=6, sticky='w')
            ttk.Entry(self._player_setup_frame_34, textvariable=self.p3_hand_var, font=('Segoe UI Symbol', 12)).grid(row=0, column=1, padx=10, pady=6, sticky='ew')
            ttk.Button(self._player_setup_frame_34, text="重置", command=self._reset_player3, width=8).grid(row=0, column=2, padx=5, pady=6)
            ttk.Label(self._player_setup_frame_34, text="玩家4 (手牌/范围):").grid(row=1, column=0, padx=10, pady=6, sticky='w')
            ttk.Entry(self._player_setup_frame_34, textvariable=self.p4_hand_var, font=('Segoe UI Symbol', 12)).grid(row=1, column=1, padx=10, pady=6, sticky='ew')
            ttk.Button(self._player_setup_frame_34, text="重置", command=self._reset_player4, width=8).grid(row=1, column=2, padx=5, pady=6)
            self._player_setup_frame_34.grid_remove()  # 默认显示 1、2 组

            sim_frame = ttk.Frame(player_setup_frame)
            sim_frame.grid(row=2, column=0, columnspan=3, pady=5, sticky='ew')
        elif self.num_players == 5:
            # 五人模式：手牌/范围栏三组切换（玩家1、2 | 玩家3、4 | 玩家5）
            self._player_setup_group_var = tk.IntVar(value=1)
            group_switch_row = ttk.Frame(player_setup_frame)
            group_switch_row.grid(row=0, column=0, columnspan=3, sticky='w', padx=5, pady=(5, 2))
            self._btn_players_12 = tk.Button(group_switch_row, text=" 玩家1、2 ", relief='flat', bg=self.P1_COLOR, fg='white', font=('Microsoft YaHei', 9, 'bold'), cursor='hand2', command=lambda: self._switch_player_setup_group(1), takefocus=0)
            self._btn_players_12.pack(side='left', padx=2, ipady=3)
            self._btn_players_34 = tk.Button(group_switch_row, text=" 玩家3、4 ", relief='flat', bg='#4a4a4a', fg='white', font=('Microsoft YaHei', 9, 'bold'), cursor='hand2', command=lambda: self._switch_player_setup_group(2), takefocus=0)
            self._btn_players_34.pack(side='left', padx=2, ipady=3)
            self._btn_players_5 = tk.Button(group_switch_row, text=" 玩家5 ", relief='flat', bg='#4a4a4a', fg='white', font=('Microsoft YaHei', 9, 'bold'), cursor='hand2', command=lambda: self._switch_player_setup_group(3), takefocus=0)
            self._btn_players_5.pack(side='left', padx=2, ipady=3)

            self._player_setup_frame_12 = ttk.Frame(player_setup_frame)
            self._player_setup_frame_12.grid(row=1, column=0, columnspan=3, sticky='ew', pady=0)
            self._player_setup_frame_12.columnconfigure(1, weight=1)
            ttk.Label(self._player_setup_frame_12, text="玩家1 (手牌/范围):").grid(row=0, column=0, padx=10, pady=6, sticky='w')
            ttk.Entry(self._player_setup_frame_12, textvariable=self.p1_hand_var, font=('Segoe UI Symbol', 12)).grid(row=0, column=1, padx=10, pady=6, sticky='ew')
            ttk.Button(self._player_setup_frame_12, text="重置", command=self._reset_player1, width=8).grid(row=0, column=2, padx=5, pady=6)
            ttk.Label(self._player_setup_frame_12, text="玩家2 (手牌/范围):").grid(row=1, column=0, padx=10, pady=6, sticky='w')
            ttk.Entry(self._player_setup_frame_12, textvariable=self.p2_hand_var, font=('Segoe UI Symbol', 12)).grid(row=1, column=1, padx=10, pady=6, sticky='ew')
            ttk.Button(self._player_setup_frame_12, text="重置", command=self._reset_player2, width=8).grid(row=1, column=2, padx=5, pady=6)

            self.p3_hand_var = tk.StringVar()
            self.p4_hand_var = tk.StringVar()
            self.p5_hand_var = tk.StringVar()
            self._player_setup_frame_34 = ttk.Frame(player_setup_frame)
            self._player_setup_frame_34.grid(row=1, column=0, columnspan=3, sticky='ew', pady=0)
            self._player_setup_frame_34.columnconfigure(1, weight=1)
            ttk.Label(self._player_setup_frame_34, text="玩家3 (手牌/范围):").grid(row=0, column=0, padx=10, pady=6, sticky='w')
            ttk.Entry(self._player_setup_frame_34, textvariable=self.p3_hand_var, font=('Segoe UI Symbol', 12)).grid(row=0, column=1, padx=10, pady=6, sticky='ew')
            ttk.Button(self._player_setup_frame_34, text="重置", command=self._reset_player3, width=8).grid(row=0, column=2, padx=5, pady=6)
            ttk.Label(self._player_setup_frame_34, text="玩家4 (手牌/范围):").grid(row=1, column=0, padx=10, pady=6, sticky='w')
            ttk.Entry(self._player_setup_frame_34, textvariable=self.p4_hand_var, font=('Segoe UI Symbol', 12)).grid(row=1, column=1, padx=10, pady=6, sticky='ew')
            ttk.Button(self._player_setup_frame_34, text="重置", command=self._reset_player4, width=8).grid(row=1, column=2, padx=5, pady=6)
            self._player_setup_frame_34.grid_remove()

            self._player_setup_frame_5 = ttk.Frame(player_setup_frame)
            self._player_setup_frame_5.grid(row=1, column=0, columnspan=3, sticky='ew', pady=0)
            self._player_setup_frame_5.columnconfigure(1, weight=1)
            ttk.Label(self._player_setup_frame_5, text="玩家5 (手牌/范围):").grid(row=0, column=0, padx=10, pady=6, sticky='w')
            ttk.Entry(self._player_setup_frame_5, textvariable=self.p5_hand_var, font=('Segoe UI Symbol', 12)).grid(row=0, column=1, padx=10, pady=6, sticky='ew')
            ttk.Button(self._player_setup_frame_5, text="重置", command=self._reset_player5, width=8).grid(row=0, column=2, padx=5, pady=6)
            self._player_setup_frame_5.grid_remove()

            sim_frame = ttk.Frame(player_setup_frame)
            sim_frame.grid(row=2, column=0, columnspan=3, pady=5, sticky='ew')
        else:
            # 两人或三人：直接显示所有玩家行
            ttk.Label(player_setup_frame, text="玩家1 (手牌/范围):").grid(row=0, column=0, padx=10, pady=8, sticky='w')
            ttk.Entry(player_setup_frame, textvariable=self.p1_hand_var, font=('Segoe UI Symbol', 12)).grid(row=0, column=1, padx=10, pady=8, sticky='ew')
            ttk.Button(player_setup_frame, text="重置", command=self._reset_player1, width=8).grid(row=0, column=2, padx=5, pady=8)
            ttk.Label(player_setup_frame, text="玩家2 (手牌/范围):").grid(row=1, column=0, padx=10, pady=8, sticky='w')
            ttk.Entry(player_setup_frame, textvariable=self.p2_hand_var, font=('Segoe UI Symbol', 12)).grid(row=1, column=1, padx=10, pady=8, sticky='ew')
            ttk.Button(player_setup_frame, text="重置", command=self._reset_player2, width=8).grid(row=1, column=2, padx=5, pady=8)
            if self.num_players >= 3:
                self.p3_hand_var = tk.StringVar()
                ttk.Label(player_setup_frame, text="玩家3 (手牌/范围):").grid(row=2, column=0, padx=10, pady=8, sticky='w')
                ttk.Entry(player_setup_frame, textvariable=self.p3_hand_var, font=('Segoe UI Symbol', 12)).grid(row=2, column=1, padx=10, pady=8, sticky='ew')
                ttk.Button(player_setup_frame, text="重置", command=self._reset_player3, width=8).grid(row=2, column=2, padx=5, pady=8)
            if self.num_players >= 4:
                self.p4_hand_var = tk.StringVar()
                ttk.Label(player_setup_frame, text="玩家4 (手牌/范围):").grid(row=3, column=0, padx=10, pady=8, sticky='w')
                ttk.Entry(player_setup_frame, textvariable=self.p4_hand_var, font=('Segoe UI Symbol', 12)).grid(row=3, column=1, padx=10, pady=8, sticky='ew')
                ttk.Button(player_setup_frame, text="重置", command=self._reset_player4, width=8).grid(row=3, column=2, padx=5, pady=8)
            sim_frame = ttk.Frame(player_setup_frame)
            sim_frame.grid(row=self.num_players, column=0, columnspan=3, pady=5, sticky='ew')
        ttk.Label(sim_frame, text="模拟次数:").pack(side='left', padx=10)
        self.num_simulations_var = tk.StringVar(value="50000")
        ttk.Entry(sim_frame, textvariable=self.num_simulations_var, width=15).pack(side='left')

        self._create_board_selector(parent_pane)
        self._create_strength_display(parent_pane) 
        
        action_frame = ttk.Frame(parent_pane)
        action_frame.pack(side='bottom', pady=10, fill='x')
        
        initial_theme = random.choice(self.pb_themes)
        self.analysis_progress_bar = GradientProgressBar(action_frame, color_list=initial_theme, height=18)
        self.analysis_progress_bar.pack(fill='x', pady=(0, 8))
        self.progress_var.trace_add('write', lambda *args: self.analysis_progress_bar.set_value(self.progress_var.get()))
        
        self.calc_button = ttk.Button(action_frame, text="开始分析", command=self.run_analysis_thread)
        self.calc_button.pack(fill='x', ipady=10, pady=5)
        ttk.Button(action_frame, text="清空全部", command=self.clear_all).pack(fill='x', ipady=10, pady=5)
        
        self.open_chart_button = ttk.Button(parent_pane, text="查看起手牌强度图表", command=self._open_strength_chart)
        self.open_chart_button.pack(side='bottom', fill='x', ipady=10, pady=5)

    def _create_analysis_pane(self, parent_pane):
        parent_pane.rowconfigure(1, weight=1)
        equity_frame = ttk.LabelFrame(parent_pane, text="胜率分析 (Equity)")
        equity_frame.pack(fill='x', pady=10, anchor='n')
        result_grid = ttk.Frame(equity_frame, padding=10)
        result_grid.pack(fill='x', expand=True)
        result_grid.columnconfigure(1, weight=1)

        ttk.Label(result_grid, text="玩家1:", font=('Microsoft YaHei', 11, 'bold')).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.p1_win_bar = ttk.Progressbar(result_grid, style="p1.Horizontal.TProgressbar")
        self.p1_win_bar.grid(row=0, column=1, sticky='ew', padx=5)
        self.p1_win_var = tk.StringVar(value="N/A")
        ttk.Label(result_grid, textvariable=self.p1_win_var, font=('Microsoft YaHei', 11, 'bold')).grid(row=0, column=2, sticky='e', padx=10)

        ttk.Label(result_grid, text="玩家2:", font=('Microsoft YaHei', 11, 'bold')).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.p2_win_bar = ttk.Progressbar(result_grid, style="p2.Horizontal.TProgressbar")
        self.p2_win_bar.grid(row=1, column=1, sticky='ew', padx=5)
        self.p2_win_var = tk.StringVar(value="N/A")
        ttk.Label(result_grid, textvariable=self.p2_win_var, font=('Microsoft YaHei', 11, 'bold')).grid(row=1, column=2, sticky='e', padx=10)

        self.p3_win_bar = None
        self.p3_win_var = None
        self.p4_win_bar = None
        self.p4_win_var = None
        self.p5_win_bar = None
        self.p5_win_var = None
        if self.num_players >= 3:
            ttk.Label(result_grid, text="玩家3:", font=('Microsoft YaHei', 11, 'bold')).grid(row=2, column=0, sticky='w', padx=5, pady=5)
            self.p3_win_bar = ttk.Progressbar(result_grid, style="p3.Horizontal.TProgressbar")
            self.p3_win_bar.grid(row=2, column=1, sticky='ew', padx=5)
            self.p3_win_var = tk.StringVar(value="N/A")
            ttk.Label(result_grid, textvariable=self.p3_win_var, font=('Microsoft YaHei', 11, 'bold')).grid(row=2, column=2, sticky='e', padx=10)
        if self.num_players >= 4:
            ttk.Label(result_grid, text="玩家4:", font=('Microsoft YaHei', 11, 'bold')).grid(row=3, column=0, sticky='w', padx=5, pady=5)
            self.p4_win_bar = ttk.Progressbar(result_grid, style="p4.Horizontal.TProgressbar")
            self.p4_win_bar.grid(row=3, column=1, sticky='ew', padx=5)
            self.p4_win_var = tk.StringVar(value="N/A")
            ttk.Label(result_grid, textvariable=self.p4_win_var, font=('Microsoft YaHei', 11, 'bold')).grid(row=3, column=2, sticky='e', padx=10)
        if self.num_players >= 5:
            ttk.Label(result_grid, text="玩家5:", font=('Microsoft YaHei', 11, 'bold')).grid(row=4, column=0, sticky='w', padx=5, pady=5)
            self.p5_win_bar = ttk.Progressbar(result_grid, style="p5.Horizontal.TProgressbar")
            self.p5_win_bar.grid(row=4, column=1, sticky='ew', padx=5)
            self.p5_win_var = tk.StringVar(value="N/A")
            ttk.Label(result_grid, textvariable=self.p5_win_var, font=('Microsoft YaHei', 11, 'bold')).grid(row=4, column=2, sticky='e', padx=10)

        tie_row = self.num_players
        self.tie_bar = ttk.Progressbar(result_grid, style="tie.Horizontal.TProgressbar")
        self.tie_var = tk.StringVar(value="N/A")
        # 五人模式不显示平局行（胜率已按人头均分，无单独平局条）
        if self.num_players != 5:
            ttk.Label(result_grid, text="平局:", font=('Microsoft YaHei', 11, 'bold')).grid(row=tie_row, column=0, sticky='w', padx=5, pady=5)
            self.tie_bar.grid(row=tie_row, column=1, sticky='ew', padx=5)
            ttk.Label(result_grid, textvariable=self.tie_var, font=('Microsoft YaHei', 11, 'bold')).grid(row=tie_row, column=2, sticky='e', padx=10)

        self._create_range_selector(parent_pane)
    
    def _create_board_selector(self, parent_frame):
        board_frame = ttk.LabelFrame(parent_frame, text="公共牌选择")
        board_frame.pack(fill='x', pady=10)
        display_frame = ttk.Frame(board_frame, relief='solid', borderwidth=1)
        display_frame.pack(fill='x', pady=5, padx=10)
        self.board_display_var = tk.StringVar(value="已选公共牌: ")
        ttk.Label(display_frame, textvariable=self.board_display_var, font=("Segoe UI Symbol", 12, "bold")).pack(pady=5, side='left', padx=10)
        ttk.Button(display_frame, text="重置", command=self._reset_board_selector).pack(side='right', padx=5, pady=5)
        card_pool_frame = ttk.Frame(board_frame)
        card_pool_frame.pack(pady=5, padx=10)
        self.board_card_buttons = {}
        suits_map = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
        suit_colors = {'s': 'black', 'h': 'red', 'd': 'blue', 'c': 'green'}
        ranks = 'AKQJT98765432'
        
        def create_board_hover_effect(btn, card_str):
            btn.animation_data = {'card_str': card_str, 'original_color': '#d0d0d0', 'is_hovering': False, 'current_step': 0, 'timer_id': None}
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            def rgb_to_hex(rgb): return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            def interpolate_color(color1, color2, factor):
                rgb1 = hex_to_rgb(color1)
                rgb2 = hex_to_rgb(color2)
                result = tuple(rgb1[i] + (rgb2[i] - rgb1[i]) * factor for i in range(3))
                return rgb_to_hex(result)
            def get_original_color():
                card = btn.animation_data['card_str']
                if card in self.board_cards: return '#555555'
                else: return '#d0d0d0'
            def animate_color():
                data = btn.animation_data
                total_steps = 15 
                if str(btn['state']) == 'disabled': return 
                original_color = get_original_color()
                if data['is_hovering']:
                    if data['current_step'] < total_steps:
                        data['current_step'] += 1
                        factor = data['current_step'] / total_steps
                        target = '#ffffff'
                        new_color = interpolate_color(original_color, target, factor * 0.35)
                        btn.configure(bg=new_color)
                        data['timer_id'] = btn.after(20, animate_color)
                else:
                    if data['current_step'] > 0:
                        data['current_step'] -= 1
                        factor = data['current_step'] / total_steps
                        target = '#ffffff'
                        new_color = interpolate_color(original_color, target, factor * 0.35)
                        btn.configure(bg=new_color)
                        data['timer_id'] = btn.after(20, animate_color)
                    else:
                        btn.configure(bg=original_color)
            def on_enter(event):
                if str(btn['state']) == 'disabled': return
                data = btn.animation_data
                if data['timer_id']: btn.after_cancel(data['timer_id'])
                data['is_hovering'] = True
                animate_color()
            def on_leave(event):
                data = btn.animation_data
                if data['timer_id']: btn.after_cancel(data['timer_id'])
                data['is_hovering'] = False
                animate_color()
            btn.bind('<Enter>', on_enter)
            btn.bind('<Leave>', on_leave)
        
        for i, suit_char in enumerate('shdc'):
            for j, rank_char in enumerate(ranks):
                card_str = f"{rank_char}{suit_char}"
                display_text = f"{rank_char}{suits_map[suit_char]}"
                btn = tk.Button(card_pool_frame, text=display_text, font=('Arial', 10, 'bold'), 
                            width=4, fg=suit_colors[suit_char], bg='#d0d0d0', relief='raised', 
                            command=lambda s=card_str: self._on_board_card_select(s), takefocus=0)
                btn.grid(row=i, column=j, padx=1, pady=1)
                self.board_card_buttons[card_str] = btn
                create_board_hover_effect(btn, card_str)

    def _create_strength_display(self, parent_frame):
        strength_frame = ttk.LabelFrame(parent_frame, text="玩家1 牌力分布")
        strength_frame.pack(fill='both', expand=True, pady=10)
        cols = ('牌型', '进度条', '概率'); self.strength_tree = ttk.Treeview(strength_frame, columns=cols, show='headings', height=9)
        self.strength_tree.heading('牌型', text='牌型'); self.strength_tree.heading('进度条', text='分布可视化'); self.strength_tree.heading('概率', text='概率')
        self.strength_tree.column('牌型', width=150, anchor='center'); self.strength_tree.column('进度条', width=190, anchor='center'); self.strength_tree.column('概率', width=100, anchor='e')
        self.strength_tree.pack(fill='both', expand=True, padx=5, pady=5)

    def _create_range_selector(self, parent_frame):
        range_frame = ttk.LabelFrame(parent_frame, text="起手牌范围选择器")
        range_frame.pack(fill='both', expand=True, pady=5)
        radio_frame = ttk.Frame(range_frame); radio_frame.pack(anchor='w', padx=10, pady=5)
        self.active_player_for_range = tk.IntVar(value=1)
        self.p1_radio_btn = tk.Button(radio_frame, text="为玩家1选择", relief='flat', bg=self.P1_COLOR, fg='white', font=('Microsoft YaHei', 9, 'bold'), borderwidth=0, activebackground='#0056b3', activeforeground='white', command=lambda: self._select_player_for_range(1), takefocus=0)
        self.p1_radio_btn.pack(side='left', padx=5, ipady=4)
        self.p2_radio_btn = tk.Button(radio_frame, text="为玩家2选择", relief='flat', bg='#4a4a4a', fg='white', font=('Microsoft YaHei', 9, 'bold'), borderwidth=0, activebackground='#6a6a6a', activeforeground='white', command=lambda: self._select_player_for_range(2), takefocus=0)
        self.p2_radio_btn.pack(side='left', padx=5, ipady=4)
        self.p3_radio_btn = None
        self.p4_radio_btn = None
        self.p5_radio_btn = None
        if self.num_players >= 3:
            self.p3_radio_btn = tk.Button(radio_frame, text="为玩家3选择", relief='flat', bg='#4a4a4a', fg='white', font=('Microsoft YaHei', 9, 'bold'), borderwidth=0, activebackground='#1e7e34', activeforeground='white', command=lambda: self._select_player_for_range(3), takefocus=0)
            self.p3_radio_btn.pack(side='left', padx=5, ipady=4)
        if self.num_players >= 4:
            self.p4_radio_btn = tk.Button(radio_frame, text="为玩家4选择", relief='flat', bg='#4a4a4a', fg='white', font=('Microsoft YaHei', 9, 'bold'), borderwidth=0, activebackground='#c9182a', activeforeground='white', command=lambda: self._select_player_for_range(4), takefocus=0)
            self.p4_radio_btn.pack(side='left', padx=5, ipady=4)
        if self.num_players >= 5:
            self.p5_radio_btn = tk.Button(radio_frame, text="为玩家5选择", relief='flat', bg='#4a4a4a', fg='white', font=('Microsoft YaHei', 9, 'bold'), borderwidth=0, activebackground='#e85d04', activeforeground='white', command=lambda: self._select_player_for_range(5), takefocus=0)
            self.p5_radio_btn.pack(side='left', padx=5, ipady=4)

        grid_frame = ttk.Frame(range_frame); grid_frame.pack(pady=10, padx=10)
        self.range_buttons = {}
        ranks = 'AKQJT98765432'
        btn_font = font.Font(family='Microsoft YaHei', size=9, weight='bold')
        
        def create_hover_effect(btn, hand_text):
            btn.animation_data = {'hand_text': hand_text, 'is_hovering': False, 'current_step': 0, 'timer_id': None}
            def hex_to_rgb(hex_color):
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            def rgb_to_hex(rgb): return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            def interpolate_color(color1, color2, factor):
                rgb1 = hex_to_rgb(color1)
                rgb2 = hex_to_rgb(color2)
                result = tuple(rgb1[i] + (rgb2[i] - rgb1[i]) * factor for i in range(3))
                return rgb_to_hex(result)
            def get_original_color():
                hand = btn.animation_data['hand_text']
                def get_categories(hand_var):
                    text = hand_var.get()
                    items = [s.strip() for s in text.split(',') if s.strip()]
                    categories = set()
                    for item in items:
                        if len(item) <= 3: categories.add(item)
                        else:
                            cat = self._specific_to_range_category(item)
                            if cat: categories.add(cat)
                    return categories
                p1_cats = get_categories(self.p1_hand_var)
                p2_cats = get_categories(self.p2_hand_var)
                p3_cats = get_categories(self.p3_hand_var) if self.p3_hand_var is not None else set()
                p4_cats = get_categories(self.p4_hand_var) if self.p4_hand_var is not None else set()
                p5_cats = get_categories(self.p5_hand_var) if self.p5_hand_var is not None else set()
                in_p1 = hand in p1_cats
                in_p2 = hand in p2_cats
                in_p3 = hand in p3_cats
                in_p4 = hand in p4_cats
                in_p5 = hand in p5_cats
                any_overlap = (in_p1 and in_p2 or in_p1 and in_p3 or in_p1 and in_p4 or in_p1 and in_p5 or in_p2 and in_p3 or in_p2 and in_p4 or in_p2 and in_p5 or in_p3 and in_p4 or in_p3 and in_p5 or in_p4 and in_p5)
                if any_overlap: return self.BOTH_COLOR
                elif in_p1: return self.P1_COLOR
                elif in_p2: return self.P2_COLOR
                elif in_p3: return self.P3_COLOR
                elif in_p4: return self.P4_COLOR
                elif in_p5: return self.P5_COLOR
                else:
                    if len(hand) == 2: return self.PAIR_BG
                    elif hand.endswith('s'): return self.SUITED_BG
                    else: return self.DEFAULT_BG
            
            def animate_color():
                data = btn.animation_data
                total_steps = 15 
                if data['is_hovering']:
                    if 'animation_start_color' not in data: data['animation_start_color'] = get_original_color()
                    original_color = data['animation_start_color']
                    if data['current_step'] < total_steps:
                        data['current_step'] += 1
                        factor = data['current_step'] / total_steps
                        target = '#ffffff'
                        new_color = interpolate_color(original_color, target, factor * 0.3)
                        btn.configure(bg=new_color)
                        data['timer_id'] = btn.after(20, animate_color)
                else:
                    current_correct_original_color = get_original_color()
                    if data['current_step'] > 0:
                        data['current_step'] -= 1
                        factor = data['current_step'] / total_steps
                        target = '#ffffff'
                        new_color = interpolate_color(current_correct_original_color, target, factor * 0.3)
                        btn.configure(bg=new_color)
                        data['timer_id'] = btn.after(20, animate_color)
                    else:
                        btn.configure(bg=current_correct_original_color)
                        if 'animation_start_color' in data: del data['animation_start_color'] 
            def on_enter(event):
                data = btn.animation_data
                if data['timer_id']: btn.after_cancel(data['timer_id'])
                data['is_hovering'] = True
                data['animation_start_color'] = get_original_color()
                animate_color()
            def on_leave(event):
                data = btn.animation_data
                if data['timer_id']: btn.after_cancel(data['timer_id'])
                data['is_hovering'] = False
                animate_color()
            btn.bind('<Enter>', on_enter)
            btn.bind('<Leave>', on_leave)
        
        for r in range(len(ranks)):
            for c in range(len(ranks)):
                if r < c: text, bg_color = f"{ranks[r]}{ranks[c]}s", self.SUITED_BG
                elif c < r: text, bg_color = f"{ranks[c]}{ranks[r]}o", self.DEFAULT_BG
                else: text, bg_color = f"{ranks[r]}{ranks[c]}", self.PAIR_BG
                btn = tk.Button(grid_frame, text=text, width=5, height=2, relief='raised', font=btn_font, fg='white', bg=bg_color, command=lambda t=text: self.toggle_range_button(t), takefocus=0)
                btn.grid(row=r, column=c, padx=1, pady=1)
                self.range_buttons[text] = btn
                create_hover_effect(btn, text)

    def _switch_player_setup_group(self, group):
        """四人/五人模式：切换玩家设置栏显示的组。四人：1=玩家1、2，2=玩家3、4。五人：1=玩家1、2，2=玩家3、4，3=玩家5。"""
        if self.num_players not in (4, 5) or not hasattr(self, '_player_setup_frame_12'):
            return
        if self.num_players == 5 and hasattr(self, '_player_setup_frame_5'):
            # 五人模式：三组
            self._player_setup_frame_12.grid_remove()
            self._player_setup_frame_34.grid_remove()
            self._player_setup_frame_5.grid_remove()
            self._btn_players_12.config(bg='#4a4a4a')
            self._btn_players_34.config(bg='#4a4a4a')
            self._btn_players_5.config(bg='#4a4a4a')
            if group == 1:
                self._player_setup_frame_12.grid(row=1, column=0, columnspan=3, sticky='ew', pady=0)
                self._btn_players_12.config(bg=self.P1_COLOR)
            elif group == 2:
                self._player_setup_frame_34.grid(row=1, column=0, columnspan=3, sticky='ew', pady=0)
                self._btn_players_34.config(bg=self.P4_COLOR)
            else:
                self._player_setup_frame_5.grid(row=1, column=0, columnspan=3, sticky='ew', pady=0)
                self._btn_players_5.config(bg=self.P5_COLOR)
            self._player_setup_group_var.set(group)
        else:
            # 四人模式：两组
            if group == 1:
                self._player_setup_frame_34.grid_remove()
                self._player_setup_frame_12.grid(row=1, column=0, columnspan=3, sticky='ew', pady=0)
                self._player_setup_group_var.set(1)
                self._btn_players_12.config(bg=self.P1_COLOR)
                self._btn_players_34.config(bg='#4a4a4a')
            else:
                self._player_setup_frame_12.grid_remove()
                self._player_setup_frame_34.grid(row=1, column=0, columnspan=3, sticky='ew', pady=0)
                self._player_setup_group_var.set(2)
                self._btn_players_12.config(bg='#4a4a4a')
                self._btn_players_34.config(bg=self.P4_COLOR)

    def _reset_player1(self):
        self.p1_hand_var.set("")
        self._update_range_grid_colors()
    
    def _reset_player2(self):
        self.p2_hand_var.set("")
        self._update_range_grid_colors()

    def _reset_player3(self):
        if self.p3_hand_var is not None:
            self.p3_hand_var.set("")
            self._update_range_grid_colors()

    def _reset_player4(self):
        if self.p4_hand_var is not None:
            self.p4_hand_var.set("")
            self._update_range_grid_colors()

    def _reset_player5(self):
        if self.p5_hand_var is not None:
            self.p5_hand_var.set("")
            self._update_range_grid_colors()
    
    def _select_player_for_range(self, player_num):
        self.active_player_for_range.set(player_num)
        self.p1_radio_btn.config(bg=self.P1_COLOR if player_num == 1 else '#4a4a4a')
        self.p2_radio_btn.config(bg=self.P2_COLOR if player_num == 2 else '#4a4a4a')
        if getattr(self, 'p3_radio_btn', None) is not None:
            self.p3_radio_btn.config(bg=self.P3_COLOR if player_num == 3 else '#4a4a4a')
        if getattr(self, 'p4_radio_btn', None) is not None:
            self.p4_radio_btn.config(bg=self.P4_COLOR if player_num == 4 else '#4a4a4a')
        if getattr(self, 'p5_radio_btn', None) is not None:
            self.p5_radio_btn.config(bg=self.P5_COLOR if player_num == 5 else '#4a4a4a')

    def _on_board_card_select(self, card_str):
        if card_str in self.board_cards:
            # 如果牌已被选中，则取消选择
            self.board_cards.remove(card_str)
            self.board_card_buttons[card_str].config(state='normal', relief='raised', bg='#d0d0d0')
            self._update_board_display()
        elif len(self.board_cards) < 5:
            # 如果牌未被选中且公共牌少于5张，则添加
            self.board_cards.append(card_str)
            self.board_card_buttons[card_str].config(state='normal', relief='sunken', bg='#555')
            self._update_board_display()

    def _format_card_display(self, card_str):
        """将牌的字母格式转换为图案格式，如 'As' -> 'A♠'"""
        suits_map = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
        if len(card_str) == 2:
            rank = card_str[0]
            suit = card_str[1].lower()
            return f"{rank}{suits_map.get(suit, suit)}"
        return card_str

    def _format_hand_display(self, hand_str):
        """将手牌范围字符串中的字母花色转换为图案显示格式
        例如: 'ASAH' -> 'A♠A♥', 'AKs' -> 'AKs', 'AKo' -> 'AKo'
        """
        suits_map = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣',
                     's': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
        hand_str = hand_str.strip()
        
        # 对于4字符的具体手牌(如ASAH, AkHd)
        if len(hand_str) == 4:
            r1 = hand_str[0].upper()
            s1 = hand_str[1]
            r2 = hand_str[2].upper()
            s2 = hand_str[3]
            s1_display = suits_map.get(s1, s1)
            s2_display = suits_map.get(s2, s2)
            return f"{r1}{s1_display}{r2}{s2_display}"
        
        # 对于范围字符串(如AKs, AKo, AA)保持原样
        return hand_str

    def _parse_hand_from_display(self, display_str):
        """将图案格式转换回字母格式用于后端解析
        例如: 'A♠A♥' -> 'ASAH', 'AKs' -> 'AKs'
        """
        suits_reverse = {'♠': 'S', '♥': 'H', '♦': 'D', '♣': 'C'}
        result = display_str.strip()
        for symbol, letter in suits_reverse.items():
            result = result.replace(symbol, letter)
        return result

    def _update_board_display(self):
        """更新公共牌显示区域"""
        if self.board_cards:
            formatted_cards = [self._format_card_display(c) for c in self.board_cards]
            self.board_display_var.set(f"已选公共牌: {' '.join(formatted_cards)}")
        else:
            self.board_display_var.set("已选公共牌: ")

    def _reset_board_selector(self):
        self.board_cards = []
        for btn in self.board_card_buttons.values(): btn.config(state='normal', relief='raised', bg='#d0d0d0')
        self.board_display_var.set("已选公共牌: ")
    
    def toggle_range_button(self, hand_text):
        active_player = self.active_player_for_range.get()
        if active_player == 1:
            entry_var = self.p1_hand_var
        elif active_player == 2:
            entry_var = self.p2_hand_var
        elif active_player == 3 and self.p3_hand_var is not None:
            entry_var = self.p3_hand_var
        elif active_player == 4 and self.p4_hand_var is not None:
            entry_var = self.p4_hand_var
        elif active_player == 5 and self.p5_hand_var is not None:
            entry_var = self.p5_hand_var
        else:
            entry_var = self.p2_hand_var
        current_text = entry_var.get()
        current_items_list = [s.strip() for s in current_text.split(',') if s.strip()]
        current_items_set = set(current_items_list)

        is_already_selected = False
        if hand_text in current_items_set:
            is_already_selected = True
        else:
            for item in current_items_set:
                if self._specific_to_range_category(item) == hand_text:
                    is_already_selected = True
                    break

        if is_already_selected:
            new_items_list = []
            for item in current_items_list:
                if item == hand_text: continue 
                if self._specific_to_range_category(item) == hand_text: continue 
                new_items_list.append(item) 

            entry_var.set(", ".join(new_items_list))
            self._update_range_grid_colors()

        else:
            def handle_selection_callback(result_hand_str):
                if result_hand_str is None: return 

                active_player_inner = self.active_player_for_range.get()
                if active_player_inner == 1:
                    entry_var_inner = self.p1_hand_var
                elif active_player_inner == 2:
                    entry_var_inner = self.p2_hand_var
                elif active_player_inner == 3 and self.p3_hand_var is not None:
                    entry_var_inner = self.p3_hand_var
                elif active_player_inner == 4 and self.p4_hand_var is not None:
                    entry_var_inner = self.p4_hand_var
                elif active_player_inner == 5 and self.p5_hand_var is not None:
                    entry_var_inner = self.p5_hand_var
                else:
                    entry_var_inner = self.p2_hand_var
                current_text_inner = entry_var_inner.get()
                current_items_list_inner = [s.strip() for s in current_text_inner.split(',') if s.strip()]
                current_items_set_inner = set(current_items_list_inner)
                new_hands_to_process = [h.strip() for h in result_hand_str.split(',') if h.strip()]
                
                for hand_item in new_hands_to_process:
                    new_hand = hand_item
                    # 将4字符手牌转换为图案显示格式
                    if len(hand_item) == 4:
                        new_hand = self._format_hand_display(hand_item.upper())
                    
                    if new_hand in current_items_set_inner:
                        current_items_set_inner.remove(new_hand)
                    else:
                        if len(new_hand) <= 3:
                            parent_range = new_hand
                            items_to_remove = set()
                            for item in current_items_set_inner:
                                if self._specific_to_range_category(item) == parent_range:
                                    items_to_remove.add(item)
                            current_items_set_inner.difference_update(items_to_remove)
                            current_items_set_inner.add(parent_range)
                        else:
                            parent_range = self._specific_to_range_category(new_hand) 
                            if parent_range in current_items_set_inner:
                                current_items_set_inner.remove(parent_range)
                                current_items_set_inner.add(new_hand)
                            else:
                                current_items_set_inner.add(new_hand)

                final_items = sorted(list(current_items_set_inner))
                entry_var_inner.set(", ".join(final_items))
                self.after(50, self._update_range_grid_colors)

            SuitSelectorWindow(self, hand_text, handle_selection_callback)

    def _update_range_grid_colors(self):
        def get_all_selected_categories(hand_var):
            text = hand_var.get()
            items = [s.strip() for s in text.split(',') if s.strip()] 
            categories = set()
            for item in items:
                if len(item) <= 3: categories.add(item) 
                else:
                    cat = self._specific_to_range_category(item) 
                    if cat: categories.add(cat)
            return categories

        p1_categories = get_all_selected_categories(self.p1_hand_var)
        p2_categories = get_all_selected_categories(self.p2_hand_var)
        p3_categories = get_all_selected_categories(self.p3_hand_var) if self.p3_hand_var is not None else set()
        p4_categories = get_all_selected_categories(self.p4_hand_var) if self.p4_hand_var is not None else set()
        p5_categories = get_all_selected_categories(self.p5_hand_var) if self.p5_hand_var is not None else set()

        for hand, btn in self.range_buttons.items():
            in_p1 = hand in p1_categories
            in_p2 = hand in p2_categories
            in_p3 = hand in p3_categories
            in_p4 = hand in p4_categories
            in_p5 = hand in p5_categories
            any_overlap = (in_p1 and in_p2 or in_p1 and in_p3 or in_p1 and in_p4 or in_p1 and in_p5 or in_p2 and in_p3 or in_p2 and in_p4 or in_p2 and in_p5 or in_p3 and in_p4 or in_p3 and in_p5 or in_p4 and in_p5)
            if any_overlap: btn.config(bg=self.BOTH_COLOR, relief='sunken')
            elif in_p1: btn.config(bg=self.P1_COLOR, relief='sunken')
            elif in_p2: btn.config(bg=self.P2_COLOR, relief='sunken')
            elif in_p3: btn.config(bg=self.P3_COLOR, relief='sunken')
            elif in_p4: btn.config(bg=self.P4_COLOR, relief='sunken')
            elif in_p5: btn.config(bg=self.P5_COLOR, relief='sunken')
            else:
                if len(hand) == 2: bg = self.PAIR_BG
                elif hand.endswith('s'): bg = self.SUITED_BG
                else: bg = self.DEFAULT_BG
                btn.config(bg=bg, relief='raised')

    def clear_all(self):
        self._reset_player1()
        self._reset_player2()
        if self.p3_hand_var is not None:
            self._reset_player3()
        if self.p4_hand_var is not None:
            self._reset_player4()
        if self.p5_hand_var is not None:
            self._reset_player5()
        self._reset_board_selector()
        for i in self.strength_tree.get_children(): self.strength_tree.delete(i)
        self._reset_equity_display()
                
    def _create_probability_bar(self, probability):
        bar_length = 80  
        filled = int(bar_length * probability / 100)
        bar = '|' * filled + ' ' * (bar_length - filled)
        return bar

    def _reset_equity_display(self):
        self.p1_win_var.set("N/A"); self.p1_win_bar['value'] = 0
        self.p2_win_var.set("N/A"); self.p2_win_bar['value'] = 0
        if self.p3_win_var is not None:
            self.p3_win_var.set("N/A"); self.p3_win_bar['value'] = 0
        if self.p4_win_var is not None:
            self.p4_win_var.set("N/A"); self.p4_win_bar['value'] = 0
        if self.p5_win_var is not None:
            self.p5_win_var.set("N/A"); self.p5_win_bar['value'] = 0
        self.tie_var.set("N/A"); self.tie_bar['value'] = 0
        self.progress_var.set(0)

    def run_analysis_thread(self):
        try:
            num_sims = int(self.num_simulations_var.get())
            if num_sims <= 0: 
                if num_sims == 0:
                    self._reset_equity_display()
                    for i in self.strength_tree.get_children(): self.strength_tree.delete(i)
                    return
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "模拟次数必须是一个非负整数。")
            return

        self._reset_equity_display()
        new_theme = random.choice(self.pb_themes)
        self.analysis_progress_bar.set_colors(new_theme)
        max_val = num_sims if num_sims > 0 else 1
        self.analysis_progress_bar.set_max(max_val)
        
        self.calc_button.config(state='disabled')
        for i in self.strength_tree.get_children(): self.strength_tree.delete(i)
        
        self.analysis_result = None
        self.analysis_thread = threading.Thread(target=self.run_analysis_calculation, daemon=True)
        self.analysis_thread.start()
        self.check_analysis_thread()

    def check_analysis_thread(self):
        if self.analysis_thread.is_alive():
            self.after(100, self.check_analysis_thread)
        else:
            try:
                if isinstance(self.analysis_result, Exception): raise self.analysis_result
                equity, strength, show_strength = self.analysis_result
                
                self.p1_win_var.set(f"{equity['p1_win']:.2f}%")
                self.p1_win_bar['value'] = equity['p1_win']
                self.p2_win_var.set(f"{equity['p2_win']:.2f}%")
                self.p2_win_bar['value'] = equity['p2_win']
                if 'p3_win' in equity or 'p4_win' in equity or 'p5_win' in equity:
                    if self.p3_win_var is not None and 'p3_win' in equity:
                        self.p3_win_var.set(f"{equity['p3_win']:.2f}%")
                        self.p3_win_bar['value'] = equity['p3_win']
                    if self.p4_win_var is not None and 'p4_win' in equity:
                        self.p4_win_var.set(f"{equity['p4_win']:.2f}%")
                        self.p4_win_bar['value'] = equity['p4_win']
                    if self.p5_win_var is not None and 'p5_win' in equity:
                        self.p5_win_var.set(f"{equity['p5_win']:.2f}%")
                        self.p5_win_bar['value'] = equity['p5_win']
                    self.tie_var.set("0.00%"); self.tie_bar['value'] = 0
                else:
                    self.tie_var.set(f"{equity['tie']:.2f}%")
                    self.tie_bar['value'] = equity['tie']
                
                if int(self.num_simulations_var.get()) == 0: self.progress_var.set(0)
                else: self.progress_var.set(self.analysis_progress_bar.max_val)

                if show_strength:
                    hand_rank_order = {v: k for k, v in self.poker_logic.rank_class_to_string_map.items()}
                    all_hand_types = sorted(hand_rank_order.keys(), key=lambda x: hand_rank_order[x])
                    for hand_name in all_hand_types:
                        prob = strength.get(hand_name, 0.0)
                        if prob > 1e-5: 
                            progress_bar = self._create_probability_bar(prob)
                            self.strength_tree.insert('', tk.END, values=(hand_name, progress_bar, f"{prob:.2f}%"))
                else:
                    self.strength_tree.insert('', tk.END, values=("(请输入玩家1手牌/范围)", "", "N/A"))
            except Exception as e:
                messagebox.showerror("分析出错", f"错误: {e}")
                self._reset_equity_display()
            finally:
                self.calc_button.config(state='normal')

    def run_analysis_calculation(self):
        try:
            # 将图案格式转换回字母格式用于后端解析
            p1_input = [self._parse_hand_from_display(s.strip()).upper() for s in self.p1_hand_var.get().split(',') if s.strip()]
            p2_input = [self._parse_hand_from_display(s.strip()).upper() for s in self.p2_hand_var.get().split(',') if s.strip()]
            board_input = "".join(self.board_cards)
            num_sims = int(self.num_simulations_var.get())
            
            def progress_update(current_sim):
                max_sims = self.analysis_progress_bar.max_val      
                self.progress_var.set(min(current_sim, max_sims))

            if self.num_players == 3:
                p3_input = [self._parse_hand_from_display(s.strip()).upper() for s in self.p3_hand_var.get().split(',') if s.strip()] if self.p3_hand_var else []
                self.analysis_result = self.poker_logic.run_analysis_3way(
                    p1_input, p2_input, p3_input, board_input,
                    num_simulations=num_sims,
                    progress_callback=progress_update
                )
            elif self.num_players == 4:
                p3_input = [self._parse_hand_from_display(s.strip()).upper() for s in self.p3_hand_var.get().split(',') if s.strip()] if self.p3_hand_var else []
                p4_input = [self._parse_hand_from_display(s.strip()).upper() for s in self.p4_hand_var.get().split(',') if s.strip()] if self.p4_hand_var else []
                self.analysis_result = self.poker_logic.run_analysis_4way(
                    p1_input, p2_input, p3_input, p4_input, board_input,
                    num_simulations=num_sims,
                    progress_callback=progress_update
                )
            elif self.num_players == 5:
                p3_input = [self._parse_hand_from_display(s.strip()).upper() for s in self.p3_hand_var.get().split(',') if s.strip()] if self.p3_hand_var else []
                p4_input = [self._parse_hand_from_display(s.strip()).upper() for s in self.p4_hand_var.get().split(',') if s.strip()] if self.p4_hand_var else []
                p5_input = [self._parse_hand_from_display(s.strip()).upper() for s in self.p5_hand_var.get().split(',') if s.strip()] if self.p5_hand_var else []
                self.analysis_result = self.poker_logic.run_analysis_5way(
                    p1_input, p2_input, p3_input, p4_input, p5_input, board_input,
                    num_simulations=num_sims,
                    progress_callback=progress_update
                )
            else:
                self.analysis_result = self.poker_logic.run_analysis(
                    p1_input, p2_input, board_input,
                    num_simulations=num_sims,
                    progress_callback=progress_update
                )
        except Exception as e:
            self.analysis_result = e

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if sys.platform.startswith('win'):
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    # 先显示启动页，选择 2/3/4/5/6 人模式
    startup = StartupWindow()
    startup.mainloop()
    try:
        startup.destroy()
    except tk.TclError:
        pass
    # 2 / 3 / 4 / 5 人模式进入主分析界面
    selected = getattr(startup, 'selected_players', None)
    if selected in (2, 3, 4, 5):
        app = PokerApp(PokerLogic(), num_players=selected)
        app.mainloop()