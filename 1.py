import tkinter as tk
from tkinter import ttk, font, messagebox
import random
import threading
from collections import defaultdict
import multiprocessing # 引入多进程模块
import sys # 引入 sys 模块，用于处理打包后的环境
import math # 引入 math 模块以使用缓动函数
import webbrowser # 新增: 用于打开网页链接

# 建议安装 'treys' 库: pip install treys
try:
    from treys import Card, Evaluator, Deck
except ImportError:
    print("错误: 未找到 'treys' 库。请使用 'pip install treys' 命令进行安装。")
    exit()

# ##################################################################
# ############### 新增的并行计算工作函数 ###########################
# ##################################################################
def _run_simulation_chunk(args):
    """
    执行一小块模拟任务。
    这个函数是独立的，以便被其他进程调用。
    """
    # 解包传入的参数
    # (V10) 增加了 calculate_p1_strength 参数
    p1_type, p1_hands, p2_type, p2_hands, board, num_sims_chunk, master_deck_cards, calculate_p1_strength = args

    evaluator = Evaluator() # 每个进程都需要创建自己的Evaluator实例
    rank_class_to_string_map = {
        1: "同花顺 (Straight Flush)", 2: "四条 (Four of a Kind)", 3: "葫芦 (Full House)",
        4: "同花 (Flush)", 5: "顺子 (Straight)", 6: "三条 (Three of a Kind)",
        7: "两对 (Two Pair)", 8: "一对 (One Pair)", 9: "高牌 (High Card)"
    }

    # 用于统计这个“块”的结果
    p1_wins, p2_wins, ties = 0, 0, 0
    p1_hand_strength_counts = defaultdict(int)
    valid_sims = 0
    # (V10) calculate_p1_strength = p1_type != 'random' # <-- 已删除, 从参数传入

    # 在循环外创建一次牌库，并移除已知的公共牌
    base_deck = list(master_deck_cards)
    for card in board:
        if card in base_deck:
            base_deck.remove(card)

    for _ in range(num_sims_chunk):
        deck_cards = list(base_deck)

        if p1_type == 'random':
            if len(deck_cards) < 2: continue
            p1_hand_sample = random.sample(deck_cards, 2)
        else:
            p1_hand_sample = random.choice(p1_hands)
        
        p1_hand_set = set(p1_hand_sample)
        if not p1_hand_set.issubset(deck_cards): continue
        for card in p1_hand_sample: deck_cards.remove(card)

        if p2_type == 'random':
            if len(deck_cards) < 2: continue
            p2_hand_sample = random.sample(deck_cards, 2)
        else:
            p2_hand_sample = random.choice(p2_hands)
        
        if not p1_hand_set.isdisjoint(p2_hand_sample): continue
        if not set(p2_hand_sample).issubset(deck_cards): continue
        for card in p2_hand_sample: deck_cards.remove(card)
        
        run_board = list(board)
        cards_needed = 5 - len(run_board)
        if len(deck_cards) < cards_needed: continue
        run_board.extend(random.sample(deck_cards, cards_needed))

        p1_score = evaluator.evaluate(run_board, p1_hand_sample)
        p2_score = evaluator.evaluate(run_board, p2_hand_sample)

        # (V10) 现在这个 if 总是为 True
        if calculate_p1_strength:
            p1_rank_class = evaluator.get_rank_class(p1_score)
            if p1_rank_class in rank_class_to_string_map:
                hand_type_str = rank_class_to_string_map[p1_rank_class]
                p1_hand_strength_counts[hand_type_str] += 1
        
        if p1_score < p2_score: p1_wins += 1
        elif p2_score < p1_score: p2_wins += 1
        else: ties += 1
        valid_sims += 1

    # 返回这个块的计算结果
    return p1_wins, p2_wins, ties, p1_hand_strength_counts, valid_sims

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
        return formatted_cards # 返回 ["As", "Ah"]

    # ##################################################################
    # ############### V9：修改: _determine_input_type ##################
    # ############### (添加了内部冲突检测) #########################
    # ##################################################################
    def _determine_input_type(self, p_input):
        # (V9) p_input 现在是 [s.strip().upper() ...], 但我们在这里再次清理以防万一
        clean_input = [s.strip().upper() for s in p_input if s.strip()]
        if not clean_input or clean_input == ['']:
            return 'random', None

        all_hand_pairs = []
        specific_hands_in_list = set()
        range_strings = []

        for item in clean_input:
            if len(item) == 4:
                try:
                    # 验证它是否是一个有效的特定手牌
                    r1 = item[0].upper()
                    s1 = item[1].lower()
                    r2 = item[2].upper()
                    s2 = item[3].lower()
                    c1 = Card.new(r1 + s1) # e.g., Card.new("As")
                    c2 = Card.new(r2 + s2) # e.g., Card.new("Ah")
                    
                    # (V9) 关键：立即检查手牌内部冲突 (e.g., "AHAH")
                    if c1 == c2:
                        raise ValueError(f"手牌 '{item}' 包含重复的牌。")
                        
                    specific_hands_in_list.add(item)
                except (ValueError, KeyError):
                    # 不是有效的4字符手牌 (e.g., "ASKA")，当作范围处理
                    range_strings.append(item)
            else:
                range_strings.append(item)
        
        # 1. 解析所有特定的手牌
        for hand_str in specific_hands_in_list:
            try:
                cards = self._split_hand_str(hand_str)
                hand = [Card.new(c) for c in cards]
                # (V9) 这里的 set 检查是多余的，因为上面已经检查过了，但保留也无妨
                if len(set(hand)) != 2:
                    raise ValueError(f"手牌 '{hand_str}' 包含重复的牌。")
                all_hand_pairs.append(hand)
            except (ValueError, KeyError) as e:
                # (V9) 将错误冒泡
                raise ValueError(f"解析特定手牌 '{hand_str}' 时出错: {e}")
                
        # 2. 解析所有范围字符串
        if range_strings:
            try:
                all_hand_pairs.extend(self._parse_hand_range(range_strings))
            except Exception as e:
                # _parse_hand_range 内部已经有打印警告，这里防止崩溃
                print(f"解析范围时出错: {e}")

        if not all_hand_pairs:
            # (V9) 如果特定手牌和范围都为空或无效
            if specific_hands_in_list or range_strings:
                 raise ValueError("无法从输入解析出任何有效手牌。")
            else:
                 # 这是 'random' 的情况 (e.g., 输入是 [''])
                 return 'random', None

        # 3. 去重
        deduped_set = {tuple(sorted(h)) for h in all_hand_pairs}
        final_hands_list = [list(t) for t in deduped_set]
        
        if not final_hands_list: 
            raise ValueError("无法解析手牌范围或特定手牌。")
            
        # (V9) 决定返回类型
        # 如果去重后只剩一手牌，并且没有范围字符串，则视为 'hand' 类型
        if len(final_hands_list) == 1 and not range_strings:
             return 'hand', final_hands_list
        else:
            # 否则，视为 'range' 类型
            return 'range', final_hands_list
            
    # ##################################################################
    # ############### V10：重写 run_analysis #########################
    # ##################################################################
    def run_analysis(self, p1_input_raw, p2_input_raw, board_str, num_simulations=50000, progress_callback=None):
        
        # (V9) 辅助函数，用于检查手牌列表与一个“已见卡牌”集合的冲突
        def check_hand_list(hand_list, seen_cards_set, player_name):
            """
            检查手牌列表中的每一张牌是否已在 seen_cards_set 中。
            如果冲突，则抛出 ValueError。
            如果不冲突，则将这些牌添加到 seen_cards_set 中。
            """
            for hand in hand_list:
                hand_set = set(hand)
                if not hand_set.isdisjoint(seen_cards_set):
                    # 找到冲突的牌
                    conflicting_card = (hand_set & seen_cards_set).pop()
                    # treys.Card.int_to_str(c) 可以将 131828 (As) 转换回 "As"
                    card_str = Card.int_to_str(conflicting_card)
                    hand_str = [Card.int_to_str(c) for c in hand]
                    raise ValueError(f"{player_name} 内部冲突: 牌 {card_str} (在手牌 {hand_str} 中) 已被使用。")
            
            # 如果没有冲突，将所有手牌添加到集合中
            for hand in hand_list:
                seen_cards_set.update(hand)

        # (V9) 辅助函数，用于从范围列表中过滤掉与“已见卡牌”冲突的手牌
        def filter_range(hand_list, seen_cards_set, player_name):
            """
            过滤手牌列表，移除任何与 seen_cards_set 冲突的手牌。
            """
            valid_hands = []
            for hand in hand_list:
                if set(hand).isdisjoint(seen_cards_set):
                    valid_hands.append(hand)
            
            if not valid_hands and hand_list: # 如果列表本来有牌，但全被过滤了
                raise ValueError(f"{player_name} 的范围与已选卡牌完全冲突。")
                
            return valid_hands

        try:
            # --- (V9) 步骤 1：解析所有输入 ---
            
            # 解析公共牌
            board = [Card.new(c) for c in self._split_hand_str(board_str)] if board_str else []
            if len(set(board)) != len(board): 
                raise ValueError("公共牌中包含重复的牌。")
            
            # 解析 P1 和 P2
            p1_type, p1_hands = self._determine_input_type(p1_input_raw)
            p2_type, p2_hands = self._determine_input_type(p2_input_raw)
            
            # ##################################################################
            # ############### V10 修改：始终计算 P1 牌力 #####################
            # ##################################################################
            calculate_p1_strength = True # 原: p1_type != 'random'

            # --- (V9) 步骤 2：健壮的冲突检测 ---
            
            # 全局“已见卡牌”集合
            seen_cards = set(board)

            # 检查 P1
            if p1_type == 'hand':
                # 'hand' 类型意味着列表只有一手牌。我们必须检查它。
                check_hand_list(p1_hands, seen_cards, "玩家1")
            elif p1_type == 'range':
                # 'range' 类型意味着列表有多手牌。我们必须过滤它。
                p1_hands = filter_range(p1_hands, seen_cards, "玩家1")
                # *不* 将 P1 范围添加到 seen_cards，因为 P2 的范围不应该与 P1 的范围冲突
                # 它们只在模拟运行时才会动态冲突

            # 检查 P2
            if p2_type == 'hand':
                # 'hand' 类型必须检查
                check_hand_list(p2_hands, seen_cards, "玩家2")
            elif p2_type == 'range':
                # 'range' 类型必须过滤
                p2_hands = filter_range(p2_hands, seen_cards, "玩家2")

        except ValueError as e:
            # (V9) 捕获所有解析和冲突错误
            raise ValueError(f"输入解析错误: {e}")

        # --- (V9) 步骤 3：并行计算设置 (V9-Fix 进度条) ---
        try:
            num_cores = max(1, multiprocessing.cpu_count() - 1)
        except NotImplementedError:
            num_cores = 1
            
        # ##################################################################
        # ############### V11-Fix: 优化任务分块大小以提升流畅度 #############
        # ##################################################################
        
        # 增加任务数量，使更新更频繁，从而使进度条更流畅
        TARGET_SMOOTHNESS_TASKS = 150  # 原为100，增加到150

        if num_simulations <= 0:
            num_tasks, chunk_size, remainder = 0, 0, 0
        else:
            # 目标是 150 个任务块，但任务数不能超过总模拟数
            num_tasks = min(TARGET_SMOOTHNESS_TASKS, num_simulations)
            
            chunk_size = num_simulations // num_tasks
            remainder = num_simulations % num_tasks
        
        # ##################################################################
        # ###################### 结束修改 #################################
        # ##################################################################
        
        tasks = []
        master_deck_cards = list(self.master_deck.cards)
        # (V10) 将 calculate_p1_strength (现在总是 True) 添加到基础参数中
        base_args = (p1_type, p1_hands, p2_type, p2_hands, board, master_deck_cards, calculate_p1_strength)

        for _ in range(num_tasks):
            if chunk_size > 0:
                # (V10) 调整元组索引
                tasks.append(base_args[:5] + (chunk_size,) + base_args[5:])
        if remainder > 0:
            # (V10) 调整元组索引
            tasks.append(base_args[:5] + (remainder,) + base_args[5:])

        total_p1_wins, total_p2_wins, total_ties = 0, 0, 0
        total_p1_strength_counts = defaultdict(int)
        total_valid_sims = 0
        
        # --- (V9) 步骤 4：执行 (与 V8 相同) ---
        with multiprocessing.Pool(processes=num_cores) as pool:
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
            
        if total_valid_sims == 0: 
            if num_simulations > 0:
                raise ValueError("无法完成任何有效模拟。请检查输入设置（例如手牌和公共牌冲突）。")
        
        # --- (V9) 步骤 5：汇总结果 (V10 修改) ---
        equity_results = {'p1_win': 0, 'p2_win': 0, 'tie': 0}
        if total_valid_sims > 0:
            equity_results = {
                'p1_win': (total_p1_wins / total_valid_sims) * 100,
                'p2_win': (total_p2_wins / total_valid_sims) * 100,
                'tie': (total_ties / total_valid_sims) * 100
            }
            
        strength_results = {}
        # (V10) 现在这个 if 总是为 True
        if calculate_p1_strength:
            total_strength_hands = sum(total_p1_strength_counts.values())
            if total_strength_hands > 0:
                for rank in range(1, 10):
                    hand_type = self.rank_class_to_string_map[rank]
                    prob = (total_p1_strength_counts.get(hand_type, 0) / total_strength_hands) * 100
                    strength_results[hand_type] = prob
        
        # (V10) calculate_p1_strength 现在总是 True
        return equity_results, strength_results, calculate_p1_strength


# ##################################################################
# ############### V8：花色选择器弹出窗口 (对子独立) ##################
# ##################################################################
class SuitSelectorWindow(tk.Toplevel):
    """
    一个弹出窗口，用于为特定的手牌类别（如 'AKs', 'T9o', 'AA'）选择具体花色。
    V8: 将对子(Pair)的选择逻辑改为与Offsuit一致，允许独立选择 '未知'。
    """
    def __init__(self, master, hand_text, callback):
        super().__init__(master)
        
        self.hand_text = hand_text
        self.callback = callback
        self.hand_type = None
        self.rank1 = hand_text[0]
        self.rank2 = hand_text[1]

        # (V7) 尝试设置图标
        try:
            self.iconbitmap(r'C:\Users\wangz\Desktop\Texas_Poker\TexasPoker.ico')  
        except tk.TclError:
            print("警告: 找不到图标文件 'TexasPoker.ico'。将使用默认图标。")
        
        # --- (V7) 字体定义 ---
        self.FONT_STATUS = ("Microsoft YaHei", 16, "bold") # 状态标签
        self.FONT_SYMBOL = ("Microsoft YaHei", 24, "bold") # 按钮上的 ♠, ♥, ?
        self.FONT_LABEL = ("Microsoft YaHei", 12)          # "牌 1 (A):"
        self.FONT_PROMPT = ("Microsoft YaHei", 14)         # "为 ... 选择花色:"
        self.FONT_SELECTED = ("Microsoft YaHei", 14, "italic") # "已选: -"
        
        # --- 颜色主题 (V6) ---
        self.BG_COLOR = '#2e2e2e'
        self.BTN_BG_NORMAL = '#4a4a4a'
        self.BTN_FG_NORMAL = 'white'
        self.BTN_BG_HOVER = '#6a6a6a'
        self.BTN_BG_DISABLED = '#3a3a3a'
        self.BTN_FG_DISABLED = '#5c5c5c' # 禁用的花色
        self.BTN_BG_SELECTED = '#007bff' # P1 Color
        self.BTN_BORDER_SELECTED = '#80bfff'
        self.LABEL_FG = '#d0d0d0'

        # 1. 判断手牌类型
        # (V8) 修改：'pair' 现在使用和 'offsuit' 一样的选择逻辑
        if hand_text.endswith('s'):
            self.hand_type = 'suited'
            self.title(f"选择 {hand_text} 花色")
            self.selection = [] # 存储1个花色, e.g., ['s']
        else:
            # (V8) 'pair' (e.g. "AA") 和 'offsuit' (e.g. "AKo") 都走这里
            self.hand_type = 'pair' if self.rank1 == self.rank2 else 'offsuit'
            self.title(f"选择 {hand_text} 花色")
            self.selection = [None, None] # 存储2个花色

        # 2. 窗口配置
        self.configure(bg=self.BG_COLOR)
        self.transient(master)
        self.grab_set()
        self.resizable(False, False)
        
        master_x = master.winfo_x()
        master_y = master.winfo_y()
        master_w = master.winfo_width()
        master_h = master.winfo_height()
        win_w, win_h = 580, 350 
        pos_x = master_x + (master_w // 2) - (win_w // 2)
        pos_y = master_y + (master_h // 2) - (win_h // 2)
        self.geometry(f"{win_w}x{win_h}+{pos_x}+{pos_y}")

        # 3. 创建控件
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
        
        # --- (V7) 自定义花色按钮创建函数 ---
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
                    # (V8) 修改：检查 'offsuit' 和 'pair'
                    if self.hand_type in ['offsuit', 'pair'] and self.selection:
                        is_selected = (btn.suit_key == self.selection[0]) or (btn.suit_key == self.selection[1])
                    
                    if not is_selected:
                        btn.config(bg=self.BTN_BG_NORMAL)
            
            btn.bind("<Button-1>", on_click)
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
            
            if state == 'disabled':
                btn.config(state='disabled', fg=self.BTN_FG_DISABLED, bg=self.BTN_BG_DISABLED, relief='flat')
            
            return btn

        # (V8) --- 布局逻辑修改 ---
        if self.hand_type == 'offsuit' or self.hand_type == 'pair':
            # (V8) Offsuit 和 Pair 共享此布局
            
            # (V8) 决定 R1 和 R2 的点击回调
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
            # (V8) rank2 (对于对子来说，这与 rank1 相同)
            ttk.Label(frame_r2, text=f"牌 2 ({self.rank2}):", font=self.FONT_LABEL, foreground=self.LABEL_FG, width=10).pack(side='left', padx=(10, 10))
            for key in self.button_key_list:
                btn = create_suit_button(frame_r2, key, r2_cmd, state='disabled') 
                btn.pack(side='left', padx=6, pady=4)
                self.buttons_r2[key] = btn
        
        elif self.hand_type == 'suited':
            # --- Suited 布局 (不变) ---
            prompt = f"为 {self.rank1}{self.rank2}s 选择花色:"
            ttk.Label(self.selection_frame, text=prompt, font=self.FONT_PROMPT, foreground=self.LABEL_FG, anchor='center').pack(pady=10, fill='x')
            
            button_frame = ttk.Frame(self.selection_frame)
            button_frame.pack(pady=10) 
            for key in self.button_key_list:
                # (V8) 移除 'pair' 的回调
                btn = create_suit_button(button_frame, key, self._on_suited_click) 
                btn.pack(side='left', padx=8, pady=5)
                self.buttons_r1[key] = btn 

        # 实时选择显示标签
        self.selection_label_var = tk.StringVar(value="已选: -")
        ttk.Label(main_frame, textvariable=self.selection_label_var, font=self.FONT_SELECTED, foreground='#ccc', anchor='center').pack(pady=(15, 0), fill='x')

        # 底部控制按钮 - 只有重置
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side='bottom', fill='x', pady=(20, 0))
        
        ttk.Button(control_frame, text="重置", 
                   command=self._reset).pack(fill='x', expand=True, padx=10, ipady=10)
                   
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._update_status_label() 

    def _update_status_label(self):
        """(V8) 更新顶部的提示标签 (Pair 逻辑合并)"""
        suits_map = self.suits_map_display 

        # (V8) Offsuit 和 Pair 逻辑现在相同
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
            
        # (V8) 旧的 'pair' 逻辑已移除

    def _enable_r2_buttons(self, r1_key_char):
        """(V8) 辅助函数：启用 R2 按钮"""
        for key, btn in self.buttons_r2.items():
            # 逻辑：
            # 如果 R1 是 'unknown', R2 可以是任何牌 (包括 'unknown')
            # 如果 R1 是 's', R2 不能是 's'
            is_disabled = (r1_key_char != 'unknown' and key == r1_key_char)

            if is_disabled:
                btn.config(state='disabled', fg=self.BTN_FG_DISABLED, bg=self.BTN_BG_DISABLED, relief='flat')
            else:
                color = self.LABEL_FG if key == 'unknown' else self.suits_map_display[key][1]
                btn.config(state='normal', fg=color, bg=self.BTN_BG_NORMAL, relief='raised')
                btn.bind("<Enter>", btn.bindtags()[0], "+")
                btn.bind("<Leave>", btn.bindtags()[0], "+")

    def _disable_all_r1_buttons(self, selected_key):
        """(V8) 辅助函数：禁用 R1 按钮并高亮显示"""
        for btn in self.buttons_r1.values():
            btn.config(state='disabled', fg=self.BTN_FG_DISABLED, bg=self.BTN_BG_DISABLED, relief='flat')
            btn.unbind("<Enter>")
            btn.unbind("<Leave>")
        # 高亮选中的 R1 按钮
        self.buttons_r1[selected_key].config(bg=self.BTN_BG_SELECTED, relief='solid', borderwidth=2, highlightbackground=self.BTN_BORDER_SELECTED) 

    def _disable_all_r2_buttons(self, selected_key):
        """(V8) 辅助函数：禁用 R2 按钮并高亮显示"""
        for btn in self.buttons_r2.values():
            btn.config(state='disabled', fg=self.BTN_FG_DISABLED, bg=self.BTN_BG_DISABLED, relief='flat')
            btn.unbind("<Enter>")
            btn.unbind("<Leave>")
        # 高亮选中的 R2 按钮
        self.buttons_r2[selected_key].config(bg=self.BTN_BG_SELECTED, relief='solid', borderwidth=2, highlightbackground=self.BTN_BORDER_SELECTED) 

    # --- (V8) 新增：对子(Pair)的点击事件 ---

    def _on_pair_r1_click(self, key_char):
        """(Pair) V8 - R1 点击，可处理 'unknown'"""
        self.selection[0] = key_char
        self._disable_all_r1_buttons(key_char)
        self._enable_r2_buttons(key_char)
        self._update_status_label()

    def _on_pair_r2_click(self, key_char):
        """(Pair) V8 - R2 点击，可处理 'unknown'。"""
        self.selection[1] = key_char
        self._disable_all_r2_buttons(key_char)
        
        s1, s2 = self.selection[0], self.selection[1]
        
        final_submission = None
        all_suits = ['s', 'h', 'd', 'c']

        if s1 != 'unknown' and s2 != 'unknown':
            # Case 1: (suit, suit) -> e.g., "AsAh"
            final_submission = f"{self.rank1}{s1}{self.rank1}{s2}".upper()
            
        elif s1 == 'unknown' and s2 == 'unknown':
            # Case 2: (unknown, unknown) -> e.g., "AA"
            final_submission = self.hand_text
            
        else:
            # Case 3: (suit, unknown) or (unknown, suit) -> 部分范围
            final_hands_list = []
            
            if s1 == 'unknown' and s2 != 'unknown':
                # (A?, Ah)
                for suit1 in all_suits:
                    if suit1 == s2: continue 
                    final_hands_list.append(f"{self.rank1}{suit1}{self.rank1}{s2}".upper())
            
            elif s1 != 'unknown' and s2 == 'unknown':
                # (As, A?)
                for suit2 in all_suits:
                    if suit2 == s1: continue
                    final_hands_list.append(f"{self.rank1}{s1}{self.rank1}{suit2}".upper())
            
            final_submission = ", ".join(final_hands_list)

        self.callback(final_submission)
        self.destroy()

    # --- (V6) Offsuit 点击事件 (现在使用辅助函数) ---

    def _on_r1_click(self, key_char):
        """(Offsuit) V6 - R1 点击，可处理 'unknown'"""
        self.selection[0] = key_char
        self._disable_all_r1_buttons(key_char)
        self._enable_r2_buttons(key_char)
        self._update_status_label()

    def _on_r2_click(self, key_char):
        """(Offsuit) V6 - R2 点击，可处理 'unknown'。这是核心。"""
        self.selection[1] = key_char
        self._disable_all_r2_buttons(key_char)
        
        s1, s2 = self.selection[0], self.selection[1]
        
        # (V6) --- 决定回调内容 ---

        final_submission = None
        all_suits = ['s', 'h', 'd', 'c']

        if s1 != 'unknown' and s2 != 'unknown':
            # Case 1: (suit, suit) -> e.g., "AsKh"
            final_submission = f"{self.rank1}{s1}{self.rank2}{s2}".upper()
            
        elif s1 == 'unknown' and s2 == 'unknown':
            # Case 2: (unknown, unknown) -> e.g., "AKo"
            final_submission = self.hand_text
            
        else:
            # Case 3: (suit, unknown) or (unknown, suit) -> 部分范围
            # e.g., AsK? -> "ASKH, ASKD, ASKC"
            # e.g., A?Ks -> "AHKS, ADKS, ACKS"
            
            final_hands_list = []
            
            if s1 == 'unknown' and s2 != 'unknown':
                # (A?, Ks)
                for suit1 in all_suits:
                    if suit1 == s2: 
                        continue # 跳过同花 (e.g., AsKs)
                    final_hands_list.append(f"{self.rank1}{suit1}{self.rank2}{s2}".upper())
            
            elif s1 != 'unknown' and s2 == 'unknown':
                # (As, K?)
                for suit2 in all_suits:
                    if suit2 == s1: 
                        continue # 跳过同花 (e.g., AsKs)
                    final_hands_list.append(f"{self.rank1}{s1}{self.rank2}{suit2}".upper())
            
            final_submission = ", ".join(final_hands_list)

        # 提交结果
        self.callback(final_submission)
        self.destroy()

    # --- (V6) Suited 点击事件 (不变) ---

    def _on_suited_click(self, key_char):
        """(Suited) V6 - 处理花色点击"""
        if key_char == 'unknown':
            # (V6) 用户点击了 '?', 提交原始范围
            self.callback(self.hand_text) # e.g., "AKs"
            self.destroy()
            return

        self.selection.append(key_char)
        # 禁用所有按钮
        for btn in self.buttons_r1.values():
            btn.config(state='disabled', fg=self.BTN_FG_DISABLED, bg=self.BTN_BG_DISABLED, relief='flat')
            btn.unbind("<Enter>")
            btn.unbind("<Leave>")
        # 高亮选中的
        self.buttons_r1[key_char].config(bg=self.BTN_BG_SELECTED, relief='solid', borderwidth=2, highlightbackground=self.BTN_BORDER_SELECTED) 

        s_disp = self.suits_map_display[key_char][0]
        self.selection_label_var.set(f"已选: {self.rank1}{s_disp}{self.rank2}{s_disp}")

        final_hand = f"{self.rank1}{key_char}{self.rank2}{key_char}"
        self.callback(final_hand.upper())
        self.destroy()

    # (V8) _on_pair_click 方法已被删除
    # (V6) _on_unknown_click 方法已被删除

    def _on_cancel(self):
        """用户关闭了窗口"""
        self.callback(None) # 回调 None 表示取消
        self.destroy()

    def _reset(self):
        """(V8) 重置选择 (Pair 逻辑合并)"""
        
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

        # (V8) Offsuit 和 Pair 逻辑相同
        if self.hand_type == 'offsuit' or self.hand_type == 'pair':
            self.selection = [None, None]
            reset_button_row(self.buttons_r1, state='normal')
            reset_button_row(self.buttons_r2, state='disabled')
        
        elif self.hand_type == 'suited':
            self.selection = []
            reset_button_row(self.buttons_r1, state='normal')
        
        self._update_status_label()

# ##################################################################
# ######################## 结束修改窗口 ############################
# ##################################################################


# --- 起手牌强度图表窗口 ---
class StrengthChartWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        
        # ##################################################################
        # #################### 关键修改：添加淡入动画 ######################
        # ##################################################################
        self.wm_attributes('-alpha', 0.0) # 1. 初始透明

        self.title("起手牌强度图表")
        self.geometry("900x750")
        try:
            # 尝试设置图标
            self.iconbitmap(r'C:\Users\wangz\Desktop\Texas_Poker\TexasPoker.ico') 
        except tk.TclError:
            print("警告: 找不到图标文件 'TexasPoker.ico'。") 
        self.configure(bg='#2e2e2e')
        self.transient(master)
        
        # ##################################################################
        # #################### 关键修改：移除 grab_set #####################
        # ##################################################################
        # self.grab_set() # <-- 移除这行，这是导致黑屏的罪魁祸首
        self.attributes('-topmost', True) # <-- 替换为 -topmost 保证窗口在最前
        # ##################################################################
        

        # ##################################################################
        # ###################### 新增: 绑定键盘移动事件 ######################
        # ##################################################################
        self.bind('<KeyPress>', self._handle_window_movement)

        # ##################################################################
        # ############ 新增: 绑定关闭窗口事件以通知主窗口 ##############
        # ##################################################################
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self._create_strength_chart()

        # ##################################################################
        # #################### 关键修改：启动淡入动画 ######################
        # ##################################################################
        self.after(10, self._start_fade_in) # 2. 启动动画


    # ##################################################################
    # ##################### 关键修改：窗口关闭处理函数 ###################
    # ##################################################################
    def _on_close(self):
        """当此窗口关闭时，通知主窗口并销毁自己"""
        # self.master 指向的是 PokerApp 实例
        self.master.strength_chart_window = None
        
        # ##################################################################
        # ############ 关键修改：重新启用主窗口的按钮 ##################
        # ##################################################################
        try:
            self.master.open_chart_button.config(state='normal')
        except Exception as e:
            print(f"启用按钮失败: {e}")
            
        self.destroy()

    def _reset(self):
        """(V5) 重置选择"""
        
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

        if self.hand_type == 'offsuit':
            self.selection = [None, None]
            reset_button_row(self.buttons_r1, state='normal')
            reset_button_row(self.buttons_r2, state='disabled')
        
        elif self.hand_type in ['suited', 'pair']:
            self.selection = []
            reset_button_row(self.buttons_r1, state='normal')
        
        self._update_status_label()
    # ##################################################################
    # ################## 新增: 窗口移动事件处理函数 ####################
    # ##################################################################
    def _handle_window_movement(self, event):
        """处理键盘事件以移动此窗口"""
        # 检查当前拥有焦点的控件是否是文本输入框 (为保持代码一致性，尽管此窗口没有)
        focused_widget = self.focus_get()
        if isinstance(focused_widget, ttk.Entry):
            return  # 如果是输入框，则不执行任何操作

        move_step = 15  # 每次按键移动的像素值
        x = self.winfo_x()
        y = self.winfo_y()

        key = event.keysym.lower()

        if key == 'w' or key == 'up':
            y -= move_step
        elif key == 's' or key == 'down':
            y += move_step
        elif key == 'a' or key == 'left':
            x -= move_step
        elif key == 'd' or key == 'right':
            x += move_step
        else:
            return # 忽略其他所有按键

        # 更新窗口位置，但不改变大小
        self.geometry(f"+{x}+{y}")

    # ##################################################################
    # ############### 新增：图表窗口的缓动动画函数 #####################
    # ##################################################################
    def _start_fade_in(self):
        """(辅助函数) 初始化并启动淡入动画"""
        self.animation_total_duration = 300  # 总毫秒数 (0.3秒)
        self.animation_step_delay = 15       # 每一步的毫秒数 (约66 FPS)
        
        try:
            total_steps = self.animation_total_duration / self.animation_step_delay
        except ZeroDivisionError:
            total_steps = 0 

        if total_steps == 0:
            self.wm_attributes('-alpha', 1.0)
            return
            
        self.progress_increment = 1.0 / total_steps
        self.current_progress = 0.0
        
        self._fade_in_step()

    def _fade_in_step(self):
        """(辅助函数) 执行淡入动画的单一步骤"""
        self.current_progress += self.progress_increment
        
        if self.current_progress >= 1.0:
            self.wm_attributes('-alpha', 1.0) # 确保最终为 1.0
        else:
            # 使用 math.sin 实现缓出（Ease-Out）
            eased_alpha = math.sin(self.current_progress * (math.pi / 2))
            
            try:
                self.wm_attributes('-alpha', eased_alpha)
            except tk.TclError:
                # 窗口可能在动画过程中被关闭
                return
            
            self.after(self.animation_step_delay, self._fade_in_step)
    # ##################################################################
    # ######################## 结束新增动画代码 ##########################
    # ##################################################################

    def _create_strength_chart(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill='both', expand=True)
        ttk.Label(main_frame, text="起手牌强度等级表", font=("Microsoft YaHei", 16, "bold")).pack(pady=(1,1))
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(pady=1, fill='x', expand=True)
        
        grid_frame = ttk.Frame(content_frame)
        grid_frame.pack(side='left', padx=(0, 20), anchor='n')
        
        # ##################################################################
        # ############### 修改: 创建右侧面板以容纳图例和按钮 ###############
        # ##################################################################
        right_content_frame = ttk.Frame(content_frame)
        right_content_frame.pack(side='left', anchor='n', fill='y')

        legend_frame = ttk.LabelFrame(right_content_frame, text="图例")
        legend_frame.pack(side='top', fill='x')
        # ##################################################################

        hand_tiers = {
            # T1: 红色 (强力、警示)
            "超级精英 (Super Premium)": (
                "#dc2626",  # Tailwind Red-600
                ['AA', 'KK', 'QQ', 'AKs']
            ),
            # T2: 蓝色 (精英、可信)
            "精英牌 (Premium)": (
                "#2563eb",  # Tailwind Blue-600
                ['JJ', 'TT', 'AQs', 'AKo']
            ),
            # T3: 青色 (可玩、安全 - 替换绿色，色盲友好)
            "强可玩牌 (Strong Playable)": (
                "#0d9488",  # Tailwind Teal-600
                ['99', '88', 'AJs', 'KQs', 'AQo']
            ),
            # T4: 琥珀色 (投机、谨慎)
            "优质投机牌 (Prime Speculative)": (
                "#f59e0b",  # Tailwind Amber-500
                ['77', '66', '55', 'A5s', 'A4s', 'A3s', 'A2s', 'KJs', 'QJs', 'JTs', 'T9s', '98s', '87s', 'KJo', 'QJo']
            ),
            # T5: 橙色 (边缘、警告)
            "边缘可玩牌 (Positional Playable)": (
                "#f97316",  # Tailwind Orange-600
                ['44', '33', '22', 'A9s', 'A8s', 'A7s', 'A6s', 'ATs', 'KTs', 'QTs', 'J9s', '76s', '65s', '54s', 'KQo', 'AJo', 'JTo', 'ATo']
            ),
            # T6: 紫色 (边缘、冷色调)
            "弱投机牌 (Weak Speculative)": (
                "#7c3aed",  # Tailwind Violet-600
                ['K9s', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s', 'Q9s', 'Q8s', 'J8s', 'T8s', '97s', '86s', '75s', '64s', '53s', 'KTo', 'QTo', 'T9o', '98o']
            ),
            # T7: 灰色 (中性、防守)
            "边缘防守牌 (Marginal Defense)": (
                "#64748b",  # Tailwind Slate-500
                ['A9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o', 'K9o', 'Q9o', 'J9o', '87o', '76o', '65o', '54o']
            ),
            
            # T8: 深灰 (弃牌) - 【已补全所有缺失牌】
            "弃牌 (Fold)": (
                "#334155",  # Tailwind Slate-700
                [
                # 缺失的同花牌 (36种)
                'Q7s', 'Q6s', 'Q5s', 'Q4s', 'Q3s', 'Q2s',
                'J7s', 'J6s', 'J5s', 'J4s', 'J3s', 'J2s',
                'T7s', 'T6s', 'T5s', 'T4s', 'T3s', 'T2s',
                '96s', '95s', '94s', '93s', '92s',
                '85s', '84s', '83s', '82s',
                '74s', '73s', '72s',
                '63s', '62s',
                '52s',
                '43s', '42s',
                '32s',
                
                # 原有的非同花牌 (51种)
                'K8o', 'K7o', 'K6o', 'K5o', 'K4o', 'K3o', 'K2o',
                'Q8o', 'Q7o', 'Q6o', 'Q5o', 'Q4o', 'Q3o', 'Q2o',
                'J8o', 'J7o', 'J6o', 'J5o', 'J4o', 'J3o', 'J2o',
                'T8o', 'T7o', 'T6o', 'T5o', 'T4o', 'T3o', 'T2o',
                '97o', '96o', '95o', '94o', '93o', '92o',
                '86o', '85o', '84o', '83o', '82o',
                '75o', '74o', '73o', '72o',
                '64o', '63o', '62o',
                '53o', '52o',
                '43o', '42o',
                '32o'
                ]
            )
        }

        hand_to_tier_color = {}
        for tier_name, (color, hands) in hand_tiers.items():
            for hand in hands: hand_to_tier_color[hand] = color
        for tier_name, (color, _) in hand_tiers.items():
            legend_item = ttk.Frame(legend_frame)
            # ##################################################################
            # ############### 修改: 恢复间距 ##################
            # ##################################################################
            legend_item.pack(anchor='w', padx=10, pady=5) # 恢复到较紧凑的间距
            tk.Label(legend_item, text=" ", bg=color, width=2, relief='solid', borderwidth=1).pack(side='left')
            ttk.Label(legend_item, text=f"  {tier_name}").pack(side='left')
            
        # ##################################################################
        # ############### 修改: 增加占位符以撑大图例框 #####################
        # ##################################################################
        ttk.Frame(legend_frame, height=60).pack(fill='x')
        
        # ##################################################################
        # ############### 新增: 添加“起手牌热力图”按钮 #####################
        # ##################################################################
        heatmap_btn = ttk.Button(right_content_frame, text="起手牌热力图", command=lambda: webbrowser.open("https://historianonvult.github.io/poker-heatmap"))
        heatmap_btn.pack(side='top', pady=10, fill='x')
        # ##################################################################

        ranks = 'AKQJT98765432'
        btn_font = font.Font(family='Arial', size=10, weight='bold')
        
        # 创建渐变变色的辅助函数
        def create_hover_effect(cell, original_color):
            # 存储动画相关的数据
            cell.animation_data = {
                'original_color': original_color,
                'is_hovering': False,
                'current_step': 0,
                'timer_id': None
            }
            
            def hex_to_rgb(hex_color):
                """将hex颜色转换为RGB"""
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            def rgb_to_hex(rgb):
                """将RGB转换为hex颜色"""
                return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            
            def interpolate_color(color1, color2, factor):
                """在两个颜色之间插值"""
                rgb1 = hex_to_rgb(color1)
                rgb2 = hex_to_rgb(color2)
                result = tuple(rgb1[i] + (rgb2[i] - rgb1[i]) * factor for i in range(3))
                return rgb_to_hex(result)
            
            def animate_color():
                """执行颜色动画"""
                data = cell.animation_data
                total_steps = 15  # 总步数，数字越大动画越慢
                
                if data['is_hovering']:
                    # 悬停时，逐渐变亮
                    if data['current_step'] < total_steps:
                        data['current_step'] += 1
                        factor = data['current_step'] / total_steps
                        # 目标颜色：原色变亮（增加RGB值）
                        target = '#ffffff'
                        new_color = interpolate_color(data['original_color'], target, factor * 0.3)
                        cell.configure(bg=new_color)
                        data['timer_id'] = cell.after(20, animate_color)  # 20ms后继续动画
                else:
                    # 离开时，逐渐恢复原色
                    if data['current_step'] > 0:
                        data['current_step'] -= 1
                        factor = data['current_step'] / total_steps
                        target = '#ffffff'
                        new_color = interpolate_color(data['original_color'], target, factor * 0.3)
                        cell.configure(bg=new_color)
                        data['timer_id'] = cell.after(20, animate_color)
                    else:
                        cell.configure(bg=data['original_color'])
            
            def on_enter(event):
                data = cell.animation_data
                if data['timer_id']:
                    cell.after_cancel(data['timer_id'])
                data['is_hovering'] = True
                animate_color()
            
            def on_leave(event):
                data = cell.animation_data
                if data['timer_id']:
                    cell.after_cancel(data['timer_id'])
                data['is_hovering'] = False
                animate_color()
            
            cell.bind('<Enter>', on_enter)
            cell.bind('<Leave>', on_leave)
        
        # 创建表格
        for r in range(len(ranks)):
            for c in range(len(ranks)):
                if r < c: text = f"{ranks[r]}{ranks[c]}s"
                elif c < r: text = f"{ranks[c]}{ranks[r]}o"
                else: text = f"{ranks[r]}{ranks[c]}"
                bg_color = hand_to_tier_color.get(text, '#343a40')
                cell = tk.Label(grid_frame, text=text, font=btn_font, fg='white', bg=bg_color, 
                            width=5, height=2, relief='solid', borderwidth=1)
                cell.grid(row=r, column=c)
                # 添加悬停效果
                create_hover_effect(cell, bg_color)
        
        strategy_frame = ttk.LabelFrame(main_frame, text="位置策略简介")
        strategy_frame.pack(fill='x', pady=1)
        strategy_text = ("位置是德州扑克中最重要的概念之一。你在牌桌上的位置决定了你的行动顺序。\n"
                        "通常来说，你的位置越靠后（越晚行动），你就可以用越宽（越多）的范围来游戏。\n\n"
                        "● 前位 (Early Position - EP): 你需要用最强的牌（如精英牌、优质牌）率先加注，因为你后面还有很多玩家未行动。\n"
                        "● 中位 (Middle Position - MP): 可以适当增加一些强可玩牌和投机牌。\n"
                        "● 后位 (Late Position - CO/BTN): 这是最好的位置。你可以用更宽的范围加注，包括很多潜力牌和边缘牌，以攻击盲注。")
        ttk.Label(strategy_frame, text=strategy_text, wraplength=800, justify='left').pack(padx=10, pady=10)

# ##################################################################
# ############### 新增：渐变进度条类 ###############################
# ##################################################################
class GradientProgressBar(tk.Canvas):
    """
    自定义渐变进度条，替代 ttk.Progressbar 以实现更丰富的颜色效果。
    使用 Canvas 绘制渐变背景，并通过遮罩层控制进度显示。
    (V11-Opt) 优化绘制逻辑：分段绘制以减少Canvas对象数量，提升性能。
    """
    def __init__(self, parent, color_list, max_val=100, width=200, height=20, bg_color='#2e2e2e', **kwargs):
        # 初始化 Canvas，无边框，无高亮
        super().__init__(parent, width=width, height=height, bg=bg_color, highlightthickness=0, borderwidth=0, **kwargs)
        self.color_list = color_list  # 渐变颜色列表 (e.g. ['#ff0000', '#00ff00'])
        self.max_val = max_val
        self.current_val = 0
        self.bg_color = bg_color
        
        # 绑定调整大小事件，以便重新绘制渐变
        self.bind('<Configure>', self._on_resize)
        
        self._width = 1
        self._height = 1
        self._mask_id = None
        
    def _hex_to_rgb(self, hex_col):
        """将 Hex 颜色转换为 RGB 元组"""
        hex_col = hex_col.lstrip('#')
        return tuple(int(hex_col[i:i+2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(self, rgb):
        """将 RGB 元组转换为 Hex 颜色"""
        return '#{:02x}{:02x}{:02x}'.format(int(max(0, min(255, rgb[0]))), int(max(0, min(255, rgb[1]))), int(max(0, min(255, rgb[2]))))

    def _interpolate(self, color1, color2, t):
        """在两个颜色之间进行线性插值"""
        c1 = self._hex_to_rgb(color1)
        c2 = self._hex_to_rgb(color2)
        new_rgb = tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(3))
        return self._rgb_to_hex(new_rgb)
    
    # (V12) 新增：允许动态修改颜色
    def set_colors(self, new_color_list):
        """更新渐变颜色"""
        self.color_list = new_color_list
        self._draw_gradient()

    def _draw_gradient(self):
        """绘制背景渐变"""
        self.delete("all")
        self._width = self.winfo_width()
        self._height = self.winfo_height()
        
        # 宽度太小或不可见时不绘制
        if self._width <= 1: return

        num_colors = len(self.color_list)
        if num_colors < 2: return
        
        # ##################################################################
        # ############### V11-Opt: 性能优化 - 分段绘制 #####################
        # ##################################################################
        
        # 旧逻辑：每个像素都画一条线 -> 对象数量 = 宽度 (e.g. 800+)
        # 新逻辑：限制最大段数 (e.g. 120)，如果宽度很大，每个段会有几个像素宽
        # 视觉上差异极小，但对象数量大幅减少，显著提升重绘性能
        
        limit_segments = 120 
        step = max(1, self._width // limit_segments)
        
        for x in range(0, self._width, step):
            # 计算当前段的中心点对应的全局进度
            center_x = x + step / 2
            t_global = center_x / self._width
            
            # 确定当前所在的颜色段索引
            idx = int(t_global * (num_colors - 1))
            idx = min(idx, num_colors - 2)
            
            # 计算该段内的局部进度 t
            t_local = (t_global * (num_colors - 1)) - idx
            
            # 计算插值颜色
            color = self._interpolate(self.color_list[idx], self.color_list[idx+1], t_local)
            
            # 绘制矩形填充这一小段
            x1 = x
            x2 = min(x + step, self._width)
            # 使用 outline="" 避免矩形边框干扰，tags="gradient"
            self.create_rectangle(x1, 0, x2, self._height, fill=color, outline="", tags="gradient")
            
        # ##################################################################
        # ########################## 结束优化 #############################
        # ##################################################################
        
        # 创建遮罩层矩形 (使用背景色覆盖未完成的部分)
        # 初始状态下覆盖整个区域 (假设 val=0)
        self._mask_id = self.create_rectangle(0, 0, self._width, self._height, fill=self.bg_color, outline="", tags="mask")
        self._update_mask_position()

    def _on_resize(self, event):
        """窗口大小改变时重绘"""
        # 简单的防抖动或限制重绘频率可以在这里添加，但对于此应用直接重绘通常足够快
        self._draw_gradient()

    def _update_mask_position(self):
        """根据当前进度更新遮罩层的位置"""
        if not self._mask_id: return
        
        if self.max_val <= 0: pct = 0
        else: pct = min(1.0, max(0.0, self.current_val / self.max_val))
        
        # 计算遮罩层的起始 x 坐标
        # 进度条显示部分为 0 到 x_pos，遮罩层覆盖 x_pos 到 width
        x_pos = int(pct * self._width)
        
        # 更新遮罩层坐标
        # y 坐标稍微超出范围以确保完全覆盖
        self.coords(self._mask_id, x_pos, -5, self._width + 5, self._height + 5)
        
    def set_value(self, value):
        """设置当前进度值"""
        self.current_val = value
        self._update_mask_position()
        
    def set_max(self, max_val):
        """设置最大值"""
        self.max_val = max_val
        self._update_mask_position()

# --- GUI 应用 (UI/UX 优化后) ---
class PokerApp(tk.Tk):
    # (PokerApp 类的 __init__, _configure_styles, _create_widgets 等方法)
    def __init__(self, poker_logic):
        super().__init__()
        
        # ##################################################################
        # ###################### 新增: 入场动画 - 步骤 1 #####################
        # ##################################################################
        # 启动时将窗口设置为完全透明
        self.wm_attributes('-alpha', 0.0)
        # ##################################################################

        self.poker_logic = poker_logic
        self.title("德州扑克分析工具")
        window_width = 1370
        window_height = 960
        try:
            # 尝试设置图标
            self.iconbitmap(r'C:\Users\wangz\Desktop\Texas_Poker\TexasPoker.ico')  
        except tk.TclError:
            print("警告: 找不到图标文件 'TexasPoker.ico'。将使用默认图标。")

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        position_x = int(screen_width / 2 - window_width / 2)
        position_y = int(screen_height / 2 - window_height / 2) - 33

        self.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.configure(bg='#2e2e2e')

        # ##################################################################
        # ############### 修改: 移除旧的 range selection ###################
        # ##################################################################
        # self.range_selection_p1 = set() # <-- 已移除
        # self.range_selection_p2 = set() # <-- 已移除
        # ##################################################################
        
        self.board_cards = []
        self.analysis_thread = None
        self.analysis_result = None
        self.progress_var = tk.DoubleVar()
        self.P1_COLOR = '#007bff'
        self.P2_COLOR = '#dc3545'
        # --- 变化: 增加了重叠颜色 ---
        self.BOTH_COLOR = '#8a2be2'  # 紫罗兰色 (BlueViolet)
        # --- 结束变化 ---
        self.DEFAULT_BG = '#4f4f4f'
        self.PAIR_BG = '#8fbc8f'
        self.SUITED_BG = '#4a7a96'
        
        # ##################################################################
        # ############### (V12) 新增：多组进度条配色方案 ####################
        # ##################################################################
        self.pb_themes = [
            # 现有
            ['#4158D0', '#C850C0', '#FFCC70'], # 经典的蓝紫-洋红-暖橙
            ['#0093E9', '#80D0C7'],            # 清新的蓝青色
            ['#8EC5FC', '#E0C3FC'],            # 柔和的蓝紫色
            ['#FA8BFF', '#2BD2FF', '#2BFF88'], # 强烈的霓虹三色
            ['#FF9A9E', '#FECFEF'],            # 温暖的粉色系
            ['#FBAB7E', '#F7CE68'],            # 活力的橙黄色
            ['#85FFBD', '#FFFB7F'],            # 清爽的柠檬绿
            ['#FF3CAC', '#784BA0', '#2B86C5'], # 深邃的紫蓝色
            # 新增
            ['#FF9966', '#FF5E62'],            # 蜜桃橘红
            ['#56ab2f', '#a8e063'],            # 清新草绿
            ['#F12711', '#F5AF19'],            # 火焰红黄
            ['#FC466B', '#3F5EFB'],            # 迷幻红蓝
            ['#C6FFDD', '#FBD786', '#f7797d'], # 马卡龙三色
            ['#12c2e9', '#c471ed', '#f64f59'], # 经典的蓝紫红
            ['#b92b27', '#1565C0'],            # 红蓝对决
            ['#00F260', '#0575E6'],            # 极光绿蓝
            ['#8E2DE2', '#4A00E0'],            # 魅影紫
            ['#00c3ff', '#ffff1c'],            # 天蓝亮黄
            ['#ee0979', '#ff6a00'],            # 热情红橙
            ['#DA22FF', '#9733EE'],            # 赛博紫
            ['#1D976C', '#93F9B9'],            # 翡翠绿
            ['#E55D87', '#5FC3E4']             # 糖果粉蓝
        ]
        # ##################################################################
        
        # ##################################################################
        # ################# 新增: 初始化图表窗口引用 ###################
        # ##################################################################
        self.strength_chart_window = None

        self._configure_styles()
        self._create_widgets()

        # ##################################################################
        # ###################### 新增: 绑定键盘移动事件 ######################
        # ##################################################################
        self.bind_all('<KeyPress>', self._handle_window_movement)
        
        # ##################################################################
        # ###################### 新增: 入场动画 - 步骤 3 #####################
        # ##################################################################
        # 在所有控件加载完毕后，开始执行淡入动画
        # 使用 self.after(10, ...) 确保在主循环开始后立即执行
        self.after(10, self._start_fade_in)

    # ##################################################################
    # ####################### 修改: 窗口移动事件处理函数 ###############
    # ##################################################################
    def _handle_window_movement(self, event):
        """处理键盘事件以移动主窗口"""
        # 如果图表窗口已打开，则主窗口不响应移动事件，由图表窗口自己处理
        if self.strength_chart_window and self.strength_chart_window.winfo_exists():
            return

        # 检查当前拥有焦点的控件是否是文本输入框
        focused_widget = self.focus_get()
        if isinstance(focused_widget, ttk.Entry):
            return  # 如果是输入框，则不执行任何操作，允许正常输入

        move_step = 15  # 每次按键移动的像素值
        x = self.winfo_x()
        y = self.winfo_y()

        key = event.keysym.lower()

        if key == 'w' or key == 'up':
            y -= move_step
        elif key == 's' or key == 'down':
            y += move_step
        elif key == 'a' or key == 'left':
            x -= move_step
        elif key == 'd' or key == 'right':
            x += move_step
        else:
            return # 忽略其他所有按键

        # 更新窗口位置，但不改变大小
        self.geometry(f"+{x}+{y}")

    # ##################################################################
    # ################# 关键修改: 打开图表窗口的函数 ###################
    # ##################################################################
    def _open_strength_chart(self):
        """打开或激活起手牌强度图表窗口"""
        # 如果窗口已存在，则将其提到顶层并给予焦点，避免重复创建
        if self.strength_chart_window and self.strength_chart_window.winfo_exists():
            self.strength_chart_window.lift()
            self.strength_chart_window.focus_force()
            return
        
        # ##################################################################
        # ################ 关键修改：禁用按钮防止重复点击 #################
        # ##################################################################
        self.open_chart_button.config(state='disabled')
        
        # 创建新窗口实例并保存引用
        self.strength_chart_window = StrengthChartWindow(self)

    # ##################################################################
    # ############ 新增: 辅助函数 - 将特定手牌转为范围类别 #############
    # ##################################################################
    def _specific_to_range_category(self, hand_str):
        """
        辅助函数：将特定的手牌字符串 (e.g., "AsKc") 转换为其范围类别 (e.g., "AKo")。
        返回 None 如果格式不正确。
        """
        try:
            if len(hand_str) != 4:
                return None
                
            r1, s1 = hand_str[0].upper(), hand_str[1].lower()
            r2, s2 = hand_str[2].upper(), hand_str[3].lower()
            
            ranks_order = 'AKQJT98765432'
            if r1 not in ranks_order or r2 not in ranks_order or s1 not in 'shdc' or s2 not in 'shdc':
                return None

            # 确保 r1 是高牌阶 (A > K)
            if ranks_order.index(r1) > ranks_order.index(r2):
                r1, r2 = r2, r1 # 交换牌阶
            
            if r1 == r2:
                return f"{r1}{r2}" # Pair, e.g., "AA"
            elif s1 == s2:
                return f"{r1}{r2}s" # Suited, e.g., "AKs"
            else:
                return f"{r1}{r2}o" # Offsuit, e.g., "AKo"
        except Exception:
            return None # 捕获任何异常，例如 index 找不到


    # ##################################################################
    # ###################### 关键修改: 缓动动画逻辑 ####################
    # ##################################################################
    def _start_fade_in(self):
        """(辅助函数) 初始化并启动淡入动画"""
        # 定义动画参数
        self.animation_total_duration = 300  # 总毫秒数 (0.3秒)
        self.animation_step_delay = 15       # 每一步的毫秒数 (约66 FPS)
        
        # 计算总步数和每一步的“进度”增量
        try:
            total_steps = self.animation_total_duration / self.animation_step_delay
        except ZeroDivisionError:
            total_steps = 0 # 避免除以零错误

        if total_steps == 0:
            self.wm_attributes('-alpha', 1.0)
            return
            
        # 这是“进度” (progress) 增量, 从 0.0 到 1.0
        self.progress_increment = 1.0 / total_steps
        self.current_progress = 0.0
        
        # 开始动画循环
        self._fade_in_step()

    def _fade_in_step(self):
        """(辅助函数) 执行淡入动画的单一步骤"""
        self.current_progress += self.progress_increment
        
        if self.current_progress >= 1.0:
            self.wm_attributes('-alpha', 1.0) # 确保最终为 1.0
        else:
            # ########################################################
            # ############### 这是关键的缓动（Ease-Out）逻辑 ##########
            # ########################################################
            # 使用 math.sin(progress * math.pi / 2) 来创建缓出效果
            # 当 self.current_progress 从 0.0 变为 1.0, 
            # eased_alpha 会从 sin(0) = 0.0 平滑地变为 sin(pi/2) = 1.0
            eased_alpha = math.sin(self.current_progress * (math.pi / 2))
            
            try:
                self.wm_attributes('-alpha', eased_alpha)
            except tk.TclError:
                # 窗口可能在动画过程中被关闭
                return
            
            # ########################################################
            
            # 安排下一步
            self.after(self.animation_step_delay, self._fade_in_step)
    # ##################################################################
    # ######################## 结束修改动画代码 ##########################
    # ##################################################################

    def _configure_styles(self):
        # --- (V7) 美化：统一字体为 "Microsoft YaHei" ---
        self.style.configure('.', background='#2e2e2e', foreground='white')
        self.style.configure('TFrame', background='#2e2e2e')
        self.style.configure('TLabel', background='#2e2e2e', foreground='white', font=('Microsoft YaHei', 10))
        self.style.configure('TLabelframe', background='#2e2e2e', bordercolor='#888')
        self.style.configure('TLabelframe.Label', background='#2e2e2e', foreground='white', font=('Microsoft YaHei', 11, 'bold'))
        self.style.configure('TEntry', fieldbackground='#4a4a4a', foreground='white', insertbackground='white')
        self.style.configure('Treeview', fieldbackground='#3c3c3c', background='#3c3c3c', foreground='white', rowheight=25)
        self.style.configure('Treeview.Heading', font=('Microsoft YaHei', 11, 'bold'), background='#4a4a4a', foreground='white')
        self.style.map('Treeview.Heading', background=[('active', '#6a6a6a')])
        
        # --- 进度条样式 (恢复标准样式) ---
        self.style.configure("p1.Horizontal.TProgressbar", background=self.P1_COLOR)
        self.style.configure("p2.Horizontal.TProgressbar", background=self.P2_COLOR)
        self.style.configure("tie.Horizontal.TProgressbar", background='#6c757d')
        
        # --- TTK 按钮样式 (V7) ---
        self.style.configure('TButton', background='#4a4a4a', foreground='white', font=('Microsoft YaHei', 10, 'bold'), borderwidth=1)
        self.style.map('TButton', background=[('active', '#6a6a6a'), ('disabled', '#3a3a3a')])
        
        # --- 移除 TTK 按钮的虚线焦点框 ---
        self.style.layout('TButton', [('Button.border', {'children': 
            [('Button.padding', {'children': 
                [('Button.label', {'side': 'left', 'expand': 1})]
            })]
        })])

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

        ttk.Label(player_setup_frame, text="玩家1 (手牌/范围):").grid(row=0, column=0, padx=10, pady=8, sticky='w')
        self.p1_hand_var = tk.StringVar()
        ttk.Entry(player_setup_frame, textvariable=self.p1_hand_var, font=('Microsoft YaHei', 9)).grid(row=0, column=1, padx=10, pady=8, sticky='ew')
        ttk.Button(player_setup_frame, text="重置", command=self._reset_player1, width=8).grid(row=0, column=2, padx=5, pady=8)

        ttk.Label(player_setup_frame, text="玩家2 (手牌/范围):").grid(row=1, column=0, padx=10, pady=8, sticky='w')
        self.p2_hand_var = tk.StringVar()
        ttk.Entry(player_setup_frame, textvariable=self.p2_hand_var, font=('Microsoft YaHei', 9)).grid(row=1, column=1, padx=10, pady=8, sticky='ew')
        ttk.Button(player_setup_frame, text="重置", command=self._reset_player2, width=8).grid(row=1, column=2, padx=5, pady=8)

        sim_frame = ttk.Frame(player_setup_frame)
        sim_frame.grid(row=2, column=0, columnspan=3, pady=5, sticky='ew')
        ttk.Label(sim_frame, text="模拟次数:").pack(side='left', padx=10)
        self.num_simulations_var = tk.StringVar(value="50000")
        ttk.Entry(sim_frame, textvariable=self.num_simulations_var, width=15).pack(side='left')

        self._create_board_selector(parent_pane)
        self._create_strength_display(parent_pane) 
        
        action_frame = ttk.Frame(parent_pane)
        action_frame.pack(side='bottom', pady=10, fill='x')
        
        # ##################################################################
        # ############### 修改: 使用自定义渐变进度条 #######################
        # ##################################################################
        # 初始随机选择一个主题
        initial_theme = random.choice(self.pb_themes)
        self.analysis_progress_bar = GradientProgressBar(
            action_frame, 
            color_list=initial_theme, 
            height=18
        )
        self.analysis_progress_bar.pack(fill='x', pady=(0, 8))
        
        # 添加追踪：当 self.progress_var 改变时，自动更新自定义进度条
        # *args 是必须的，因为 trace 回调会传递变数名等参数
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

        # ##################################################################
        # ############### 恢复: 使用标准 TTK 进度条 #######################
        # ##################################################################

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

        ttk.Label(result_grid, text="平局:", font=('Microsoft YaHei', 11, 'bold')).grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.tie_bar = ttk.Progressbar(result_grid, style="tie.Horizontal.TProgressbar")
        self.tie_bar.grid(row=2, column=1, sticky='ew', padx=5)
        self.tie_var = tk.StringVar(value="N/A")
        ttk.Label(result_grid, textvariable=self.tie_var, font=('Microsoft YaHei', 11, 'bold')).grid(row=2, column=2, sticky='e', padx=10)

        self._create_range_selector(parent_pane)
    
    def _create_board_selector(self, parent_frame):
        board_frame = ttk.LabelFrame(parent_frame, text="公共牌选择")
        board_frame.pack(fill='x', pady=10)
        display_frame = ttk.Frame(board_frame, relief='solid', borderwidth=1)
        display_frame.pack(fill='x', pady=5, padx=10)
        self.board_display_var = tk.StringVar(value="已选公共牌: ")
        ttk.Label(display_frame, textvariable=self.board_display_var, font=("Microsoft YaHei", 11, "bold")).pack(pady=5, side='left', padx=10)
        ttk.Button(display_frame, text="重置", command=self._reset_board_selector).pack(side='right', padx=5, pady=5)
        card_pool_frame = ttk.Frame(board_frame)
        card_pool_frame.pack(pady=5, padx=10)
        self.board_card_buttons = {}
        suits_map = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
        suit_colors = {'s': 'black', 'h': 'red', 'd': 'blue', 'c': 'green'}
        ranks = 'AKQJT98765432'
        
        # 创建渐变变色的辅助函数
        def create_board_hover_effect(btn, card_str):
            # 存储动画相关的数据
            btn.animation_data = {
                'card_str': card_str,
                'original_color': '#d0d0d0',
                'is_hovering': False,
                'current_step': 0,
                'timer_id': None
            }
            
            def hex_to_rgb(hex_color):
                """将hex颜色转换为RGB"""
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            def rgb_to_hex(rgb):
                """将RGB转换为hex颜色"""
                return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            
            def interpolate_color(color1, color2, factor):
                """在两个颜色之间插值"""
                rgb1 = hex_to_rgb(color1)
                rgb2 = hex_to_rgb(color2)
                result = tuple(rgb1[i] + (rgb2[i] - rgb1[i]) * factor for i in range(3))
                return rgb_to_hex(result)
            
            def get_original_color():
                """获取按钮的原始颜色"""
                card = btn.animation_data['card_str']
                # 如果按钮被禁用（已选中），返回灰色
                if card in self.board_cards:
                    return '#555555'
                else:
                    return '#d0d0d0'
            
            def animate_color():
                """执行颜色动画"""
                data = btn.animation_data
                total_steps = 15  # 总步数，数字越大动画越慢
                
                # 检查按钮是否被禁用
                if str(btn['state']) == 'disabled':
                    return  # 如果禁用了就不执行动画
                
                original_color = get_original_color()
                
                if data['is_hovering']:
                    # 悬停时，逐渐变亮
                    if data['current_step'] < total_steps:
                        data['current_step'] += 1
                        factor = data['current_step'] / total_steps
                        # 目标颜色：原色变亮（增加RGB值）
                        target = '#ffffff'
                        new_color = interpolate_color(original_color, target, factor * 0.35)
                        btn.configure(bg=new_color)
                        data['timer_id'] = btn.after(20, animate_color)  # 20ms后继续动画
                else:
                    # 离开时，逐渐恢复原色
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
                # 如果按钮被禁用就不执行动画
                if str(btn['state']) == 'disabled':
                    return
                data = btn.animation_data
                if data['timer_id']:
                    btn.after_cancel(data['timer_id'])
                data['is_hovering'] = True
                animate_color()
            
            def on_leave(event):
                data = btn.animation_data
                if data['timer_id']:
                    btn.after_cancel(data['timer_id'])
                data['is_hovering'] = False
                animate_color()
            
            btn.bind('<Enter>', on_enter)
            btn.bind('<Leave>', on_leave)
        
        # 创建公共牌按钮
        for i, suit_char in enumerate('shdc'):
            for j, rank_char in enumerate(ranks):
                card_str = f"{rank_char}{suit_char}"
                display_text = f"{rank_char}{suits_map[suit_char]}"
                # (V7) 美化：公共牌按钮字体保持 Arial 以获得最佳的 ♠ 符号渲染
                btn = tk.Button(card_pool_frame, text=display_text, font=('Arial', 10, 'bold'), 
                            width=4, fg=suit_colors[suit_char], bg='#d0d0d0', relief='raised', 
                            command=lambda s=card_str: self._on_board_card_select(s), takefocus=0)
                btn.grid(row=i, column=j, padx=1, pady=1)
                self.board_card_buttons[card_str] = btn
                # 添加悬停效果
                create_board_hover_effect(btn, card_str)

    def _create_strength_display(self, parent_frame):
        strength_frame = ttk.LabelFrame(parent_frame, text="玩家1 牌力分布")
        strength_frame.pack(fill='both', expand=True, pady=10)
        cols = ('牌型', '概率'); self.strength_tree = ttk.Treeview(strength_frame, columns=cols, show='headings', height=9)
        self.strength_tree.heading('牌型', text='牌型 (Hand Rank)'); self.strength_tree.heading('概率', text='概率 (Probability)')
        self.strength_tree.column('牌型', width=200, anchor='center'); self.strength_tree.column('概率', width=150, anchor='e')
        self.strength_tree.pack(fill='both', expand=True, padx=5, pady=5)

    def _create_range_selector(self, parent_frame):
        range_frame = ttk.LabelFrame(parent_frame, text="起手牌范围选择器")
        range_frame.pack(fill='both', expand=True, pady=5)
        radio_frame = ttk.Frame(range_frame); radio_frame.pack(anchor='w', padx=10, pady=5)
        self.active_player_for_range = tk.IntVar(value=1)
        
        # (V7) 美化：字体改为 "Microsoft YaHei"
        self.p1_radio_btn = tk.Button(radio_frame, text="为玩家1选择", relief='flat', bg=self.P1_COLOR, fg='white', font=('Microsoft YaHei', 9, 'bold'), borderwidth=0, activebackground='#0056b3', activeforeground='white', command=lambda: self._select_player_for_range(1), takefocus=0)
        self.p1_radio_btn.pack(side='left', padx=5, ipady=4)
        self.p2_radio_btn = tk.Button(radio_frame, text="为玩家2选择", relief='flat', bg='#4a4a4a', fg='white', font=('Microsoft YaHei', 9, 'bold'), borderwidth=0, activebackground='#6a6a6a', activeforeground='white', command=lambda: self._select_player_for_range(2), takefocus=0)
        self.p2_radio_btn.pack(side='left', padx=5, ipady=4)

        grid_frame = ttk.Frame(range_frame); grid_frame.pack(pady=10, padx=10)
        self.range_buttons = {}
        ranks = 'AKQJT98765432'
        # (V7) 美化：字体改为 "Microsoft YaHei"
        btn_font = font.Font(family='Microsoft YaHei', size=9, weight='bold')
        
        # 创建渐变变色的辅助函数
        def create_hover_effect(btn, hand_text):
            # 存储动画相关的数据
            btn.animation_data = {
                'hand_text': hand_text,
                'is_hovering': False,
                'current_step': 0,
                'timer_id': None
            }
            
            def hex_to_rgb(hex_color):
                """将hex颜色转换为RGB"""
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            def rgb_to_hex(rgb):
                """将RGB转换为hex颜色"""
                return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            
            def interpolate_color(color1, color2, factor):
                """在两个颜色之间插值"""
                rgb1 = hex_to_rgb(color1)
                rgb2 = hex_to_rgb(color2)
                result = tuple(rgb1[i] + (rgb2[i] - rgb1[i]) * factor for i in range(3))
                return rgb_to_hex(result)
            
            def get_original_color():
                """
                (新) 动态获取按钮的 *正确* 原始颜色
                这修复了一个bug：当按钮颜色因选择而改变时，悬停动画会恢复到 *旧的* 颜色。
                """
                hand = btn.animation_data['hand_text']
                
                # 1. 这是一个 'mini' 版的 _update_range_grid_colors
                # 我们需要实时检查文本框内容
                def get_categories(hand_var):
                    text = hand_var.get()
                    # (V6-Fix-3) 修复大小写问题
                    items = [s.strip() for s in text.split(',') if s.strip()]
                    categories = set()
                    for item in items:
                        if len(item) <= 3:
                            categories.add(item)
                        else:
                            cat = self._specific_to_range_category(item)
                            if cat: categories.add(cat)
                    return categories
                
                # 必须在函数调用时获取，否则 self.p1_hand_var 可能不存在
                p1_cats = get_categories(self.p1_hand_var)
                p2_cats = get_categories(self.p2_hand_var)

                # 2. 决定颜色
                in_p1 = hand in p1_cats
                in_p2 = hand in p2_cats

                if in_p1 and in_p2:
                    return self.BOTH_COLOR
                elif in_p1:
                    return self.P1_COLOR
                elif in_p2:
                    return self.P2_COLOR
                else:
                    if len(hand) == 2: 
                        return self.PAIR_BG
                    elif hand.endswith('s'): 
                        return self.SUITED_BG
                    else: 
                        return self.DEFAULT_BG
            
            def animate_color():
                """执行颜色动画"""
                data = btn.animation_data
                total_steps = 15  # 总步数，数字越大动画越慢
                
                if data['is_hovering']:
                    # 悬停时，目标是变亮。
                    if 'animation_start_color' not in data:
                         data['animation_start_color'] = get_original_color()
                    
                    original_color = data['animation_start_color']

                    if data['current_step'] < total_steps:
                        data['current_step'] += 1
                        factor = data['current_step'] / total_steps
                        target = '#ffffff'
                        new_color = interpolate_color(original_color, target, factor * 0.3)
                        btn.configure(bg=new_color)
                        data['timer_id'] = btn.after(20, animate_color)
                else:
                    # 离开时，逐渐恢复 *当前* 正确的颜色
                    
                    # ##########################################################
                    # #################### 关键修复 ############################
                    # ##########################################################
                    # 重新计算正确的 "原始" 颜色，以防在悬停期间它被更改了
                    current_correct_original_color = get_original_color()
                    # 动画开始时的颜色（如果不存在，则使用当前正确的颜色）
                    start_color = data.get('animation_start_color', current_correct_original_color)
                    # ##########################################################
                    
                    if data['current_step'] > 0:
                        data['current_step'] -= 1
                        factor = data['current_step'] / total_steps
                        target = '#ffffff'
                        
                        # 我们从 "start_color" 变亮的颜色 恢复
                        # 为了防止在“恢复”期间颜色发生跳跃（如果基础颜色已改变），
                        # 我们的插值基准应该是 current_correct_original_color
                        
                        new_color = interpolate_color(current_correct_original_color, target, factor * 0.3)

                        btn.configure(bg=new_color)
                        data['timer_id'] = btn.after(20, animate_color)
                    else:
                        # 动画结束，设置_最终_正确的颜色
                        btn.configure(bg=current_correct_original_color)
                        if 'animation_start_color' in data:
                            del data['animation_start_color'] # 清理
            
            def on_enter(event):
                data = btn.animation_data
                if data['timer_id']:
                    btn.after_cancel(data['timer_id'])
                data['is_hovering'] = True
                
                # ##########################################################
                # #################### 关键修复 ############################
                # ##########################################################
                # 在动画开始时捕获 *当前* 的原始颜色
                data['animation_start_color'] = get_original_color()
                # ##########################################################
                
                animate_color()
            
            def on_leave(event):
                data = btn.animation_data
                if data['timer_id']:
                    btn.after_cancel(data['timer_id'])
                data['is_hovering'] = False
                animate_color()
            
            btn.bind('<Enter>', on_enter)
            btn.bind('<Leave>', on_leave)
        
        # 创建表格
        for r in range(len(ranks)):
            for c in range(len(ranks)):
                if r < c: text, bg_color = f"{ranks[r]}{ranks[c]}s", self.SUITED_BG
                elif c < r: text, bg_color = f"{ranks[c]}{ranks[r]}o", self.DEFAULT_BG
                else: text, bg_color = f"{ranks[r]}{ranks[c]}", self.PAIR_BG
                btn = tk.Button(grid_frame, text=text, width=5, height=2, relief='raised', font=btn_font, fg='white', bg=bg_color, command=lambda t=text: self.toggle_range_button(t), takefocus=0)
                btn.grid(row=r, column=c, padx=1, pady=1)
                self.range_buttons[text] = btn
                # 添加悬停效果
                create_hover_effect(btn, text)

    # ##################################################################
    # ############### 修改: reset_player 方法 ##########################
    # ##################################################################
    def _reset_player1(self):
        self.p1_hand_var.set("")
        # self.range_selection_p1.clear() # <-- 已移除
        self._update_range_grid_colors()
    
    def _reset_player2(self):
        self.p2_hand_var.set("")
        # self.range_selection_p2.clear() # <-- 已移除
        self._update_range_grid_colors()
    # ##################################################################
    
    def _select_player_for_range(self, player_num):
        self.active_player_for_range.set(player_num)
        self.p1_radio_btn.config(bg=self.P1_COLOR if player_num == 1 else '#4a4a4a')
        self.p2_radio_btn.config(bg=self.P2_COLOR if player_num == 2 else '#4a4a4a')

    def _on_board_card_select(self, card_str):
        if len(self.board_cards) < 5 and card_str not in self.board_cards:
            self.board_cards.append(card_str)
            self.board_card_buttons[card_str].config(state='disabled', relief='sunken', bg='#555')
            self.board_display_var.set(f"已选公共牌: {' '.join(self.board_cards)}")

    def _reset_board_selector(self):
        self.board_cards = []
        for btn in self.board_card_buttons.values(): btn.config(state='normal', relief='raised', bg='#d0d0d0')
        self.board_display_var.set("已选公共牌: ")
    
    # ##################################################################
    # ############# 修改: toggle_range_button (V6-Fix-3) ###############
    # ##################################################################
    def toggle_range_button(self, hand_text):
        """
        (V6-Fix-3) 修复了 V6 的回调逻辑错误。
        - 修复了“未知”范围 (e.g., "AKo") 无法被添加的bug。
        - 修复了添加特定手牌 (e.g., "AsKh") 时与
          已存在的父范围 (e.g., "AKo") 之间的冲突逻辑。
        """
        active_player = self.active_player_for_range.get()
        entry_var = self.p1_hand_var if active_player == 1 else self.p2_hand_var
        current_text = entry_var.get()
        # (V6-Fix-3) 修复大小写问题：读取时保持原样
        current_items_list = [s.strip() for s in current_text.split(',') if s.strip()]
        current_items_set = set(current_items_list)

        # --- 步骤 1: 检查此类别是否已任何形式被选中 ---
        is_already_selected = False
        if hand_text in current_items_set:
            is_already_selected = True
        else:
            for item in current_items_set:
                if self._specific_to_range_category(item) == hand_text:
                    is_already_selected = True
                    break

        # --- 步骤 2: 执行新逻辑 ---

        if is_already_selected:
            # --- "取消"逻辑 (第二次点击) ---
            new_items_list = []
            for item in current_items_list:
                if item == hand_text:
                    continue # 移除范围 (e.g., "AKs")
                if self._specific_to_range_category(item) == hand_text:
                    continue # 移除特定组合 (e.g., "AsKs")
                
                new_items_list.append(item) # 保留其他所有牌

            entry_var.set(", ".join(new_items_list))
            self._update_range_grid_colors()

        else:
            # --- "打开选择器"逻辑 (第一次点击) ---
            
            # (V6-Fix) 在这里定义回调函数
            def handle_selection_callback(result_hand_str):
                if result_hand_str is None:
                    return # 用户取消了

                # (V6-Fix) 在回调 *内部* 获取变量，确保 player num 正确
                active_player_inner = self.active_player_for_range.get()
                entry_var_inner = self.p1_hand_var if active_player_inner == 1 else self.p2_hand_var
                current_text_inner = entry_var_inner.get()
                # (V6-Fix-3) 修复大小写问题：读取时保持原样
                current_items_list_inner = [s.strip() for s in current_text_inner.split(',') if s.strip()]
                
                # (V6-Fix) 必须使用 set 来进行添加/删除
                current_items_set_inner = set(current_items_list_inner)

                # (V6-Fix-3) 不在这里盲目 .upper()
                new_hands_to_process = [h.strip() for h in result_hand_str.split(',') if h.strip()]
                
                for hand_item in new_hands_to_process:
                    
                    # (V6-Fix-3) 范围(<=3)保持原样, 特定手牌(4)转为大写
                    new_hand = hand_item
                    if len(hand_item) == 4:
                        new_hand = hand_item.upper() # e.g., "AsKh" -> "ASKH"
                    # else: "AKo" 保持 "AKo"
                    
                    if new_hand in current_items_set_inner:
                        # (V6-Fix) 如果 V6 窗口返回了一个已存在的项，则移除它
                        current_items_set_inner.remove(new_hand)
                    
                    else:
                        # --- (V6-Fix) 关键逻辑：添加新项 ---
                        
                        if len(new_hand) <= 3:
                            # --- 1. 正在添加一个范围 (e.g., "AKo") ---
                            
                            # (V6-Fix-3) 使用 new_hand 作为父范围
                            parent_range = new_hand
                            items_to_remove = set()
                            for item in current_items_set_inner:
                                # (V6-Fix-3) 比较类别与父范围
                                if self._specific_to_range_category(item) == parent_range:
                                    items_to_remove.add(item)
                            
                            current_items_set_inner.difference_update(items_to_remove)
                            
                            # (V6-Fix-3) 添加正确的父范围 (e.g., "AKo")
                            current_items_set_inner.add(parent_range)

                        else:
                            # --- 2. 正在添加一个特定手牌 (e.g., "ASKH") ---
                            parent_range = self._specific_to_range_category(new_hand) # "AKo"
                            
                            if parent_range in current_items_set_inner:
                                # (V6-Fix) 关键：父范围存在！
                                # 移除父范围 (e.g., "AKo")
                                current_items_set_inner.remove(parent_range)
                                # 添加这个特定的子手牌 (e.g., "ASKH")
                                current_items_set_inner.add(new_hand)
                            
                            else:
                                # (V6-Fix) 父范围不存在，直接添加子手牌
                                current_items_set_inner.add(new_hand)

                # 4. 更新输入框 (从 set 转换回来)
                final_items = sorted(list(current_items_set_inner))
                entry_var_inner.set(", ".join(final_items))
                
                # 5. 更新网格颜色
                self.after(50, self._update_range_grid_colors)

            # --- 启动花色选择器 ---
            SuitSelectorWindow(self, hand_text, handle_selection_callback)

    # ##################################################################
    # ############# 修改: _update_range_grid_colors (V6-Fix-3) #########
    # ##################################################################
    def _update_range_grid_colors(self):
        
        # 1. 从输入框解析所有选中的项目
        def get_all_selected_categories(hand_var):
            text = hand_var.get()
            # (V6-Fix-3) 错误：不能在这里盲目 .upper()
            # items = [s.strip().upper() for s in text.split(',') if s.strip()]
            items = [s.strip() for s in text.split(',') if s.strip()] 
            categories = set()
            for item in items:
                if len(item) <= 3:
                    # (V6-Fix-3) item 是一个范围 (e.g., "AKo"), 直接添加
                    categories.add(item) 
                else:
                    # (V6-Fix-3) item 是特定手牌 (e.g., "ASKH"), 转换它
                    cat = self._specific_to_range_category(item) 
                    if cat:
                        categories.add(cat) # 添加 "AKo"
            return categories

        p1_categories = get_all_selected_categories(self.p1_hand_var)
        p2_categories = get_all_selected_categories(self.p2_hand_var)

        # 2. 更新按钮颜色
        for hand, btn in self.range_buttons.items():
            in_p1 = hand in p1_categories
            in_p2 = hand in p2_categories

            if in_p1 and in_p2:
                btn.config(bg=self.BOTH_COLOR, relief='sunken')
            elif in_p1:
                btn.config(bg=self.P1_COLOR, relief='sunken')
            elif in_p2:
                btn.config(bg=self.P2_COLOR, relief='sunken')
            else:
                # 恢复默认背景
                if len(hand) == 2: bg = self.PAIR_BG
                elif hand.endswith('s'): bg = self.SUITED_BG
                else: bg = self.DEFAULT_BG
                btn.config(bg=bg, relief='raised')

    # ##################################################################
    # ############### 删除: update_range_entry (不再需要) ##############
    # ##################################################################
    # def update_range_entry(self, player_num): # <-- 整个方法已删除
    #     ...
            
    def clear_all(self):
        self._reset_player1(); self._reset_player2(); self._reset_board_selector()
        for i in self.strength_tree.get_children(): self.strength_tree.delete(i)
        self._reset_equity_display()
                
    def _reset_equity_display(self):
        # (恢复) 重置标准 TTK 进度条
        self.p1_win_var.set("N/A"); self.p1_win_bar['value'] = 0
        self.p2_win_var.set("N/A"); self.p2_win_bar['value'] = 0
        self.tie_var.set("N/A"); self.tie_bar['value'] = 0
        
        # (修改) 重置自定义渐变进度条
        self.progress_var.set(0)

    def run_analysis_thread(self):
        try:
            num_sims = int(self.num_simulations_var.get())
            if num_sims <= 0: 
                if num_sims == 0:
                    # 允许 0 次模拟，直接清空结果
                    self._reset_equity_display()
                    for i in self.strength_tree.get_children(): self.strength_tree.delete(i)
                    return
                raise ValueError
        except ValueError:
            messagebox.showerror("输入错误", "模拟次数必须是一个非负整数。")
            return

        self._reset_equity_display()
        
        # --- (V12) 新增: 每次开始分析时随机切换进度条颜色 ---
        new_theme = random.choice(self.pb_themes)
        self.analysis_progress_bar.set_colors(new_theme)
        # -----------------------------------------
        
        # (修改) 使用自定义方法设置最大值
        max_val = num_sims if num_sims > 0 else 1
        self.analysis_progress_bar.set_max(max_val)
        
        self.calc_button.config(state='disabled')
        for i in self.strength_tree.get_children(): self.strength_tree.delete(i)
        
        self.analysis_result = None
        # 注意：这里仍然使用 threading.Thread，这是正确的！
        # 它的作用是防止UI线程被阻塞，而真正的并行计算由 PokerLogic 内部的 multiprocessing 处理。
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
                
                # (恢复) 更新标准 TTK 进度条
                self.p1_win_var.set(f"{equity['p1_win']:.2f}%")
                self.p1_win_bar['value'] = equity['p1_win']
                
                self.p2_win_var.set(f"{equity['p2_win']:.2f}%")
                self.p2_win_bar['value'] = equity['p2_win']
                
                self.tie_var.set(f"{equity['tie']:.2f}%")
                self.tie_bar['value'] = equity['tie']
                
                # 确保进度条在0模拟时也归零
                if int(self.num_simulations_var.get()) == 0:
                    self.progress_var.set(0)
                else:
                    # (修改) 确保进度条在计算完成后显示为100%
                    self.progress_var.set(self.analysis_progress_bar.max_val)

                # (V10) show_strength 现在总是 True
                if show_strength:
                    hand_rank_order = {v: k for k, v in self.poker_logic.rank_class_to_string_map.items()}
                    all_hand_types = sorted(hand_rank_order.keys(), key=lambda x: hand_rank_order[x])
                    for hand_name in all_hand_types:
                        prob = strength.get(hand_name, 0.0)
                        if prob > 1e-5: self.strength_tree.insert('', tk.END, values=(hand_name, f"{prob:.2f}%"))
                # (V10) 这个 else 永远不会被触发了，但保留也无妨
                else:
                    self.strength_tree.insert('', tk.END, values=("(请输入玩家1手牌/范围)", "N/A"))
            except Exception as e:
                # (V9) 这里是捕获和显示错误的地方
                messagebox.showerror("分析出错", f"错误: {e}")
                self._reset_equity_display()
            finally:
                self.calc_button.config(state='normal')

    def run_analysis_calculation(self):
        try:
            # ##################################################################
            # ############### (V9) 修改: 解析带逗号和空格的字符串 ###########
            # ##################################################################
            # (V9) 这里的 .upper() 是故意保留的，因为 _determine_input_type
            # 期望的是大写输入，以便正确解析 "AsKc" vs "asko"
            p1_input = [s.strip().upper() for s in self.p1_hand_var.get().split(',') if s.strip()]
            p2_input = [s.strip().upper() for s in self.p2_hand_var.get().split(',') if s.strip()]
            # ##################################################################
            
            board_input = "".join(self.board_cards)
            num_sims = int(self.num_simulations_var.get())
            
            def progress_update(current_sim):
                # 确保进度条不会超过最大值（在多进程回调中可能发生轻微的竞态）
                # (修改) 获取自定义进度条最大值
                max_sims = self.analysis_progress_bar.max_val      
                self.progress_var.set(min(current_sim, max_sims))

            self.analysis_result = self.poker_logic.run_analysis(
                p1_input, p2_input, board_input, 
                num_simulations=num_sims, 
                progress_callback=progress_update
            )
        except Exception as e:
            # (V9) 将错误传递给主线程
            self.analysis_result = e

if __name__ == "__main__":
    # ##########################################################################
    # ### 关键修复：显式设置多进程启动方法，解决打包后子进程无限递归的问题 ###
    # ##########################################################################
    
    # 1. 必须在所有其他代码之前调用 freeze_support()
    # 这是让打包后的 .exe 正常使用多进程的关键
    multiprocessing.freeze_support()

    # 2. 显式设置启动方法（保持你原有的设置）
    # 在Windows上，强制使用 'spawn' 模式，必须在创建任何进程或进程池之前调用。
    if sys.platform.startswith('win'):
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # 如果在某些特殊环境中方法已经被设置，忽略错误。
            pass
    
    # 3. 现在可以安全地启动你的应用
    app = PokerApp(PokerLogic())
    app.mainloop()