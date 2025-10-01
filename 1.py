import tkinter as tk
from tkinter import ttk, font
import random
import threading
from collections import defaultdict

# It's recommended to install the 'treys' library: pip install treys
try:
    from treys import Card, Evaluator, Deck
except ImportError:
    print("Error: 'treys' library not found. Please install it using 'pip install treys'")
    exit()

# --- Core Poker Logic Module (No changes needed here) ---
class PokerLogic:
    def __init__(self):
        self.evaluator = Evaluator()
        self.rank_class_to_string_map = {
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

    def _parse_hand_range(self, range_str_list):
        """Parse hand range strings into specific hand combinations"""
        hand_pairs = []
        
        for r_str in filter(None, [s.strip().upper() for s in range_str_list]):
            try:
                if len(r_str) == 2 and r_str[0] == r_str[1]:
                    # Pocket pair like "AA", "KK"
                    rank = r_str[0]
                    suits = 'shdc'
                    # Generate all 6 combinations for pocket pairs
                    for i in range(len(suits)):
                        for j in range(i + 1, len(suits)):
                            card1 = Card.new(f"{rank}{suits[i]}")
                            card2 = Card.new(f"{rank}{suits[j]}")
                            hand_pairs.append([card1, card2])
                            
                elif len(r_str) == 3:
                    # Suited/unsuited hands like "AKs", "AKo"
                    r1, r2, s = r_str[0], r_str[1], r_str[2].lower()
                    suits = 'shdc'
                    
                    if s == 's':  # Suited
                        # Generate 4 suited combinations
                        for suit in suits:
                            card1 = Card.new(f"{r1}{suit}")
                            card2 = Card.new(f"{r2}{suit}")
                            hand_pairs.append([card1, card2])
                    elif s == 'o':  # Offsuit
                        # Generate 12 offsuit combinations
                        for s1 in suits:
                            for s2 in suits:
                                if s1 != s2:
                                    card1 = Card.new(f"{r1}{s1}")
                                    card2 = Card.new(f"{r2}{s2}")
                                    hand_pairs.append([card1, card2])
                elif len(r_str) == 2:
                    # Handle cases like "AK" (both suited and unsuited)
                    r1, r2 = r_str[0], r_str[1]
                    suits = 'shdc'
                    
                    # Add all 16 combinations (4 suited + 12 offsuit)
                    for s1 in suits:
                        for s2 in suits:
                            if s1 != s2:  # Offsuit
                                card1 = Card.new(f"{r1}{s1}")
                                card2 = Card.new(f"{r2}{s2}")
                                hand_pairs.append([card1, card2])
                            else:  # Suited
                                # This case is covered by the implicit assumption, 
                                # but explicit handling is better for clarity.
                                # To avoid duplicates when parsing "AK" and "AKs", 
                                # a simple implementation might just create all 16.
                                # However, your original logic for "AK" was complex,
                                # so I'll simplify it to be more robust.
                                # Let's assume "AK" means both suited and offsuit.
                                pass # Suited handled in the s1 == s2 case below
                    
                    # Suited combinations
                    for suit in suits:
                        card1 = Card.new(f"{r1}{suit}")
                        card2 = Card.new(f"{r2}{suit}")
                        hand_pairs.append([card1, card2])
                            
            except (ValueError, KeyError) as e:
                print(f"Warning: Ignoring invalid range string '{r_str}': {e}")
                continue
                
        return hand_pairs

    def _split_hand_str(self, s):
        """Split hand string into individual cards"""
        return [s[i:i+2] for i in range(0, len(s), 2)]

    def _determine_input_type(self, p_input):
        """Determine if input is a specific hand, range, or random"""
        clean_input = [s.strip().upper() for s in p_input if s.strip()]
        
        if not clean_input:
            return 'random', None
            
        # Check if it's a specific hand (4 characters, 2 cards)
        if len(clean_input) == 1 and len(clean_input[0]) == 4 and all(c in '23456789TJQKA' or c in 'SHDC' for c in clean_input[0]):
            try:
                hand_str = clean_input[0]
                cards = self._split_hand_str(hand_str)
                hand = [Card.new(c) for c in cards]
                
                if len(set(hand)) != 2:
                    raise ValueError(f"手牌包含重复的牌: {hand_str}")
                    
                return 'hand', [hand]
                
            except ValueError as e:
                raise ValueError(f"无效的手牌字符串: {clean_input[0]} ({e})")
        else:
            # It's a range
            parsed_hands = self._parse_hand_range(clean_input)
            if not parsed_hands:
                raise ValueError("无法解析手牌范围")
            return 'range', parsed_hands

    def run_analysis(self, p1_input_raw, p2_input_raw, board_str, num_simulations=50000):
        """Run Monte Carlo simulation to calculate equity and hand strength"""
        
        # Stage 1: Input Parsing and Validation
        try:
            # Parse board cards
            if board_str:
                board = [Card.new(c) for c in self._split_hand_str(board_str)]
                if len(set(board)) != len(board):
                    raise ValueError("公共牌中包含重复的牌。")
            else:
                board = []
                
            board_set = set(board)

            # Parse player inputs
            p1_type, p1_hands = self._determine_input_type(p1_input_raw)
            p2_type, p2_hands = self._determine_input_type(p2_input_raw)
            
        except ValueError as e:
            raise ValueError(f"输入解析错误: {e}")

        calculate_p1_strength = p1_type != 'random'

        # Stage 2: Pre-simulation Conflict Check
        if p1_type == 'hand' and not board_set.isdisjoint(p1_hands[0]):
            raise ValueError("玩家1的手牌与公共牌冲突。")
        if p2_type == 'hand' and not board_set.isdisjoint(p2_hands[0]):
            raise ValueError("玩家2的手牌与公共牌冲突。")

        # Filter ranges based on board conflicts
        if p1_type == 'range':
            p1_hands = [h for h in p1_hands if board_set.isdisjoint(h)]
            if not p1_hands: 
                raise ValueError("玩家1的范围与公共牌完全冲突。")
                
        if p2_type == 'range':
            p2_hands = [h for h in p2_hands if board_set.isdisjoint(h)]
            if not p2_hands: 
                raise ValueError("玩家2的范围与公共牌完全冲突。")

        # Check for conflicts between players
        if p1_type == 'hand' and p2_type == 'hand':
            if not set(p1_hands[0]).isdisjoint(p2_hands[0]):
                raise ValueError("玩家1和玩家2的手牌存在冲突。")
        elif p1_type == 'hand' and p2_type == 'range':
            p1_hand_set = set(p1_hands[0])
            p2_hands = [h for h in p2_hands if p1_hand_set.isdisjoint(h)]
            if not p2_hands: 
                raise ValueError("玩家2的范围与玩家1的手牌完全冲突。")
        elif p2_type == 'hand' and p1_type == 'range':
            p2_hand_set = set(p2_hands[0])
            p1_hands = [h for h in p1_hands if p2_hand_set.isdisjoint(h)]
            if not p1_hands: 
                raise ValueError("玩家1的范围与玩家2的手牌完全冲突。")

        # Stage 3: Monte Carlo Simulation
        p1_wins, p2_wins, ties = 0, 0, 0
        p1_hand_strength_counts = defaultdict(int)
        valid_sims = 0
        total_p1_hands = 0

        for _ in range(num_simulations):
            # Create fresh deck
            deck = Deck()
            
            # Remove board cards from deck
            for card in board:
                if card in deck.cards:
                    deck.cards.remove(card)

            try:
                # Sample player 1 hand
                if p1_type == 'random':
                    p1_hand_sample = deck.draw(2)
                else:
                    p1_hand_sample = random.choice(p1_hands)
                    # Remove P1 cards from deck
                    for card in p1_hand_sample:
                        if card in deck.cards:
                            deck.cards.remove(card)
                        else:
                            # Card conflict, skip this simulation
                            continue

                # Sample player 2 hand
                if p2_type == 'random':
                    if len(deck.cards) < 2:
                        continue
                    p2_hand_sample = deck.draw(2)
                else:
                    # Filter P2 hands that don't conflict with P1
                    available_p2_hands = [h for h in p2_hands 
                                          if set(h).isdisjoint(p1_hand_sample)]
                    if not available_p2_hands:
                        continue
                    p2_hand_sample = random.choice(available_p2_hands)
                    # Remove P2 cards from deck
                    for card in p2_hand_sample:
                        if card in deck.cards:
                            deck.cards.remove(card)
                        else:
                            continue

                # Complete the board to 5 cards
                run_board = list(board)
                cards_needed = 5 - len(run_board)
                
                if len(deck.cards) < cards_needed:
                    continue
                    
                run_board.extend(deck.draw(cards_needed))

                # Evaluate hands
                p1_score = self.evaluator.evaluate(run_board, p1_hand_sample)
                p2_score = self.evaluator.evaluate(run_board, p2_hand_sample)
                
                # --- 牌型概率统计重构 ---
                if calculate_p1_strength:
                    # 找到玩家1的最佳5张牌组合
                    # treys没有直接给出最佳5张组合的API，但evaluate已是最优
                    # 统计最终牌型
                    p1_rank_class = self.evaluator.get_rank_class(p1_score)
                    if p1_rank_class in self.rank_class_to_string_map:
                        hand_type = self.rank_class_to_string_map[p1_rank_class]
                        p1_hand_strength_counts[hand_type] += 1
                    total_p1_hands += 1
                # --- 结束重构 ---
                
                # Determine winner (lower score = better hand in treys)
                if p1_score < p2_score:
                    p1_wins += 1
                elif p2_score < p1_score:
                    p2_wins += 1
                else:
                    ties += 1
                
                valid_sims += 1
                
            except (ValueError, IndexError):
                # Skip this simulation due to card conflicts or deck issues
                continue

        if valid_sims == 0:
            raise ValueError("无法完成任何有效模拟。请检查输入设置。")

        # Calculate results
        equity_results = {
            'p1_win': (p1_wins / valid_sims) * 100,
            'p2_win': (p2_wins / valid_sims) * 100,
            'tie': (ties / valid_sims) * 100
        }
        
        strength_results = {}
        if calculate_p1_strength and total_p1_hands > 0:
            # 按主流软件顺序输出所有牌型概率，未出现的为0
            for rank in range(1, 10):
                hand_type = self.rank_class_to_string_map[rank]
                prob = (p1_hand_strength_counts[hand_type] / total_p1_hands) * 100 if total_p1_hands else 0.0
                strength_results[hand_type] = prob
        
        return equity_results, strength_results, calculate_p1_strength

# --- GUI Application ---
class StrengthChartWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("起手牌强度图表")
        self.geometry("900x720")
        self.configure(bg='#2e2e2e')
        self.transient(master)
        self.grab_set()
        self._create_strength_chart()

    def _create_strength_chart(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill='both', expand=True)
        ttk.Label(main_frame, text="起手牌强度等级图表", font=("Arial", 16, "bold")).pack(pady=5)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(pady=10, fill='x', expand=True)
        grid_frame = ttk.Frame(content_frame)
        grid_frame.pack(side='left', padx=(0, 20), anchor='n')
        legend_frame = ttk.LabelFrame(content_frame, text="图例")
        legend_frame.pack(side='left', fill='y', anchor='n')
        hand_tiers = {
            "精英牌 (Elite)": ('#28a745', ['AA', 'KK', 'QQ', 'AKs']),
            "优质牌 (Premium)": ('#20c997', ['JJ', 'TT', 'AQs', 'AJs', 'KQs', 'AKo']),
            "强可玩牌 (Strong Playable)": ('#17a2b8', ['99', 'ATs', 'KJs', 'QJs', 'JTs', 'AQo']),
            "投机可玩牌 (Speculative Playable)": ('#ffc107', ['88', '77', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'KTs', 'QTs', 'J9s', 'T9s', '98s', 'AJo', 'KQo']),
            "潜力牌 (Potential Hands)": ('#fd7e14', ['66', '55', 'A4s', 'A3s', 'A2s', 'K9s', 'Q9s', 'J8s', 'T8s', '97s', '87s', '76s', 'ATo', 'KJo', 'QJo', 'JTo']),
            "边缘牌 (Marginal)": ('#6f42c1', ['KTo', 'QTo', '44', '33', '22', 'K8s', 'K7s', 'K6s', 'K5s', 'K4s', 'K3s', 'K2s', 'Q8s', 'J7s', 'T7s', '96s', '86s', '75s', '65s', '54s']),
            "弱牌 (Weak)": ('#6c757d', ['A9o', 'K9o', 'Q9o', 'J9o', 'T9o', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o']),
            "不建议游戏 (Fold)": ('#343a40', [])
        }
        hand_to_tier_color = {}
        for tier_name, (color, hands) in hand_tiers.items():
            for hand in hands: hand_to_tier_color[hand] = color
        for tier_name, (color, _) in hand_tiers.items():
            legend_item = ttk.Frame(legend_frame)
            legend_item.pack(anchor='w', padx=10, pady=5)
            tk.Label(legend_item, text=" ", bg=color, width=2, relief='solid', borderwidth=1).pack(side='left')
            ttk.Label(legend_item, text=f"  {tier_name}").pack(side='left')
        ranks = 'AKQJT98765432'
        btn_font = font.Font(family='Arial', size=10, weight='bold')
        for r in range(len(ranks)):
            for c in range(len(ranks)):
                if r < c: text = f"{ranks[r]}{ranks[c]}s"
                elif c < r: text = f"{ranks[c]}{ranks[r]}o"
                else: text = f"{ranks[r]}{ranks[c]}"
                bg_color = hand_to_tier_color.get(text, '#343a40')
                cell = tk.Label(grid_frame, text=text, font=btn_font, fg='white', bg=bg_color, width=5, height=2, relief='solid', borderwidth=1)
                cell.grid(row=r, column=c)
        strategy_frame = ttk.LabelFrame(main_frame, text="位置策略简介")
        strategy_frame.pack(fill='x', pady=15)
        strategy_text = ("位置是德州扑克中最重要的概念之一。你在牌桌上的位置决定了你的行动顺序。\n"
                         "通常来说，你的位置越靠后（越晚行动），你就可以用越宽（越多）的范围来游戏。\n\n"
                         "● 前位 (Early Position - EP): 你需要用最强的牌（如精英牌、优质牌）率先加注，因为你后面还有很多玩家未行动。\n"
                         "● 中位 (Middle Position - MP): 可以适当增加一些强可玩牌和投机牌。\n"
                         "● 后位 (Late Position - CO/BTN): 这是最好的位置。你可以用更宽的范围加注，包括很多潜力牌和边缘牌，以攻击盲注。")
        ttk.Label(strategy_frame, text=strategy_text, wraplength=800, justify='left').pack(padx=10, pady=10)

class PokerApp(tk.Tk):
    def __init__(self, poker_logic):
        super().__init__()
        self.poker_logic = poker_logic
        self.title("德州扑克分析")
        self.geometry("1270x900")
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.configure(bg='#2e2e2e')
        self._configure_styles()
        self.range_selection_p1 = set()
        self.range_selection_p2 = set()
        self.board_cards = []
        self.analysis_thread = None
        self.analysis_result = None
        self._create_widgets()

    def _configure_styles(self):
        self.style.configure('.', background='#2e2e2e', foreground='white')
        self.style.configure('TFrame', background='#2e2e2e')
        self.style.configure('TLabel', background='#2e2e2e', foreground='white', font=('Arial', 10))
        self.style.configure('TLabelframe', background='#2e2e2e', bordercolor='#888')
        self.style.configure('TLabelframe.Label', background='#2e2e2e', foreground='white', font=('Arial', 11, 'bold'))
        self.style.configure('TButton', background='#4a4a4a', foreground='white', font=('Arial', 10, 'bold'), borderwidth=1)
        self.style.map('TButton', background=[('active', '#6a6a6a')])
        self.style.configure('TEntry', fieldbackground='#4a4a4a', foreground='white', insertbackground='white')
        self.style.configure('Treeview', fieldbackground='#3c3c3c', background='#3c3c3c', foreground='white', rowheight=25)
        self.style.configure('Treeview.Heading', font=('Arial', 11, 'bold'), background='#4a4a4a', foreground='white')
        self.style.map('Treeview.Heading', background=[('active', '#6a6a6a')])

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill='both', expand=True)
        left_pane = ttk.Frame(main_frame)
        left_pane.pack(side='left', fill='both', expand=True, padx=(0, 10))
        right_pane = ttk.Frame(main_frame)
        right_pane.pack(side='left', fill='both', expand=True)
        self._create_control_pane(left_pane)
        self._create_analysis_pane(right_pane)

    # --- MODIFIED METHOD ---
    def _create_control_pane(self, parent_pane):
        parent_pane.columnconfigure(0, weight=1)
        parent_pane.rowconfigure(2, weight=1)
        
        player_setup_frame = ttk.LabelFrame(parent_pane, text="玩家设置")
        player_setup_frame.pack(fill='x', pady=5)
        
        # Player 1 setup
        ttk.Label(player_setup_frame, text="玩家1 (手牌/范围):").grid(row=0, column=0, padx=10, pady=8, sticky='w')
        self.p1_hand_var = tk.StringVar()
        ttk.Entry(player_setup_frame, textvariable=self.p1_hand_var, width=35, font=('Arial', 9)).grid(row=0, column=1, padx=10, pady=8)
        # --- NEW WIDGET ---
        ttk.Button(player_setup_frame, text="重置", command=self._reset_player1, width=8).grid(row=0, column=2, padx=5, pady=8)

        # Player 2 setup
        ttk.Label(player_setup_frame, text="玩家2 (手牌/范围):").grid(row=1, column=0, padx=10, pady=8, sticky='w')
        self.p2_hand_var = tk.StringVar()
        ttk.Entry(player_setup_frame, textvariable=self.p2_hand_var, width=35, font=('Arial', 9)).grid(row=1, column=1, padx=10, pady=8)
        # --- NEW WIDGET ---
        ttk.Button(player_setup_frame, text="重置", command=self._reset_player2, width=8).grid(row=1, column=2, padx=5, pady=8)

        self._create_board_selector(parent_pane)
        self._create_strength_display(parent_pane)
        
        action_frame = ttk.Frame(parent_pane)
        action_frame.pack(side='bottom', pady=10, fill='x')
        
        self.calc_button = ttk.Button(action_frame, text="开始分析", command=self.run_analysis_thread)
        self.calc_button.pack(fill='x', ipady=10, pady=5)
        
        ttk.Button(action_frame, text="清空全部", command=self.clear_all).pack(fill='x', ipady=10, pady=5)
        ttk.Button(parent_pane, text="查看起手牌强度图表", command=self.open_strength_chart).pack(side='bottom', fill='x', ipady=10, pady=5)

    def _create_analysis_pane(self, parent_pane):
        parent_pane.rowconfigure(1, weight=1)
        equity_frame = ttk.LabelFrame(parent_pane, text="胜率分析")
        equity_frame.pack(fill='x', pady=10, anchor='n')
        self.equity_result_var = tk.StringVar(value="请设置牌局并点击'开始分析'")
        ttk.Label(equity_frame, textvariable=self.equity_result_var, font=("Arial", 13, "bold"), anchor='center', wraplength=500).pack(fill='x', pady=20)
        self._create_range_selector(parent_pane)
        
    def _create_board_selector(self, parent_frame):
        board_frame = ttk.LabelFrame(parent_frame, text="公共牌选择")
        board_frame.pack(fill='x', pady=10)
        display_frame = ttk.Frame(board_frame, relief='solid', borderwidth=1)
        display_frame.pack(fill='x', pady=5, padx=10)
        self.board_display_var = tk.StringVar(value="已选公共牌: ")
        ttk.Label(display_frame, textvariable=self.board_display_var, font=("Arial", 11, "bold")).pack(pady=5, side='left', padx=10)
        ttk.Button(display_frame, text="重置", command=self._reset_board_selector).pack(side='right', padx=5, pady=5)
        card_pool_frame = ttk.Frame(board_frame)
        card_pool_frame.pack(pady=5, padx=10)
        self.board_card_buttons = {}
        suits_map = {'s': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
        suit_colors = {'s': 'black', 'h': 'red', 'd': 'blue', 'c': 'green'}
        ranks = 'AKQJT98765432'
        for i, suit_char in enumerate('shdc'):
            for j, rank_char in enumerate(ranks):
                card_str = f"{rank_char}{suit_char}"
                display_text = f"{rank_char}{suits_map[suit_char]}"
                btn = tk.Button(card_pool_frame, text=display_text, font=('Arial', 10, 'bold'), width=4, fg=suit_colors[suit_char], bg='#d0d0d0', relief='raised', command=lambda s=card_str: self._on_board_card_select(s))
                btn.grid(row=i, column=j, padx=1, pady=1)
                self.board_card_buttons[card_str] = btn

    def _create_strength_display(self, parent_frame):
        strength_frame = ttk.LabelFrame(parent_frame, text="玩家1 牌力分布")
        strength_frame.pack(fill='both', expand=True, pady=10)
        cols = ('牌型', '概率')
        self.strength_tree = ttk.Treeview(strength_frame, columns=cols, show='headings', height=9)
        self.strength_tree.heading('牌型', text='牌型 (Hand Rank)')
        self.strength_tree.heading('概率', text='概率 (Probability)')
        self.strength_tree.column('牌型', width=200, anchor='center')
        self.strength_tree.column('概率', width=150, anchor='e')
        self.strength_tree.pack(fill='both', expand=True, padx=5, pady=5)

    def _create_range_selector(self, parent_frame):
        range_frame = ttk.LabelFrame(parent_frame, text="起手牌范围选择器")
        range_frame.pack(fill='both', expand=True, pady=5)
        radio_frame = ttk.Frame(range_frame); radio_frame.pack(anchor='w', padx=10, pady=5)
        self.active_player_for_range = tk.IntVar(value=1)
        self.p1_radio_btn = tk.Button(radio_frame, text="为玩家1选择", relief='flat', bg='#007bff', fg='white', font=('Arial', 9, 'bold'), borderwidth=0, activebackground='#0056b3', activeforeground='white', command=lambda: self._select_player_for_range(1))
        self.p1_radio_btn.pack(side='left', padx=5, ipady=4)
        self.p2_radio_btn = tk.Button(radio_frame, text="为玩家2选择", relief='flat', bg='#4a4a4a', fg='white', font=('Arial', 9, 'bold'), borderwidth=0, activebackground='#6a6a6a', activeforeground='white', command=lambda: self._select_player_for_range(2))
        self.p2_radio_btn.pack(side='left', padx=5, ipady=4)
        grid_frame = ttk.Frame(range_frame); grid_frame.pack(pady=10, padx=10)
        self.range_buttons = {}
        ranks = 'AKQJT98765432'
        btn_font = font.Font(family='Arial', size=9, weight='bold')
        for r in range(len(ranks)):
            for c in range(len(ranks)):
                if r < c: text, bg_color = f"{ranks[r]}{ranks[c]}s", '#4a7a96'
                elif c < r: text, bg_color = f"{ranks[c]}{ranks[r]}o", '#4f4f4f'
                else: text, bg_color = f"{ranks[r]}{ranks[c]}", '#8fbc8f'
                btn = tk.Button(grid_frame, text=text, width=5, height=2, relief='raised', font=btn_font, fg='white', bg=bg_color, command=lambda t=text: self.toggle_range_button(t))
                btn.grid(row=r, column=c, padx=1, pady=1)
                self.range_buttons[text] = btn

    # --- NEW METHODS ---
    def _reset_player1(self):
        """Clears the hand/range for Player 1."""
        self.p1_hand_var.set("")
        # Deselect any buttons associated with player 1's range
        for hand in self.range_selection_p1:
            if hand in self.range_buttons:
                self.range_buttons[hand].config(relief='raised')
        self.range_selection_p1.clear()

    def _reset_player2(self):
        """Clears the hand/range for Player 2."""
        self.p2_hand_var.set("")
        # Deselect any buttons associated with player 2's range
        for hand in self.range_selection_p2:
            if hand in self.range_buttons:
                self.range_buttons[hand].config(relief='raised')
        self.range_selection_p2.clear()

    def open_strength_chart(self):
        StrengthChartWindow(self)

    def _select_player_for_range(self, player_num):
        self.active_player_for_range.set(player_num)
        self.p1_radio_btn.config(bg='#007bff' if player_num == 1 else '#4a4a4a')
        self.p2_radio_btn.config(bg='#007bff' if player_num == 2 else '#4a4a4a')

    def _on_board_card_select(self, card_str):
        if len(self.board_cards) < 5 and card_str not in self.board_cards:
            self.board_cards.append(card_str)
            self.board_card_buttons[card_str].config(state='disabled', relief='sunken', bg='#555')
            self._update_board_display()

    def _update_board_display(self):
        self.board_display_var.set(f"已选公共牌: {' '.join(self.board_cards)}")

    def _reset_board_selector(self):
        self.board_cards = []
        for btn in self.board_card_buttons.values(): 
            btn.config(state='normal', relief='raised', bg='#d0d0d0')
        self._update_board_display()

    def toggle_range_button(self, hand_text):
        active_player = self.active_player_for_range.get()
        active_selection = self.range_selection_p1 if active_player == 1 else self.range_selection_p2
        if hand_text in active_selection:
            active_selection.remove(hand_text)
            self.range_buttons[hand_text].config(relief='raised')
        else:
            active_selection.add(hand_text)
            self.range_buttons[hand_text].config(relief='sunken')
        self.update_range_entry(active_player)

    def update_range_entry(self, player_num):
        selection = self.range_selection_p1 if player_num == 1 else self.range_selection_p2
        entry_var = self.p1_hand_var if player_num == 1 else self.p2_hand_var
        if selection:
            # A more logical sorting for ranges
            ranks = 'AKQJT98765432'
            def sort_key(hand):
                if len(hand) == 2: # Pair
                    return (ranks.index(hand[0]), ranks.index(hand[1]), 2)
                elif hand.endswith('s'): # Suited
                    return (ranks.index(hand[0]), ranks.index(hand[1]), 0)
                else: # Offsuit
                    return (ranks.index(hand[0]), ranks.index(hand[1]), 1)
            sorted_range = sorted(list(selection), key=sort_key)
            entry_var.set(",".join(sorted_range))
        else:
            entry_var.set("")
            
    def clear_all(self):
        self._reset_player1()
        self._reset_player2()
        self.equity_result_var.set("请设置牌局并点击'开始分析'")
        for i in self.strength_tree.get_children(): self.strength_tree.delete(i)
        self._reset_board_selector()
                
    def run_analysis_thread(self):
        self.equity_result_var.set("正在分析中，请稍候...")
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
                if isinstance(self.analysis_result, Exception):
                    raise self.analysis_result
                equity, strength, show_strength = self.analysis_result
                result_str = f"玩家1胜率: {equity['p1_win']:.2f}%  |  玩家2胜率: {equity['p2_win']:.2f}%  |  平局: {equity['tie']:.2f}%"
                self.equity_result_var.set(result_str)
                if show_strength:
                    # Display hand strengths in order
                    hand_rank_order = {v: k for k, v in self.poker_logic.rank_class_to_string_map.items()}
                    all_hand_types = sorted(hand_rank_order.keys(), key=lambda x: hand_rank_order[x])
                    
                    has_results = False
                    for hand_name in all_hand_types:
                        prob = strength.get(hand_name, 0.0)
                        if prob > 1e-5: # Use a small epsilon to avoid floating point issues
                            self.strength_tree.insert('', tk.END, values=(hand_name, f"{prob:.2f}%"))
                            has_results = True
                    
                    if not has_results:
                        self.strength_tree.insert('', tk.END, values=("无数据", "0.00%"))
                else:
                    self.strength_tree.insert('', tk.END, values=("(请输入玩家1手牌/范围)", "N/A"))
            except Exception as e:
                self.equity_result_var.set(f"错误: {e}")
            finally:
                self.calc_button.config(state='normal')

    def run_analysis_calculation(self):
        try:
            p1_input = self.p1_hand_var.get().split(',')
            p2_input = self.p2_hand_var.get().split(',')
            board_input = "".join(self.board_cards)
            self.analysis_result = self.poker_logic.run_analysis(p1_input, p2_input, board_input, num_simulations=50000)
        except Exception as e:
            self.analysis_result = e

if __name__ == "__main__":
    app = PokerApp(PokerLogic())
    app.mainloop()