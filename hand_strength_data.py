# 起手牌强度表格数据

HAND_TIERS = {
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
