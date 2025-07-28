import tkinter as tk
from tkinter import ttk, font
import json
import os
import sys

def get_resource_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    else:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)



class EquityHeatmapApp(tk.Tk):
    """
    一个用于即时显示预先计算好的德州扑克翻前胜率热力图的GUI应用。
    此版本增加了清晰的横轴和纵轴。
    """

    def __init__(self):
        super().__init__()
        self.title("德州扑克胜率热力图")
        self.geometry("900x800") 
        
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.configure(bg='#2e2e2e')
        self.style.configure('.', background='#2e2e2e', foreground='white')
        self.style.configure('TFrame', background='#2e2e2e')
        self.style.configure('TLabel', background='#2e2e2e', foreground='white', font=('Arial', 12))
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Axis.TLabel', font=('Arial', 11, 'bold')) # 坐标轴标签样式

        self.ranks = 'AKQJT98765432'
        self.grid_buttons = {}
        
        # 加载预计算的数据库
        self.equity_database = self._load_database()
        
        self._create_widgets()

        if not self.equity_database:
            # 如果未找到数据库，将错误信息显示在顶部标题处
            self.title_label.config(text="错误: 未找到 'full_equity_database.json'。请先运行生成脚本。")
        else:
            self.title_label.config(text="请在下方网格中点击一手牌开始分析")

    def _load_database(self):
        """在启动时加载预计算的JSON数据库文件。"""
        json_path = get_resource_path('full_equity_database.json')
        if not os.path.exists(json_path):
            return None
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                print(f"成功从 '{json_path}' 加载数据库。")
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"错误: 无法加载或解析数据库文件 '{json_path}'. 错误: {e}")
            return None

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill='both', expand=True)

        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.pack(fill='x')
        self.title_label = ttk.Label(control_frame, text="", style='Title.TLabel', anchor='center')
        self.title_label.pack()

        # --- 创建带坐标轴的网格 ---
        grid_container = ttk.Frame(main_frame, padding="10")
        grid_container.pack(pady=8)
        
        btn_font = font.Font(family='Arial', size=10, weight='bold')

        # 添加顶部横坐标轴
        for i, rank in enumerate(self.ranks):
            axis_label = ttk.Label(grid_container, text=rank, style='Axis.TLabel', anchor='center')
            axis_label.grid(row=0, column=i + 1, sticky='nsew', padx=1, pady=1)

        # 添加左侧纵坐标轴
        for i, rank in enumerate(self.ranks):
            axis_label = ttk.Label(grid_container, text=rank, style='Axis.TLabel', anchor='center')
            axis_label.grid(row=i + 1, column=0, sticky='nsew', padx=1, pady=1)

        # 创建13x13的按钮网格，并放置在坐标轴内部
        for r_idx, r1 in enumerate(self.ranks):
            for c_idx, r2 in enumerate(self.ranks):
                if r_idx < c_idx: text, bg_color = f"{r1}{r2}s", '#4a7a96'
                elif c_idx < r_idx: text, bg_color = f"{self.ranks[c_idx]}{self.ranks[r_idx]}o", '#4f4f4f'
                else: text, bg_color = f"{r1}{r2}", '#8fbc8f'
                
                btn = tk.Button(
                    grid_container, text=text, font=btn_font, fg='white', bg=bg_color,
                    width=6, height=2, relief='raised', borderwidth=2,
                    command=lambda t=text: self.display_heatmap(t)
                )
                # 将按钮放置在 (r+1, c+1) 的位置
                btn.grid(row=r_idx + 1, column=c_idx + 1, padx=1, pady=1)
                self.grid_buttons[text] = btn
        
        # --- 核心改动：移除了底部的 status_label ---

    def display_heatmap(self, hero_hand_text):
        """根据英雄手牌，从数据库中提取数据并立即更新UI。"""
        if not self.equity_database:
            print("数据库未加载，无法显示热力图。")
            return
        
        self._reset_grid_visuals()
        self.title_label.config(text=f"Hero: {hero_hand_text}")

        results = self.equity_database.get(hero_hand_text)
        if not results:
            print(f"在数据库中未找到手牌 '{hero_hand_text}' 的数据。")
            return

        for villain_hand, equity in results.items():
            self._update_grid_cell(villain_hand, equity)
        
        self._mark_hero_cell(hero_hand_text)

    def _get_color_for_equity(self, equity):
        e = max(0, min(100, equity)) / 100.0
        blue, yellow, red = (23, 92, 201), (255, 255, 224), (214, 40, 40)
        if e < 0.5:
            t = e * 2; r, g, b = [int(blue[i]*(1-t) + yellow[i]*t) for i in range(3)]
        else:
            t = (e - 0.5) * 2; r, g, b = [int(yellow[i]*(1-t) + red[i]*t) for i in range(3)]
        return f'#{r:02x}{g:02x}{b:02x}'

    def _get_text_color_for_bg(self, bg_hex):
        r, g, b = [int(bg_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]
        return '#000000' if (0.299*r + 0.587*g + 0.114*b) > 150 else '#FFFFFF'

    def _reset_grid_visuals(self):
        for hand, button in self.grid_buttons.items():
            button.config(text=hand, fg='white')
            if len(hand) == 3 and hand.endswith('s'): button.config(bg='#4a7a96')
            elif len(hand) == 3 and hand.endswith('o'): button.config(bg='#4f4f4f')
            else: button.config(bg='#8fbc8f')

    def _update_grid_cell(self, hand_text, equity):
        button = self.grid_buttons.get(hand_text)
        if button:
            bg_color = self._get_color_for_equity(equity)
            fg_color = self._get_text_color_for_bg(bg_color)
            button.config(bg=bg_color, fg=fg_color, text=f"{equity:.1f}%")

    def _mark_hero_cell(self, hero_hand):
        hero_btn = self.grid_buttons.get(hero_hand)
        if hero_btn:
            hero_btn.config(text="HERO", bg='#FFD700', fg='#000000')

if __name__ == "__main__":
    app = EquityHeatmapApp()
    app.mainloop()
