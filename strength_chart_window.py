import tkinter as tk
from tkinter import ttk, font
import math
import webbrowser
from hand_strength_data import HAND_TIERS

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

        hand_tiers = HAND_TIERS

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
