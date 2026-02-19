import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time
from collections import deque
from pathlib import Path
import pandas as pd

# Pasta para todos os ficheiros CSV
DATA_DIR = Path("data")
from analise.gui_integration import integrate_comparison_system

# Importar as classes do pacote abordagem
from abordagem.abordagem_a import (
    ApproachASimulation, BaselineSimulation, Environment, BaselineA_Greedy,
    compare_approaches, plot_comparison_results
)
from abordagem.abordagem_b import (
    ApproachBSimulation, BaselineB_BFS, EnvironmentB,
    compare_approaches_b
)
from abordagem.abordagem_c import (
    ApproachCSimulation, BaselineC_AStar, EnvironmentC,
    compare_approaches_c
)

class IAProjectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Projecto IA - Exploração Colaborativa de Ambientes")
        self.root.geometry("1400x900")
        
        # Configurar estilo
        self.setup_styles()
        
        # Variáveis de controle
        self.simulation_thread = None
        self.stop_simulation = False
        self.current_simulation = None
        self.simulation_results = []
        self.log_history = deque(maxlen=1000)
        self.start_time = None
        
        # Criar interface
        self.setup_ui()
        self.comparison_ext = integrate_comparison_system(self)
        
    def setup_styles(self):
        """Configurar estilos para a interface"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Cores
        self.bg_color = "#f0f0f0"
        self.header_color = "#2c3e50"
        self.button_color = "#3498db"
        self.success_color = "#27ae60"
        self.danger_color = "#e74c3c"
        self.warning_color = "#f39c12"
        
        self.root.configure(bg=self.bg_color)
        
    def setup_ui(self):
        """Configurar todos os elementos da interface"""
        
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # ===== TÍTULO =====
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=3, pady=(0, 20), sticky=(tk.W, tk.E))
        
        title_label = tk.Label(
            title_frame,
            text=" PROJETO DE INTELIGÊNCIA ARTIFICIAL",
            font=("Arial", 24, "bold"),
            fg=self.header_color,
            bg=self.bg_color
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Exploração Colaborativa de Ambientes Desconhecidos",
            font=("Arial", 14),
            fg="#7f8c8d",
            bg=self.bg_color
        )
        subtitle_label.pack()
        
        # ===== PAINEL DE CONTROLE =====
        control_frame = ttk.LabelFrame(main_frame, text="🎮 CONTROLE DA SIMULAÇÃO", padding="15")
        control_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))
        
        # Botões de abordagem
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(btn_frame, text="SELECIONAR ABORDAGEM:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        self.approach_var = tk.StringVar(value="A")
        
        approach_buttons = ttk.Frame(btn_frame)
        approach_buttons.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            approach_buttons,
            text=" ABORDAGEM A",
            command=lambda: self.select_approach("A"),
            style="ApproachA.TButton"
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        ttk.Button(
            approach_buttons,
            text=" ABORDAGEM B",
            command=lambda: self.select_approach("B"),
            style="ApproachB.TButton"
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        ttk.Button(
            approach_buttons,
            text=" ABORDAGEM C",
            command=lambda: self.select_approach("C"),
            style="ApproachC.TButton"
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Configurações da simulação
        config_frame = ttk.LabelFrame(control_frame, text=" CONFIGURAÇÕES", padding="10")
        config_frame.pack(fill=tk.X, pady=10)
        
        # Número de agentes
        ttk.Label(config_frame, text="Número de Agentes:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.agent_spinbox = ttk.Spinbox(config_frame, from_=2, to=10, width=15)
        self.agent_spinbox.set(4)
        self.agent_spinbox.grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # Proporção de bombas
        ttk.Label(config_frame, text="Proporção de Bombas (%):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.bomb_scale = ttk.Scale(config_frame, from_=20, to=80, length=150)
        self.bomb_scale.set(20)
        self.bomb_label = ttk.Label(config_frame, text="50%")
        self.bomb_scale.config(command=self.update_bomb_label)
        self.bomb_scale.grid(row=1, column=1, sticky=tk.W, pady=5)
        self.bomb_label.grid(row=1, column=2, padx=(5, 0))
        
        # Número de tesouros
        ttk.Label(config_frame, text="Número de Tesouros:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.treasure_spinbox = ttk.Spinbox(config_frame, from_=10, to=30, width=15)
        self.treasure_spinbox.set(10)
        self.treasure_spinbox.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Tipo de grupo
        ttk.Label(config_frame, text="Tipo de Grupo:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.group_type = tk.StringVar(value="heterogeneous")
        ttk.Radiobutton(config_frame, text="Homogêneo", variable=self.group_type, value="homogeneous").grid(row=3, column=1, sticky=tk.W)
        ttk.Radiobutton(config_frame, text="Heterogêneo", variable=self.group_type, value="heterogeneous").grid(row=4, column=1, sticky=tk.W)
        ttk.Radiobutton(config_frame, text="Baseline", variable=self.group_type, value="baseline").grid(row=5, column=1, sticky=tk.W)
        
        # Máximo de passos
        ttk.Label(config_frame, text="Máximo de Passos:").grid(row=6, column=0, sticky=tk.W, pady=5)
        self.max_steps_spinbox = ttk.Spinbox(config_frame, from_=100, to=5000, width=15)
        self.max_steps_spinbox.set(150)
        self.max_steps_spinbox.grid(row=6, column=1, sticky=tk.W, pady=5)
        
        # Botões de controle
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=20)
        
        self.start_button = ttk.Button(
            button_frame,
            text="▶️ INICIAR SIMULAÇÃO",
            command=self.start_simulation,
            style="Start.TButton"
        )
        self.start_button.pack(fill=tk.X, pady=5)
        
        self.stop_button = ttk.Button(
            button_frame,
            text="⏹️ PARAR SIMULAÇÃO",
            command=self.stop_simulation_func,
            style="Stop.TButton",
            state=tk.DISABLED
        )
        self.stop_button.pack(fill=tk.X, pady=5)
        
        self.compare_button = ttk.Button(
            button_frame,
            text="📊 COMPARAR ABORDAGENS",
            command=self.run_comparison
        )
        self.compare_button.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame,
            text=" LIMPAR LOGS",
            command=self.clear_logs
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame,
            text=" EXPORTAR DADOS",
            command=self.export_data
        ).pack(fill=tk.X, pady=5)
        
        # Painel de status
        status_frame = ttk.LabelFrame(control_frame, text="📈 STATUS", padding="10")
        status_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.status_labels = {}
        
        metrics = [
            ("Tesouros Encontrados:", "0/0"),
            ("Agentes Vivos:", "0/0"),
            ("Passos Executados:", "0"),
            ("Tempo de Execução:", "0.00s"),
            ("Status:", "Aguardando...")
        ]
        
        for i, (label, value) in enumerate(metrics):
            ttk.Label(status_frame, text=label, font=("Arial", 9)).grid(row=i, column=0, sticky=tk.W, pady=3)
            self.status_labels[label] = ttk.Label(status_frame, text=value, font=("Arial", 9, "bold"))
            self.status_labels[label].grid(row=i, column=1, sticky=tk.W, pady=3, padx=(10, 0))
        
        # ===== PAINEL DE VISUALIZAÇÃO =====
        visual_frame = ttk.LabelFrame(main_frame, text=" VISUALIZAÇÃO DO AMBIENTE", padding="10")
        visual_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(0, 10))
        visual_frame.columnconfigure(0, weight=1)
        visual_frame.rowconfigure(0, weight=1)
        
        # Canvas para matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, visual_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Inicializar visualização vazia
        self.initialize_visualization()
        
        # ===== PAINEL DE LOGS =====
        log_frame = ttk.LabelFrame(main_frame, text=" LOGS DE EXECUÇÃO", padding="10")
        log_frame.grid(row=1, column=2, sticky=(tk.N, tk.S, tk.E, tk.W))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # Área de logs com scroll
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            width=50,
            height=20,
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#ffffff"
        )
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar tags para colorir logs
        self.log_text.tag_config("SUCCESS", foreground="#27ae60")
        self.log_text.tag_config("ERROR", foreground="#e74c3c")
        self.log_text.tag_config("WARNING", foreground="#f39c12")
        self.log_text.tag_config("INFO", foreground="#3498db")
        self.log_text.tag_config("TREASURE", foreground="#f1c40f")
        
        # Barra de progresso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # ===== RODAPÉ =====
        footer_frame = ttk.Frame(main_frame)
        footer_frame.grid(row=3, column=0, columnspan=3, pady=(15, 0), sticky=(tk.W, tk.E))
        
        ttk.Label(
            footer_frame,
            text="Projecto IA 2025/2026 - Sistemas Multiagentes | ISPTEC",
            font=("Arial", 9, "italic"),
            foreground="#7f8c8d"
        ).pack(side=tk.LEFT)
        
        self.footer_label = ttk.Label(
            footer_frame,
            text="👥 Agentes: 0 | ⏱️ Tempo: 0s | 🏆 Tesouros: 0",
            font=("Arial", 9)
        )
        self.footer_label.pack(side=tk.RIGHT)
        
        # Configurar estilos personalizados
        self.configure_styles()
        
    def configure_styles(self):
        """Configurar estilos personalizados para os botões"""
        style = ttk.Style()
        
        # Botão Abordagem A
        style.configure("ApproachA.TButton",
                       foreground="white",
                       background="#3498db",
                       font=("Arial", 10, "bold"),
                       padding=10)
        style.map("ApproachA.TButton",
                 background=[("active", "#2980b9")])
        
        # Botão Abordagem B
        style.configure("ApproachB.TButton",
                       foreground="white",
                       background="#e74c3c",
                       font=("Arial", 10, "bold"),
                       padding=10)
        style.map("ApproachB.TButton",
                 background=[("active", "#c0392b")])
        
        # Botão Abordagem C
        style.configure("ApproachC.TButton",
                       foreground="white",
                       background="#2ecc71",
                       font=("Arial", 10, "bold"),
                       padding=10)
        style.map("ApproachC.TButton",
                 background=[("active", "#27ae60")])
        
        # Botão Iniciar
        style.configure("Start.TButton",
                       foreground="white",
                       background="#27ae60",
                       font=("Arial", 11, "bold"),
                       padding=8)
        style.map("Start.TButton",
                 background=[("active", "#219653")])
        
        # Botão Parar
        style.configure("Stop.TButton",
                       foreground="white",
                       background="#e74c3c",
                       font=("Arial", 11, "bold"),
                       padding=8)
        style.map("Stop.TButton",
                 background=[("active", "#c0392b")])
    
    def initialize_visualization(self):
        """Inicializar a visualização do ambiente"""
        self.ax.clear()
        self.ax.set_title("Ambiente 10x10 - Aguardando simulação...", fontsize=12)
        self.ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.3)
        self.ax.set_xlim(-0.5, 9.5)
        self.ax.set_ylim(-0.5, 9.5)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()  # Para ter (0,0) no canto superior esquerdo
        self.canvas.draw()
    
    def select_approach(self, approach):
        """Selecionar a abordagem"""
        self.approach_var.set(approach)
        self.log(f"Abordagem {approach} selecionada", "INFO")
        
        # Atualizar interface baseado na abordagem selecionada
        if approach == "A":
            self.group_type.set("heterogeneous")  # Padrão para abordagem A
        elif approach == "B":
            self.group_type.set("homogeneous")    # Padrão para abordagem B
        elif approach == "C":
            self.group_type.set("heterogeneous")  # Padrão para abordagem C
    
    
    def update_bomb_label(self, value):
        """Atualiza label com avisos visuais para Abordagem B"""
        bomb_pct = int(float(value))
        if bomb_pct <= 30:
            color, warning = "green", ""
        elif bomb_pct <= 50:
            color, warning = "orange", " ⚠️"
        elif bomb_pct <= 65:
            color, warning = "red", " ⚠️ Alto!"
        else:
            color, warning = "darkred", " ⚠️ LIMITADO!"
        
        self.bomb_label.config(text=f"{bomb_pct}%{warning}", foreground=color)
        
        if bomb_pct > 65 and hasattr(self, 'approach_var') and self.approach_var.get() == "B":
            self.log(f"⚠️  Bombas > 65% serão limitadas a 65% (garante 100% exploração)", "WARNING")

    def log(self, message, level="INFO"):
        """Adicionar mensagem ao log"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.log_history.append((timestamp, level, message))
        
        # Inserir no widget de texto
        self.log_text.insert(tk.END, formatted_message)
        
        # Aplicar cor baseado no nível
        start_index = self.log_text.index(tk.END + "-1c linestart")
        end_index = self.log_text.index(tk.END)
        
        if level == "SUCCESS":
            self.log_text.tag_add("SUCCESS", start_index, end_index)
        elif level == "ERROR":
            self.log_text.tag_add("ERROR", start_index, end_index)
        elif level == "WARNING":
            self.log_text.tag_add("WARNING", start_index, end_index)
        elif level == "TREASURE":
            self.log_text.tag_add("TREASURE", start_index, end_index)
        else:
            self.log_text.tag_add("INFO", start_index, end_index)
        
        # Scroll automático
        self.log_text.see(tk.END)
        
        # Atualizar também no console para debug
        print(f"{level}: {message}")
    
    def update_status(self, key, value):
        """Atualizar um campo de status"""
        if key in self.status_labels:
            self.status_labels[key].config(text=value)
    
    def update_visualization(self, simulation):
        """Atualizar a visualização do ambiente"""
        if not simulation:
            return
        
        self.ax.clear()
        
        # Criar matriz visual
        visual_grid = np.zeros((10, 10))
        
        # Se shared_memory for None (Baseline A), criar adapter
        if not hasattr(simulation, 'shared_memory') or simulation.shared_memory is None:
            # Adapter para Baseline A: agregar dados de memória individual de cada agente
            class BaselineVisualizationAdapter:
                def __init__(self, baseline_sim_obj):
                    self.env = baseline_sim_obj.env
                    self.agents = baseline_sim_obj.agents
                    self.num_agents = baseline_sim_obj.num_agents
                    # Agregar dados de memória individual de todos os agentes
                    self.shared_memory = type('obj', (object,), {
                        'explored': set(),
                        'bombs_found': set(),
                        'treasures_found': set(),
                        'treasures_collected': set(),
                        'true_flag_position': None
                    })()
                    # Combinar explored e bombs_found de todos os agentes
                    for agent in baseline_sim_obj.agents:
                        if hasattr(agent, 'memory'):
                            self.shared_memory.explored.update(agent.memory.explored)
                            self.shared_memory.bombs_found.update(agent.memory.bombs_found)
                            self.shared_memory.treasures_found.update(agent.memory.treasures_found)
            
            simulation = BaselineVisualizationAdapter(simulation)
        
        # Verificar se é abordagem B (sem tesouros)
        is_approach_b = hasattr(simulation, 'env') and isinstance(simulation.env, EnvironmentB)
        # Verificar se é abordagem C (com bandeira)
        is_approach_c = hasattr(simulation, 'env') and isinstance(simulation.env, EnvironmentC)

        if is_approach_b:
            # Marcar bombas encontradas
            for (x, y) in simulation.shared_memory.bombs_found:
                visual_grid[x, y] = 1  # Bomba
            # Marcar células livres exploradas
            for (x, y) in simulation.shared_memory.explored:
                if simulation.env.grid[x, y] == 'L':
                    visual_grid[x, y] = 2  # Livre explorada
            # Marcar posições dos agentes vivos
            for agent in simulation.agents:
                if agent.alive:
                    x, y = agent.position
                    visual_grid[x, y] = 4  # Agente
        else:
            # Abordagem A/C: marcar tesouros encontrados
            for (x, y) in simulation.shared_memory.treasures_found:
                visual_grid[x, y] = 3  # Tesouro
            # Marcar bombas encontradas
            for (x, y) in simulation.shared_memory.bombs_found:
                visual_grid[x, y] = 1  # Bomba
            # Marcar células livres exploradas
            for (x, y) in simulation.shared_memory.explored:
                if visual_grid[x, y] == 0:  # Não é tesouro ou bomba
                    visual_grid[x, y] = 2  # Livre explorada
            if is_approach_c:
                # A bandeira está SEMPRE visível na UI (roxo), mas desconhecida dos agentes
                flag_pos = None
                if hasattr(simulation.shared_memory, 'true_flag_position') and simulation.shared_memory.true_flag_position:
                    flag_pos = simulation.shared_memory.true_flag_position
                elif hasattr(simulation, 'env') and hasattr(simulation.env, 'flag_position') and simulation.env.flag_position:
                    flag_pos = simulation.env.flag_position
                
                if flag_pos:
                    fx, fy = flag_pos
                    visual_grid[fx, fy] = 5  # Bandeira (roxo)
            # Marcar posições dos agentes vivos
            for agent in simulation.agents:
                if agent.alive:
                    x, y = agent.position
                    visual_grid[x, y] = 4  # Agente
        
        # Configurar mapa de cores (adicionar roxo para bandeira)
        cmap = plt.cm.colors.ListedColormap(['white', 'red', 'lightblue', 'gold', 'green', 'purple'])
        bounds = [0, 1, 2, 3, 4, 5, 6]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        
        im = self.ax.imshow(visual_grid, cmap=cmap, norm=norm, interpolation='nearest')
        
        # Adicionar grade
        self.ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
        self.ax.grid(which='minor', color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Adicionar coordenadas para células não exploradas
        for i in range(10):
            for j in range(10):
                if visual_grid[i, j] == 0:
                    self.ax.text(j, i, '?', ha='center', va='center', 
                                fontsize=8, color='gray', alpha=0.5)
        
        # Adicionar legenda
        from matplotlib.patches import Patch
        if is_approach_b:
            legend_elements = [
                Patch(facecolor='white', edgecolor='black', label='Desconhecido'),
                Patch(facecolor='red', edgecolor='black', label='Bomba'),
                Patch(facecolor='lightblue', edgecolor='black', label='Livre Explorada'),
                Patch(facecolor='green', edgecolor='black', label='Agente')
            ]
        elif is_approach_c:
            legend_elements = [
                Patch(facecolor='white', edgecolor='black', label='Desconhecido'),
                Patch(facecolor='red', edgecolor='black', label='Bomba'),
                Patch(facecolor='lightblue', edgecolor='black', label='Livre Explorada'),
                Patch(facecolor='gold', edgecolor='black', label='Tesouro'),
                Patch(facecolor='green', edgecolor='black', label='Agente'),
                Patch(facecolor='purple', edgecolor='black', label='🚩 Bandeira')
            ]
        else:
            # Abordagem A
            legend_elements = [
                Patch(facecolor='white', edgecolor='black', label='Desconhecido'),
                Patch(facecolor='red', edgecolor='black', label='Bomba'),
                Patch(facecolor='lightblue', edgecolor='black', label='Livre Explorada'),
                Patch(facecolor='gold', edgecolor='black', label='Tesouro'),
                Patch(facecolor='green', edgecolor='black', label='Agente')
            ]
        
        # Título e configurações do gráfico
        if is_approach_b:
            approach = "B"
            title_suffix = "Exploração Completa"
        elif is_approach_c:
            approach = "C"
            title_suffix = "Busca da Bandeira "
        else:
            approach = "A"
            title_suffix = "Caça ao Tesouro"
        
        self.ax.set_title(f'Abordagem {approach} - {title_suffix} ({simulation.num_agents} agentes)', 
                         fontsize=12, fontweight='bold')
        self.ax.set_xlim(-0.5, 9.5)
        self.ax.set_ylim(9.5, -0.5)
        self.ax.set_xticks(range(10))
        self.ax.set_yticks(range(10))
        
        self.ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def start_simulation(self):
        """Iniciar a simulação em uma thread separada"""
        if self.simulation_thread and self.simulation_thread.is_alive():
            messagebox.showwarning("Aviso", "Uma simulação já está em execução!")
            return
        
        # Obter parâmetros da interface
        num_agents = int(self.agent_spinbox.get())
        bomb_ratio = float(self.bomb_scale.get()) / 100.0
        treasure_count = int(self.treasure_spinbox.get())
        max_steps = int(self.max_steps_spinbox.get())
        group_type = self.group_type.get()
        approach = self.approach_var.get()
        
        # Log dos parâmetros
        self.log(f"Iniciando simulação Abordagem {approach}", "INFO")
        self.log(f"Parâmetros: {num_agents} agentes, {bomb_ratio*100}% bombas, {treasure_count} tesouros", "INFO")
        self.log(f"Tipo de grupo: {group_type}, Máximo de passos: {max_steps}", "INFO")
        
        # Resetar interface
        self.start_time = time.time()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.stop_simulation = False
        self.progress_var.set(0)
        
        # Iniciar thread de simulação
        self.simulation_thread = threading.Thread(
            target=self.run_simulation_thread,
            args=(approach, num_agents, bomb_ratio, treasure_count, max_steps, group_type)
        )
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
    
    def run_simulation_thread(self, approach, num_agents, bomb_ratio, treasure_count, max_steps, group_type):
        """Função executada na thread de simulação"""
        try:
            # IMPORTANTE: Converter Baseline A para simulação iterativa também
            if group_type == "baseline" and approach == "A":
                # Usar BaselineSimulation wrapper para passo a passo
                self.current_simulation = BaselineSimulation(
                    num_agents=num_agents,
                    bomb_ratio=bomb_ratio,
                    treasure_count=treasure_count,
                    max_steps=max_steps
                )
            # Criar simulação baseada na abordagem
            elif approach == "A":
                homogeneous = (group_type == "homogeneous")
                self.current_simulation = ApproachASimulation(
                    num_agents=num_agents,
                    bomb_ratio=bomb_ratio,
                    treasure_count=treasure_count,
                    homogeneous=homogeneous,
                    max_steps=max_steps
                )
            elif approach == "B":
                if group_type == "baseline":
                    # Baseline B com múltiplos agentes BFS colaborativos
                    self.current_simulation = BaselineB_BFS(
                        num_agents=num_agents,
                        bomb_ratio=bomb_ratio,
                        max_steps=max_steps
                    )
                else:
                    # Homogêneo ou Heterogêneo
                    homogeneous = (group_type == "homogeneous")
                    self.current_simulation = ApproachBSimulation(
                        num_agents=num_agents,
                        bomb_ratio=bomb_ratio,
                        homogeneous=homogeneous,
                        max_steps=max_steps
                    )
            elif approach == "C":
                # Abordagem C: Encontrar a bandeira
                if group_type == "baseline":
                    # Baseline C: A*
                    self.current_simulation = BaselineC_AStar(
                        num_agents=num_agents,
                        bomb_ratio=bomb_ratio,
                        treasure_count=treasure_count,
                        max_steps=max_steps
                    )
                else:
                    # Homogêneo ou Heterogêneo
                    homogeneous = (group_type == "homogeneous")
                    self.current_simulation = ApproachCSimulation(
                        num_agents=num_agents,
                        bomb_ratio=bomb_ratio,
                        treasure_count=treasure_count,
                        homogeneous=homogeneous,
                        max_steps=max_steps
                    )
            else:
                self.current_simulation = BaselineSimulation(
                    num_agents=num_agents,
                    bomb_ratio=bomb_ratio,
                    treasure_count=treasure_count,
                    max_steps=max_steps
                )
            
            # Executar simulação passo a passo
            start_time = time.time()
            step = 0
            
            # Configurar ambiente inicial
            self.root.after(0, self.update_visualization, self.current_simulation)
            self.root.after(0, self.update_status, "Status:", "Executando...")
            
            # Simulação principal
            while (step < max_steps and 
                   len([a for a in self.current_simulation.agents if a.alive]) > 0 and
                   not self.stop_simulation):
                
                step += 1
                
                # Executar um passo
                if group_type == "baseline" and approach == "A":
                    # Para baseline A, executar passo manualmente
                    self.execute_baseline_step(step)
                else:
                    # Para ML agents (homogêneo/heterogêneo)
                    self.execute_simulation_step(step)
                
                # VERIFICAÇÃO CRÍTICA: Se Abordagem C e bandeira foi encontrada, terminar
                if (approach == "C" and 
                    hasattr(self.current_simulation, 'shared_memory') and
                    self.current_simulation.shared_memory.flag_found):
                    self.root.after(0, self.log, "🚩 BANDEIRA ENCONTRADA! Encerrando simulação...", "SUCCESS")
                    break
                
                # Atualizar progresso
                progress = (step / max_steps) * 100
                self.root.after(0, self.progress_var.set, progress)
                
                # Atualizar status a cada 10 passos ou no final
                if step % 10 == 0 or step >= max_steps:
                    self.update_simulation_status(step, max_steps)
                
                # Pausa para visualização
                time.sleep(0.05)  # 50ms entre passos para visualização
            
            # Finalizar simulação
            self.finalize_simulation(start_time)
            
        except Exception as e:
            self.root.after(0, self.log, f"Erro na simulação: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            self.root.after(0, self.reset_buttons)
            self.root.after(0, self.update_status, "Status:", "Executando...")
            
            # Simulação principal
            while (step < max_steps and 
                   len([a for a in self.current_simulation.agents if a.alive]) > 0 and
                   not self.stop_simulation):
                
                step += 1
                
                # Executar um passo
                self.execute_simulation_step(step)
                
                # VERIFICAÇÃO CRÍTICA: Se Abordagem C e bandeira foi encontrada, terminar
                if (approach == "C" and 
                    hasattr(self.current_simulation, 'shared_memory') and
                    self.current_simulation.shared_memory.flag_found):
                    self.root.after(0, self.log, "🚩 BANDEIRA ENCONTRADA! Encerrando simulação...", "SUCCESS")
                    break
                
                # Atualizar progresso
                progress = (step / max_steps) * 100
                self.root.after(0, self.progress_var.set, progress)
                
                # Atualizar status a cada 10 passos ou no final
                if step % 10 == 0 or step >= max_steps:
                    self.update_simulation_status(step, max_steps)
                
                # Pausa para visualização
                time.sleep(0.05)  # 50ms entre passos para visualização
            
            # Finalizar simulação
            self.finalize_simulation(start_time)
            
        except Exception as e:
            self.root.after(0, self.log, f"Erro na simulação: {str(e)}", "ERROR")
            self.root.after(0, self.reset_buttons)
    
    def execute_simulation_step(self, step):
        """Executar um passo da simulação"""
        self.current_simulation.logs.append(f"\n--- Passo {step} ---")
        
        # ✅ FIX: FASE 1 - Todos os agentes escolhem suas ações ANTES de qualquer um se mover
        actions = {}
        for agent in self.current_simulation.agents:
            if not agent.alive:
                actions[agent.id] = None
                continue
            
            # Treinar modelos periodicamente
            if step % 10 == 0:
                agent.train_models(self.current_simulation.shared_memory, self.current_simulation.env)
            
            # Escolher ação (SEM executar ainda)
            next_pos = agent.choose_action(self.current_simulation.shared_memory, self.current_simulation.env)
            actions[agent.id] = next_pos
        
        # ✅ FIX: FASE 2 - Agora todos executam seus movimentos
        for agent in self.current_simulation.agents:
            if not agent.alive:
                continue
            
            # Executar ação escolhida na Fase 1
            next_pos = actions.get(agent.id)
            log_msg = agent.move_to(next_pos, self.current_simulation.shared_memory, self.current_simulation.env)
            
            # Adicionar log na interface
            if "BANDEIRA" in log_msg:
                self.root.after(0, self.log, log_msg, "SUCCESS")
            elif "TESOURO" in log_msg:
                self.root.after(0, self.log, log_msg, "TREASURE")
            elif "DESTRUÍDO" in log_msg:
                self.root.after(0, self.log, log_msg, "ERROR")
            else:
                self.root.after(0, self.log, log_msg)
        
        # IMPORTANTE: Calcular explored_percentage para Abordagem B
        if hasattr(self.current_simulation, 'env') and isinstance(self.current_simulation.env, EnvironmentB):
            # Calcular células livres exploradas
            explored_free_cells = 0
            for i in range(self.current_simulation.env.size):
                for j in range(self.current_simulation.env.size):
                    pos = (i, j)
                    if (pos in self.current_simulation.shared_memory.explored and 
                        self.current_simulation.env.grid[i, j] != 'B'):
                        explored_free_cells += 1
            
            # Atualizar métrica
            if hasattr(self.current_simulation, 'free_cells'):
                explored_pct = (explored_free_cells / self.current_simulation.free_cells * 100) if self.current_simulation.free_cells > 0 else 0
            else:
                # Calcular free_cells se não existir
                bomb_count = sum(1 for row in self.current_simulation.env.grid for cell in row if cell == 'B')
                free_cells = self.current_simulation.env.size * self.current_simulation.env.size - bomb_count
                explored_pct = (explored_free_cells / free_cells * 100) if free_cells > 0 else 0
            
            self.current_simulation.metrics['explored_percentage'] = explored_pct
            self.current_simulation.metrics['explored_free_cells'] = explored_free_cells
        
        # Atualizar visualização
        self.root.after(0, self.update_visualization, self.current_simulation)
    
    def execute_baseline_step(self, step):
        """Executar um passo da simulação Baseline A (sem memória compartilhada)"""
        self.current_simulation.logs.append(f"\n--- Passo {step} (Baseline) ---")
        
        # FASE 1: Todos os agentes escolhem suas ações
        actions = {}
        for agent in self.current_simulation.agents:
            if not agent.alive:
                actions[agent.id] = None
                continue
            
            # Para baseline, usar memória individual do agente
            next_pos = agent.choose_action(None, self.current_simulation.env)  # None = usa memória privada
            actions[agent.id] = next_pos
        
        # FASE 2: Todos executam seus movimentos
        for agent in self.current_simulation.agents:
            if not agent.alive:
                continue
            
            next_pos = actions.get(agent.id)
            log_msg = agent.move_to(next_pos, None, self.current_simulation.env)  # None = usa memória privada
            
            # Adicionar log na interface
            if "TESOURO" in log_msg:
                self.root.after(0, self.log, log_msg, "TREASURE")
            elif "DESTRUÍDO" in log_msg:
                self.root.after(0, self.log, log_msg, "ERROR")
            else:
                self.root.after(0, self.log, log_msg, "INFO")
        
        # Atualizar visualização
        self.root.after(0, self.update_visualization, self.current_simulation)
    
    def update_simulation_status(self, step, max_steps):
        """Atualizar informações de status"""
        approach = self.approach_var.get()
        agents_alive = len([a for a in self.current_simulation.agents if a.alive])
        self.root.after(0, self.update_status, "Agentes Vivos:", f"{agents_alive}/{self.current_simulation.num_agents}")
        self.root.after(0, self.update_status, "Passos Executados:", f"{step}/{max_steps}")
        if approach == "B":
            explored_pct = self.current_simulation.metrics.get('explored_percentage', 0)
            self.root.after(0, self.update_status, "Explorado (%):", f"{explored_pct:.1f}%")
            if self.start_time:
                elapsed_time = time.time() - self.start_time
                self.root.after(0, lambda: self.footer_label.config(text=f"👥 Agentes: {agents_alive} | ⏱️ Tempo: {elapsed_time:.1f}s | 🌍 Explorado: {explored_pct:.1f}%"))
            if step % 50 == 0:
                self.root.after(0, self.log, f"Passo {step}: {explored_pct:.1f}% explorado, {agents_alive} agentes vivos", "INFO")
        elif approach == "C":
            # Abordagem C: mostrar status da bandeira
            flag_found = self.current_simulation.shared_memory.flag_found if hasattr(self.current_simulation, 'shared_memory') else False
            treasures_found = len(self.current_simulation.shared_memory.treasures_collected) if hasattr(self.current_simulation, 'shared_memory') else 0
            
            flag_status = "✅ Encontrada" if flag_found else "❌ Procurando..."
            self.root.after(0, self.update_status, "Bandeira:", flag_status)
            if treasures_found > 0:
                self.root.after(0, self.update_status, "Tesouros:", f"{treasures_found}")
            
            if self.start_time:
                elapsed_time = time.time() - self.start_time
                self.root.after(0, lambda: self.footer_label.config(text=f"👥 Agentes: {agents_alive} | ⏱️ Tempo: {elapsed_time:.1f}s | 🚩 Bandeira: {'✅' if flag_found else '❌'}"))
            
            if step % 50 == 0:
                self.root.after(0, self.log, f"Passo {step}: Bandeira {'ENCONTRADA' if flag_found else 'não encontrada'}, {agents_alive} agentes vivos", "INFO")
        else:
            # Abordagem A
            # Para Baseline, agregar tesouros de todos os agentes individuais
            if self.group_type.get() == "baseline":
                treasures_found = sum(agent.treasures_collected for agent in self.current_simulation.agents)
            else:
                treasures_found = len(self.current_simulation.shared_memory.treasures_found)
            
            total_treasures = self.current_simulation.env.treasure_count
            self.root.after(0, self.update_status, "Tesouros Encontrados:", f"{treasures_found}/{total_treasures}")
            if self.start_time:
                elapsed_time = time.time() - self.start_time
                self.root.after(0, lambda: self.footer_label.config(text=f"👥 Agentes: {agents_alive} | ⏱️ Tempo: {elapsed_time:.1f}s | 🏆 Tesouros: {treasures_found}"))
            if step % 50 == 0:
                self.root.after(0, self.log, f"Passo {step}: {treasures_found} tesouros, {agents_alive} agentes vivos", "INFO")
    
    def finalize_simulation(self, start_time):
        """Finalizar a simulação e calcular métricas"""
        end_time = time.time()
        execution_time = end_time - start_time
        
        approach = self.approach_var.get()
        agents_alive = len([a for a in self.current_simulation.agents if a.alive])
        explored_pct = self.current_simulation.metrics.get('explored_percentage', 0)
        # Inicializar para evitar erro de variável não associada
        treasures_found = 0
        total_treasures = 0
        # Calcular sucesso baseado na abordagem
        success = False
        if approach == "A":
            # Para Baseline A, agregar tesouros de todos os agentes; para ML, usar shared_memory
            if self.group_type.get() == "baseline":
                treasures_found = sum(agent.treasures_collected for agent in self.current_simulation.agents)
            else:
                treasures_found = len(self.current_simulation.shared_memory.treasures_found)
            
            total_treasures = self.current_simulation.env.treasure_count
            success = treasures_found > (total_treasures * 0.5)
            self.root.after(0, self.update_status, "Tesouros Encontrados:", f"{treasures_found}/{total_treasures}")
            self.root.after(0, lambda: self.footer_label.config(text=f"👥 Agentes: {agents_alive} | ⏱️ Tempo: {execution_time:.1f}s | 🏆 Tesouros: {treasures_found}"))
            if success:
                self.root.after(0, self.log, f"✅ SIMULAÇÃO CONCLUÍDA COM SUCESSO! Tesouros: {treasures_found}/{total_treasures}", "SUCCESS")
            else:
                self.root.after(0, self.log, f"⚠️ SIMULAÇÃO CONCLUÍDA SEM SUCESSO. Tesouros: {treasures_found}/{total_treasures}", "WARNING")
        elif approach == "B":
            success = explored_pct >= 100.0 and agents_alive > 0
            self.root.after(0, self.update_status, "Explorado (%):", f"{explored_pct:.1f}%")
            self.root.after(0, lambda: self.footer_label.config(text=f"👥 Agentes: {agents_alive} | ⏱️ Tempo: {execution_time:.1f}s | 🌍 Explorado: {explored_pct:.1f}%"))
            if success:
                self.root.after(0, self.log, f"✅ SIMULAÇÃO CONCLUÍDA COM SUCESSO! Ambiente explorado: {explored_pct:.1f}%", "SUCCESS")
            else:
                self.root.after(0, self.log, f"⚠️ SIMULAÇÃO CONCLUÍDA SEM SUCESSO. Ambiente explorado: {explored_pct:.1f}%", "WARNING")
        elif approach == "C":
            # Abordagem C: Sucesso = Bandeira encontrada
            flag_found = False
            if hasattr(self.current_simulation, 'shared_memory'):
                flag_found = self.current_simulation.shared_memory.flag_found
            elif hasattr(self.current_simulation, 'metrics'):
                flag_found = self.current_simulation.metrics.get('flag_found', False)
            
            treasures_found = 0
            if hasattr(self.current_simulation, 'shared_memory'):
                treasures_found = len(self.current_simulation.shared_memory.treasures_collected)
            
            # Custo do caminho (se disponível)
            path_cost = 0
            if hasattr(self.current_simulation, 'metrics'):
                path_cost = self.current_simulation.metrics.get('min_path_cost', 0)
            
            success = flag_found
            self.current_simulation.metrics['flag_found'] = flag_found
            self.current_simulation.metrics['treasures_found'] = treasures_found
            # Atualizar status
            self.root.after(0, self.update_status, "Bandeira:", "✅ Encontrada" if flag_found else "❌ Não encontrada")
            if treasures_found > 0:
                self.root.after(0, self.update_status, "Tesouros Coletados:", f"{treasures_found}")
            if path_cost > 0 and path_cost != float('inf'):
                self.root.after(0, self.update_status, "Custo do Caminho:", f"{path_cost:.2f}")
            
            # Footer com informações da Abordagem C
            self.root.after(0, lambda: self.footer_label.config(
                text=f"👥 Agentes: {agents_alive} | ⏱️ Tempo: {execution_time:.1f}s | 🚩 Bandeira: {'✅' if flag_found else '❌'}"
            ))
            
            # Log de conclusão
            if success:
                msg = f"✅ SIMULAÇÃO CONCLUÍDA COM SUCESSO! Bandeira encontrada!"
                if path_cost > 0 and path_cost != float('inf'):
                    msg += f" | Custo do caminho: {path_cost:.2f}"
                if treasures_found > 0:
                    msg += f" | Tesouros coletados: {treasures_found}"
                self.root.after(0, self.log, msg, "SUCCESS")
            else:
                self.root.after(0, self.log, f"⚠️ SIMULAÇÃO CONCLUÍDA SEM SUCESSO. Bandeira não encontrada.", "WARNING")
        
        self.current_simulation.metrics['execution_time'] = execution_time
        # Armazenar resultados
        result = {
            'approach': approach,
            'agents': self.current_simulation.num_agents,
            'treasures_found': treasures_found,
            'total_treasures': total_treasures,
            'agents_alive': agents_alive,
            'execution_time': execution_time,
            'success': success,
            'bomb_ratio': float(self.bomb_scale.get()),
            'group_type': self.group_type.get()
        }
        self.simulation_results.append(result)
        
        # Resetar botões
        self.root.after(0, self.reset_buttons)
    
    def reset_buttons(self):
        """Resetar estado dos botões"""
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_var.set(100)
    
    def run_baseline_simulation(self, approach, bomb_ratio, treasure_count, max_steps):
        """Executar simulação baseline com agentes reais no ambiente"""
        try:
            # Obter número de agentes da interface
            num_agents = int(self.agent_spinbox.get())
            
            self.log(f"Executando simulação baseline com {num_agents} agentes...", "INFO")
            self.log("⚠️ Baseline: Agentes trabalham SEM colaboração (memória individual)", "INFO")
            
            # Criar simulação baseline com agentes reais
            if approach == "A":
                baseline_sim = BaselineSimulation(
                    num_agents=num_agents,
                    bomb_ratio=bomb_ratio,
                    treasure_count=treasure_count,
                    max_steps=max_steps
                )
            elif approach == "B":
                # Para abordagem B, usar BaselineSimulation (sem tesouros)
                self.log("⚠️ Baseline B: Sem tesouros, foco em exploração", "INFO")
                baseline_sim = BaselineSimulation(
                    num_agents=num_agents,
                    bomb_ratio=bomb_ratio,
                    treasure_count=0,  # B não tem tesouros
                    max_steps=max_steps
                )
            elif approach == "C":
                # Para abordagem C, usar BaselineC_AStar
                self.log("⚠️ Baseline C: Usando A* para encontrar bandeira", "INFO")
                baseline_sim = BaselineC_AStar(
                    num_agents=num_agents,
                    bomb_ratio=bomb_ratio,
                    treasure_count=treasure_count,
                    max_steps=max_steps
                )
            else:
                raise ValueError("Abordagem não suportada")
            
            # Executar baseline com visualização
            start_time = time.time()
            
            # Armazenar referência para visualização
            self.current_simulation = baseline_sim
            
            # Executar simulação
            metrics = baseline_sim.run_simulation(verbose=True)
            execution_time = time.time() - start_time
            
            # Atualizar métricas para incluir execution_time
            metrics['execution_time'] = execution_time
            
            # Log dos resultados
            self.log(f"Baseline {approach} concluído!", "SUCCESS")
            self.log(f"Tesouros encontrados: {metrics.get('treasures_found', 0)}/{treasure_count}", "INFO")
            self.log(f"Agentes sobreviventes: {metrics.get('agents_alive', 0)}/{num_agents}", "INFO")
            self.log(f"Passos executados: {metrics['steps_taken']}", "INFO")
            self.log(f"Tempo de execução: {execution_time:.2f}s", "INFO")
            self.log(f"Sucesso: {'Sim' if metrics['success'] else 'Não'}", "INFO")
            
            # Atualizar interface
            self.root.after(0, self.update_status, "Status:", "Baseline concluído")
            self.root.after(0, self.update_status, "Tempo de Execução:", f"{execution_time:.2f}s")
            self.root.after(0, self.update_status, "Agentes Vivos:", f"{metrics.get('agents_alive', 0)}/{num_agents}")
            self.root.after(0, self.update_status, "Tesouros:", f"{metrics.get('treasures_found', 0)}/{treasure_count}")
            
            # Exibir logs da simulação
            if hasattr(baseline_sim, 'logs'):
                for log in baseline_sim.logs:
                    self.log(log, "INFO")
            
            # Salvar resultados no storage se disponível
            if hasattr(self, 'storage'):
                parameters = {
                    'num_agents': num_agents,
                    'bomb_ratio': bomb_ratio,
                    'treasure_count': treasure_count,
                    'max_steps': max_steps,
                    'homogeneous': False  # Baseline é sempre "não-colaborativo"
                }
                self.storage.save_result(approach, 'baseline', metrics, parameters)
                self.log("✅ Resultados salvos no armazenamento", "SUCCESS")
            
            # Visualizar ambiente final
            if hasattr(baseline_sim, 'env'):
                # Criar um wrapper para Baseline ser compatível com update_visualization
                class BaselineVisualizationAdapter:
                    def __init__(self, baseline_sim_obj):
                        self.env = baseline_sim_obj.env
                        self.agents = baseline_sim_obj.agents
                        self.num_agents = baseline_sim_obj.num_agents
                        # Agregar dados de memória individual de todos os agentes
                        self.shared_memory = type('obj', (object,), {
                            'explored': set(),
                            'bombs_found': set(),
                            'treasures_found': set(),
                            'treasures_collected': set(),
                            'true_flag_position': None
                        })()
                        # Combinar explored e bombs_found de todos os agentes
                        for agent in baseline_sim_obj.agents:
                            if hasattr(agent, 'memory'):
                                self.shared_memory.explored.update(agent.memory.explored)
                                self.shared_memory.bombs_found.update(agent.memory.bombs_found)
                                self.shared_memory.treasures_found.update(agent.memory.treasures_found)
                
                adapter = BaselineVisualizationAdapter(baseline_sim)
                self.root.after(0, self.update_visualization, adapter)
            
            # Resetar botões
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
            
        except Exception as e:
            self.log(f"Erro na simulação baseline: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
    
    def stop_simulation_func(self):
        """Parar a simulação em execução"""
        self.stop_simulation = True
        self.log("Simulação interrompida pelo usuário", "WARNING")
        self.update_status("Status:", "Interrompido")
        self.reset_buttons()

    def run_comparison(self):
        """Executar comparação entre abordagens - VERSÃO CORRIGIDA"""
        try:
            # Usar o sistema de comparação avançada se disponível
            if hasattr(self, 'comparison_ext'):
                self.comparison_ext.open_comparison_window()
            else:
                # Fallback: criar janela simples com dados do JSON
                self.open_simple_comparison()
                
        except Exception as e:
            self.log(f"Erro na comparação: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
            
    def open_simple_comparison(self):
        """Abre comparação simples usando dados salvos"""
        try:
            import json
            from pathlib import Path
            import tkinter as tk
            from tkinter import ttk, messagebox
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import numpy as np
            
            # Carregar dados do JSON
            json_file = Path("simulation_results/all_results.json")
            
            if not json_file.exists():
                messagebox.showwarning("Aviso", "Nenhum dado salvo!\n\nExecute simulações primeiro.")
                return
            
            with open(json_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            
            # Criar janela
            window = tk.Toplevel(self.root)
            window.title("Comparação de Abordagens")
            window.geometry("1200x800")
            
            # Notebook
            notebook = ttk.Notebook(window)
            notebook.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Aba de Gráficos
            graph_frame = ttk.Frame(notebook)
            notebook.add(graph_frame, text="📊 Gráficos")
            
            # Criar figura
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            fig.suptitle('Comparação por Abordagem', fontsize=14, fontweight='bold')
            
            colors_map = {
                'homogeneous': '#e67e22',
                'heterogeneous': '#9b59b6',
                'baseline': '#e74c3c'
            }
            
            for idx, (approach, ax) in enumerate(zip(['A', 'B', 'C'], axes)):
                results = all_results.get(approach, [])
                
                if not results:
                    ax.text(0.5, 0.5, f'Abordagem {approach}\n\nSem dados\n\nExecute simulações',
                        ha='center', va='center', fontsize=12)
                    ax.set_title(f'Abordagem {approach}', fontweight='bold')
                    ax.axis('off')
                    continue
                
                # Agrupar por tipo
                groups = {}
                for r in results:
                    gt = r['group_type']
                    if gt not in groups:
                        groups[gt] = []
                    groups[gt].append(r)
                
                # Calcular médias
                group_names = []
                group_values = []
                group_colors = []
                
                # Métrica principal por abordagem
                if approach == 'A':
                    metric_key = 'treasure_percentage'
                    title = 'Tesouros (%)'
                elif approach == 'B':
                    metric_key = 'explored_percentage'
                    title = 'Explorado (%)'
                else:  # C
                    metric_key = 'success_rate'
                    title = 'Taxa Sucesso'
                
                for gt_name in ['homogeneous', 'heterogeneous', 'baseline']:
                    if gt_name in groups:
                        values = [r['metrics'].get(metric_key, 0) for r in groups[gt_name]]
                        if values:
                            group_names.append(gt_name.title()[:5])
                            group_values.append(np.mean(values))
                            group_colors.append(colors_map[gt_name])
                
                if group_names:
                    bars = ax.bar(group_names, group_values, color=group_colors, alpha=0.8, edgecolor='black')
                    
                    # Valores nas barras
                    for bar, val in zip(bars, group_values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.1f}',
                            ha='center', va='bottom', fontweight='bold')
                    
                    ax.set_title(f'Abordagem {approach}', fontweight='bold')
                    ax.set_ylabel(title)
                    ax.grid(axis='y', alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center')
                    ax.set_title(f'Abordagem {approach}', fontweight='bold')
            
            plt.tight_layout()
            
            # Embed
            canvas = FigureCanvasTkAgg(fig, graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # Aba de Dados
            data_frame = ttk.Frame(notebook)
            notebook.add(data_frame, text="📋 Dados")
            
            # Texto com resumo
            text = tk.Text(data_frame, wrap='word', font=('Consolas', 10))
            text.pack(fill='both', expand=True, padx=10, pady=10)
            
            summary = "="*60 + "\n"
            summary += "RESUMO DOS DADOS SALVOS\n"
            summary += "="*60 + "\n\n"
            
            for approach in ['A', 'B', 'C']:
                results = all_results.get(approach, [])
                summary += f"\nABORDAGEM {approach}: {len(results)} simulações\n"
                summary += "-"*40 + "\n"
                
                # Agrupar
                groups = {}
                for r in results:
                    gt = r['group_type']
                    if gt not in groups:
                        groups[gt] = 0
                    groups[gt] += 1
                
                for gt, count in groups.items():
                    summary += f"  • {gt.title()}: {count} simulações\n"
            
            summary += "\n" + "="*60 + "\n"
            summary += "Use 'Comparação Avançada' para análise completa\n"
            
            text.insert('1.0', summary)
            text.config(state='disabled')
            
            # Botão fechar
            btn_frame = ttk.Frame(window)
            btn_frame.pack(fill='x', padx=10, pady=10)
            ttk.Button(btn_frame, text="❌ Fechar", command=window.destroy).pack(side='right')
            
            self.log("Comparação simples aberta", "INFO")
            
        except Exception as e:
            self.log(f"Erro ao abrir comparação simples: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
    
    def execute_comparison(self):
        """Executar comparação completa"""
        try:
            # Preparar janela de comparação
            comparison_window = tk.Toplevel(self.root)
            comparison_window.title("Comparação de Abordagens")
            comparison_window.geometry("1200x800")
            
            # Criar notebook para abas
            notebook = ttk.Notebook(comparison_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Aba 1: Gráficos comparativos
            graph_frame = ttk.Frame(notebook)
            notebook.add(graph_frame, text=" Gráficos")
            
            # Executar simulações comparativas baseado na abordagem selecionada
            approach = self.approach_var.get()
            if approach == "A":
                results = compare_approaches()
            elif approach == "B":
                results = compare_approaches_b()
            elif approach == "C":
                results = compare_approaches_c()
            else:
                self.log(f"Comparação não implementada para abordagem {approach}", "WARNING")
                return
            
            # Criar gráficos
            self.create_comparison_charts(graph_frame, results)
            
            # Aba 2: Tabela de resultados
            table_frame = ttk.Frame(notebook)
            notebook.add(table_frame, text="📋 Tabela")
            
            self.create_results_table(table_frame, results)
            
            # Aba 3: Análise estatística
            stats_frame = ttk.Frame(notebook)
            notebook.add(stats_frame, text=" Estatísticas")
            
            self.create_statistics_panel(stats_frame, results)
            
            self.log("Comparação concluída com sucesso!", "SUCCESS")
            
        except Exception as e:
            self.log(f"Erro na execução da comparação: {str(e)}", "ERROR")
    
    def create_comparison_charts(self, parent, results):
        """Criar gráficos comparativos com cores apropriadas"""
        # Extrair dados
        names = [r['name'] for r in results]
        group_types = [r.get('group_type', 'unknown') for r in results]
        
        # Usar cores baseadas no tipo de resultado
        colors = []
        for gt in group_types:
            if gt == 'homogeneous':
                colors.append('#e67e22')  # Laranja para homogêneo
            elif gt == 'heterogeneous':
                colors.append('#9b59b6')  # Roxo para heterogêneo
            elif gt == 'baseline':
                colors.append('#e74c3c')  # Vermelho para baseline
            else:
                colors.append('#95a5a6')  # Cinza para outros
        
        success_rates = [r['metrics']['success_rate'] if 'success_rate' in r['metrics'] else (1 if r['metrics']['success'] else 0) for r in results]
        treasures = [r['metrics'].get('avg_treasures_found', r['metrics'].get('treasures_found', 0)) for r in results]
        times = [r['metrics'].get('avg_execution_time', r['metrics'].get('execution_time', 0)) for r in results]
        survivors = [r['metrics'].get('avg_agents_alive', r['metrics'].get('agents_alive', 1)) for r in results]
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Gráfico 1: Taxa de Sucesso
        bars1 = axes[0, 0].bar(names, success_rates, color=colors)
        axes[0, 0].set_title('Taxa de Sucesso', fontweight='bold')
        axes[0, 0].set_ylim(0, 1.1)
        axes[0, 0].set_ylabel('Taxa de Sucesso')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, v in zip(bars1, success_rates):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{v:.1%}', ha='center', va='bottom', fontsize=9)
        
        # Gráfico 2: Tesouros Encontrados
        bars2 = axes[0, 1].bar(names, treasures, color=colors)
        axes[0, 1].set_title('Tesouros Encontrados (Média)', fontweight='bold')
        axes[0, 1].set_ylabel('Quantidade')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, v in zip(bars2, treasures):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{v:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Gráfico 3: Tempo de Execução
        bars3 = axes[1, 0].bar(names, times, color=colors)
        axes[1, 0].set_title('Tempo de Execução (Média)', fontweight='bold')
        axes[1, 0].set_ylabel('Segundos')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Gráfico 4: Agentes Sobreviventes
        bars4 = axes[1, 1].bar(names, survivors, color=colors)
        axes[1, 1].set_title('Agentes Sobreviventes (Média)', fontweight='bold')
        axes[1, 1].set_ylabel('Quantidade')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e67e22', label='Homogêneo'),
            Patch(facecolor='#9b59b6', label='Heterogêneo'),
            Patch(facecolor='#e74c3c', label='Baseline Clássico')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
        
        plt.suptitle('COMPARAÇÃO: Grupos ML vs Algoritmos Clássicos por Abordagem', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        # Embed no tkinter
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_results_table(self, parent, results):
        """Criar tabela organizada por abordagem"""
        # Organizar dados por abordagem
        table_data = []
        
        approaches_data = {}
        for r in results:
            approach = r['approach']
            if approach not in approaches_data:
                approaches_data[approach] = []
            approaches_data[approach].append(r)
        
        for approach, approach_results in approaches_data.items():
            # Adicionar cabeçalho da abordagem
            table_data.append({
                'Abordagem': f'=== ABORDAGEM {approach} ===',
                'Tipo': '',
                'Sucesso': '',
                'Tesouros': '',
                'Tempo': '',
                'Agentes': ''
            })
            
            # Grupos ML
            ml_groups = [r for r in approach_results if r.get('group_type') in ['homogeneous', 'heterogeneous']]
            for group in ml_groups:
                success_rate = group['metrics'].get('success_rate', 0)
                avg_treasures = group['metrics'].get('avg_treasures_found', 0)
                avg_time = group['metrics'].get('avg_execution_time', 0)
                avg_agents = group['metrics'].get('avg_agents_alive', 0)
                
                table_data.append({
                    'Abordagem': group['name'],
                    'Tipo': 'Grupo ML',
                    'Sucesso': f'{success_rate:.1%}',
                    'Tesouros': f'{avg_treasures:.1f}',
                    'Tempo': f'{avg_time:.2f}s',
                    'Agentes': f'{avg_agents:.1f}'
                })
            
            # Baselines
            baselines = [r for r in approach_results if r.get('group_type') == 'baseline']
            for baseline in baselines:
                success_rate = baseline['metrics'].get('success_rate', 0)
                avg_treasures = baseline['metrics'].get('avg_treasures_found', baseline['metrics'].get('treasures_found', 0))
                avg_time = baseline['metrics'].get('avg_execution_time', baseline['metrics'].get('execution_time', 0))
                avg_agents = baseline['metrics'].get('avg_agents_alive', baseline['metrics'].get('agents_alive', 1))
                
                table_data.append({
                    'Abordagem': baseline['name'],
                    'Tipo': 'Baseline',
                    'Sucesso': f'{success_rate:.1%}',
                    'Tesouros': f'{avg_treasures:.1f}',
                    'Tempo': f'{avg_time:.2f}s',
                    'Agentes': f'{avg_agents:.1f}'
                })
            
            # Linha vazia entre abordagens
            table_data.append({
                'Abordagem': '',
                'Tipo': '',
                'Sucesso': '',
                'Tesouros': '',
                'Tempo': '',
                'Agentes': ''
            })
        
        # Criar DataFrame
        df = pd.DataFrame(table_data)
        
        # Criar Treeview
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        # Treeview
        columns = list(df.columns)
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings',
                           yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)
        
        # Configurar colunas
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor='center')
        
        # Adicionar dados
        for i, row in df.iterrows():
            # Destacar cabeçalhos de abordagem
            if str(row['Abordagem']).startswith('==='):
                tree.insert("", "end", values=list(row), tags=('header',))
            elif row['Abordagem'] == '':
                tree.insert("", "end", values=list(row), tags=('spacer',))
            else:
                tree.insert("", "end", values=list(row))
        
        # Configurar tags para styling
        tree.tag_configure('header', background='#e8f4f8', font=('Arial', 10, 'bold'))
        tree.tag_configure('spacer', background='#f8f8f8')
        
        # Layout
        tree.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        vsb.grid(row=0, column=1, sticky=(tk.N, tk.S))
        hsb.grid(row=1, column=0, sticky=(tk.E, tk.W))
        
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)
        
        # Botão de exportar
        export_btn = ttk.Button(parent, text="📊 Exportar para CSV", 
                               command=lambda: self.export_table_to_csv(df))
        export_btn.pack(pady=10)
    
    def create_statistics_panel(self, parent, results):
        """Criar painel de estatísticas"""
        # Frame principal com scroll
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Texto com análise
        analysis_text = scrolledtext.ScrolledText(
            main_frame,
            wrap=tk.WORD,
            font=("Arial", 10),
            height=20
        )
        analysis_text.pack(fill=tk.BOTH, expand=True)
        
        # Gerar análise
        analysis = self.generate_statistical_analysis(results)
        analysis_text.insert(tk.END, analysis)
        analysis_text.config(state=tk.DISABLED)
    
    def generate_statistical_analysis(self, results):
        """Gerar análise estatística correta distinguindo grupos e baselines"""
        analysis = "="*70 + "\n"
        analysis += "ANÁLISE ESTATÍSTICA: GRUPOS ML vs ALGORITMOS CLÁSSICOS\n"
        analysis += "="*70 + "\n\n"
        
        # Organizar resultados por abordagem
        approaches_data = {}
        for r in results:
            approach = r['approach']
            if approach not in approaches_data:
                approaches_data[approach] = []
            approaches_data[approach].append(r)
        
        analysis += "ESTRUTURA DA ANÁLISE:\n"
        analysis += "-"*50 + "\n"
        analysis += "• Para cada abordagem (A, B, C):\n"
        analysis += "  - Comparar grupos Homogêneo vs Heterogêneo\n"
        analysis += "  - Identificar melhor grupo (maior taxa de sucesso)\n"
        analysis += "  - Comparar melhor grupo com algoritmo baseline\n\n"
        
        # Análise por abordagem
        for approach, approach_results in approaches_data.items():
            analysis += f"ABORDAGEM {approach}:\n"
            analysis += "-"*30 + "\n"
            
            # Separar grupos ML e baselines
            ml_groups = [r for r in approach_results if r.get('group_type') in ['homogeneous', 'heterogeneous']]
            baselines = [r for r in approach_results if r.get('group_type') == 'baseline']
            
            if ml_groups:
                analysis += "GRUPOS DE APRENDIZAGEM DE MÁQUINA:\n"
                for group in ml_groups:
                    success_rate = group['metrics'].get('success_rate', 0)
                    avg_treasures = group['metrics'].get('avg_treasures_found', 0)
                    avg_time = group['metrics'].get('avg_execution_time', 0)
                    analysis += f"  • {group['name']}: {success_rate:.1%} sucesso, "
                    analysis += f"{avg_treasures:.1f} tesouros, {avg_time:.2f}s\n"
                
                # Identificar melhor grupo
                best_ml = max(ml_groups, key=lambda x: x['metrics'].get('success_rate', 0))
                analysis += f"  → Melhor grupo: {best_ml['name']} ({best_ml['metrics'].get('success_rate', 0):.1%})\n\n"
            
            if baselines:
                analysis += "ALGORITMOS CLÁSSICOS (BASELINES):\n"
                for baseline in baselines:
                    success_rate = baseline['metrics'].get('success_rate', 0)
                    avg_treasures = baseline['metrics'].get('avg_treasures_found', baseline['metrics'].get('treasures_found', 0))
                    avg_time = baseline['metrics'].get('avg_execution_time', baseline['metrics'].get('execution_time', 0))
                    analysis += f"  • {baseline['name']}: {success_rate:.1%} sucesso, "
                    analysis += f"{avg_treasures:.1f} tesouros, {avg_time:.2f}s\n"
                
                # Comparação ML vs Baseline
                if ml_groups:
                    best_ml_success = best_ml['metrics'].get('success_rate', 0)
                    baseline_success = baselines[0]['metrics'].get('success_rate', 0)
                    
                    analysis += f"\nCOMPARAÇÃO ML vs CLASSICAL:\n"
                    analysis += f"  • Melhor grupo ML: {best_ml_success:.1%} sucesso\n"
                    analysis += f"  • Baseline clássico: {baseline_success:.1%} sucesso\n"
                    
                    if best_ml_success > baseline_success:
                        analysis += f"  → VANTAGEM: Grupo ML superior ao baseline (+{best_ml_success - baseline_success:.1%})\n"
                    elif baseline_success > best_ml_success:
                        analysis += f"  → VANTAGEM: Baseline superior ao grupo ML (+{baseline_success - best_ml_success:.1%})\n"
                    else:
                        analysis += f"  → EMPATE: Performance similar\n"
            
            analysis += "\n"
        
        # Conclusões gerais
        analysis += "CONCLUSÕES GERAIS:\n"
        analysis += "-"*30 + "\n"
        
        # Calcular estatísticas gerais
        all_ml_success = []
        all_baseline_success = []
        
        for r in results:
            if r.get('group_type') in ['homogeneous', 'heterogeneous']:
                all_ml_success.append(r['metrics'].get('success_rate', 0))
            elif r.get('group_type') == 'baseline':
                all_baseline_success.append(r['metrics'].get('success_rate', 0))
        
        if all_ml_success and all_baseline_success:
            avg_ml = np.mean(all_ml_success)
            avg_baseline = np.mean(all_baseline_success)
            
            analysis += f"• Taxa média de sucesso ML: {avg_ml:.1%}\n"
            analysis += f"• Taxa média de sucesso Baseline: {avg_baseline:.1%}\n"
            
            if avg_ml > avg_baseline:
                analysis += f"• RESULTADO: Grupos ML superiores aos baselines clássicos\n"
            else:
                analysis += f"• RESULTADO: Baselines clássicos superiores aos grupos ML\n"
        
        analysis += "\nRECOMENDAÇÕES:\n"
        analysis += "-"*20 + "\n"
        analysis += "• Usar abordagem heterogênea quando disponível\n"
        analysis += "• Considerar baselines clássicos para problemas específicos\n"
        analysis += "• Avaliar trade-off entre complexidade ML e performance\n"
        
        return analysis
    
    def export_table_to_csv(self, df):
        """Exportar tabela para CSV (pasta data)"""
        DATA_DIR.mkdir(exist_ok=True)
        filename = DATA_DIR / f"resultados_comparacao_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        self.log(f"Tabela exportada para {filename}", "SUCCESS")
        messagebox.showinfo("Exportação", f"Dados exportados para {filename}")
    
    def clear_logs(self):
        """Limpar todos os logs"""
        self.log_text.delete(1.0, tk.END)
        self.log_history.clear()
        self.log("Logs limpos", "INFO")
    
    def export_data(self):
        """Exportar dados da simulação"""
        if not self.simulation_results:
            messagebox.showwarning("Aviso", "Nenhum dado disponível para exportar!")
            return
        
        try:
            # Criar DataFrame
            df = pd.DataFrame(self.simulation_results)
            
            # Exportar para CSV (pasta data)
            DATA_DIR.mkdir(exist_ok=True)
            filename = DATA_DIR / f"dados_simulacao_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            
            self.log(f"Dados exportados para {filename}", "SUCCESS")
            messagebox.showinfo("Exportação", f"Dados exportados para {filename}")
            
        except Exception as e:
            self.log(f"Erro ao exportar dados: {str(e)}", "ERROR")
    
    def on_closing(self):
        """Lidar com o fechamento da janela"""
        self.stop_simulation = True
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)
        self.root.destroy()

def main():
    """Função principal"""
    root = tk.Tk()
    app = IAProjectGUI(root)
    
    # Configurar fechamento
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Centralizar janela
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()