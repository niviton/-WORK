"""
Controle da Garra pelo computador (didático)
============================================

Conversa com o Arduino (garra.ino) pela porta USB/Serial, mandando os mesmos
comandos de texto que você digitaria no Monitor Serial:

    x 120   -> vai para X = 120 mm          xr 10  -> anda +10 mm em X
    z 50    -> vai para Z = 50 mm           zr -10 -> anda -10 mm em Z
    h       -> volta ao zero                zero   -> posição atual vira o zero
    !       -> PARAR                        ?      -> pede a posição

O Arduino responde com texto. Duas linhas importam para o programa:
    POS <x_mm> <z_mm> <curso_x> <curso_z>   -> posição atual
    ok                                      -> terminou o comando

Abas:
    Manual   -> setas, ir para X/Z, home, velocidade
    Programa -> grava uma lista de pontos e a garra repete sozinha
                (é assim que se "ensina" um robô industrial)

Como usar:
    1. Feche o Monitor Serial da Arduino IDE (só um programa usa a porta por vez).
    2. Rode:  python controle_pc.py COM15
    3. Use os botões, as setas do teclado, ou clique no desenho para mover a garra.

Precisa de:  pip install pyserial
"""

import json
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import serial
import serial.tools.list_ports

BAUD = 115200
ESCALA = 1.0          # pixels por mm no desenho
MARGEM = 40           # margem do desenho em pixels
GRADE_MM = 50         # espaçamento da grade do desenho

# Cores
FUNDO = "#1e1f24"
PAINEL = "#2a2c33"
TEXTO = "#e6e6e6"
APAGADO = "#8a8d96"
ROXO = "#b04fb0"        # mesma cor das peças impressas
VERDE = "#4caf50"
LARANJA = "#ff9800"
VERMELHO = "#e53935"
AZUL = "#42a5f5"


class ConexaoGarra:
    """Cuida da porta serial: envia comandos e lê respostas numa thread separada."""

    def __init__(self, ao_receber_linha):
        self.porta = None
        self.ao_receber_linha = ao_receber_linha
        self.fila_envio = []       # comandos esperando a vez
        self.aguardando_ok = 0     # quantos "ok" ainda faltam chegar
        self.lock = threading.Lock()

    def conectar(self, nome_porta):
        self.porta = serial.Serial(nome_porta, BAUD, timeout=0.1)
        # Abrir a porta reinicia o Arduino: espera ele ligar
        time.sleep(2.0)
        self.porta.reset_input_buffer()
        threading.Thread(target=self._ler_sempre, daemon=True).start()

    def desconectar(self):
        if self.porta:
            self.porta.close()
            self.porta = None
        self.fila_envio.clear()
        self.aguardando_ok = 0

    @property
    def conectado(self):
        return self.porta is not None and self.porta.is_open

    @property
    def ocupado(self):
        return self.aguardando_ok > 0 or bool(self.fila_envio)

    def enviar(self, comando):
        """Coloca o comando na fila. Só manda quando o anterior terminar ("ok")."""
        if not self.conectado:
            self.ao_receber_linha("! Não conectado: comando ignorado (" + comando + ")")
            return
        with self.lock:
            self.fila_envio.append(comando)
        self._enviar_proximo()

    def parar(self):
        """PARAR fura a fila: limpa os pendentes e manda "!" na hora."""
        with self.lock:
            self.fila_envio.clear()
        self._escrever("!")

    def _enviar_proximo(self):
        with self.lock:
            if self.aguardando_ok > 0 or not self.fila_envio:
                return
            comando = self.fila_envio.pop(0)
            # Conta o "ok" esperado ainda dentro do lock: assim nunca há um instante
            # em que a fila está vazia e nada está pendente (pareceria "parado")
            self.aguardando_ok += 1
        self._escrever(comando, ja_contado=True)

    def _escrever(self, comando, ja_contado=False):
        if not self.conectado:
            return
        if not ja_contado:
            with self.lock:
                self.aguardando_ok += 1
        try:
            self.porta.write((comando + "\n").encode())
        except serial.SerialException as erro:
            self._perdeu_conexao(erro)
            return
        self.ao_receber_linha(">> " + comando)

    def _perdeu_conexao(self, erro):
        """O cabo USB caiu (ou outro programa pegou a porta): fecha tudo e avisa a tela."""
        try:
            self.porta.close()
        except Exception:
            pass
        self.porta = None
        self.fila_envio.clear()
        self.aguardando_ok = 0
        self.ao_receber_linha(f"!! CONEXAO PERDIDA: {erro}")

    def _ler_sempre(self):
        while self.conectado:
            try:
                linha = self.porta.readline().decode(errors="replace").strip()
            except (TypeError, AttributeError):
                break      # porta fechada por nós (Desconectar)
            except serial.SerialException as erro:
                self._perdeu_conexao(erro)
                break
            if not linha:
                continue
            if linha == "ok":
                with self.lock:
                    self.aguardando_ok = max(0, self.aguardando_ok - 1)
                self._enviar_proximo()
            self.ao_receber_linha(linha)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Controle da Garra")
        self.configure(bg=FUNDO)
        self.resizable(False, False)

        self.linhas = queue.Queue()
        self.garra = ConexaoGarra(self.linhas.put)
        self.x_mm = 0.0
        self.z_mm = 0.0
        self.curso_x = 390.0
        self.curso_z = 390.0
        self.alvo = None            # (x, z) para onde a garra está indo
        self.z_planejado = 0.0      # Z depois do último comando da fila
        self.pontos = []            # programa: lista de [x, z]
        self.executando = False
        self.ponto_atual = -1

        self._estilo()
        self._montar_tela()
        self._atualizar_portas()
        self.after(50, self._laco)

        # Atalhos de teclado (não valem enquanto digita num campo)
        for tecla, eixo, sinal in (("<Left>", "x", +1), ("<Right>", "x", -1),
                                   ("<Up>", "z", -1), ("<Down>", "z", +1)):
            self.bind(tecla, lambda e, a=eixo, s=sinal: self._tecla_seta(e, a, s))
        self.bind("<space>", lambda e: None if self._digitando(e) else self.parar())
        self.bind("<Escape>", lambda e: self.parar())

    # ------------------------------------------------------------------ visual
    def _estilo(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".", background=FUNDO, foreground=TEXTO, fieldbackground=PAINEL,
                    bordercolor="#3a3d46", font=("Segoe UI", 10))
        s.configure("TFrame", background=FUNDO)
        s.configure("Painel.TFrame", background=PAINEL)
        s.configure("TLabel", background=FUNDO, foreground=TEXTO)
        s.configure("Painel.TLabel", background=PAINEL)
        s.configure("Apagado.TLabel", background=PAINEL, foreground=APAGADO, font=("Segoe UI", 9))
        s.configure("Pos.TLabel", background=PAINEL, font=("Consolas", 20, "bold"))
        s.configure("TLabelframe", background=PAINEL, bordercolor="#3a3d46")
        s.configure("TLabelframe.Label", background=PAINEL, foreground=APAGADO)
        s.configure("TButton", background="#3a3d46", foreground=TEXTO, borderwidth=0, padding=6)
        s.map("TButton", background=[("active", "#4a4e5a"), ("disabled", "#2e3037")],
              foreground=[("disabled", "#666")])
        s.configure("Seta.TButton", font=("Segoe UI", 14, "bold"), padding=(10, 4))
        s.configure("Destaque.TButton", background=ROXO)
        s.map("Destaque.TButton", background=[("active", "#c565c5")])
        s.configure("TNotebook", background=FUNDO, borderwidth=0)
        s.configure("TNotebook.Tab", background="#2e3037", foreground=APAGADO, padding=(14, 6))
        s.map("TNotebook.Tab", background=[("selected", PAINEL)], foreground=[("selected", TEXTO)])
        s.configure("TRadiobutton", background=PAINEL, foreground=TEXTO)
        s.configure("TCheckbutton", background=PAINEL, foreground=TEXTO)
        s.configure("TEntry", fieldbackground="#1a1b20", foreground=TEXTO, insertcolor=TEXTO)
        s.configure("TCombobox", fieldbackground="#1a1b20", foreground=TEXTO)

    def _montar_tela(self):
        # --- Topo: conexão e estado
        topo = ttk.Frame(self, padding=(12, 10))
        topo.grid(row=0, column=0, columnspan=2, sticky="ew")
        ttk.Label(topo, text="Controle da Garra", font=("Segoe UI", 14, "bold")).pack(side="left")
        self.estado = tk.Label(topo, text="● Desconectado", bg=FUNDO, fg=APAGADO,
                               font=("Segoe UI", 10, "bold"))
        self.estado.pack(side="right")
        self.botao_conectar = ttk.Button(topo, text="Conectar", command=self._conectar)
        self.botao_conectar.pack(side="right", padx=8)
        ttk.Button(topo, text="↻", width=3, command=self._atualizar_portas).pack(side="right")
        self.combo_porta = ttk.Combobox(topo, width=10, state="readonly")
        self.combo_porta.pack(side="right", padx=4)
        ttk.Label(topo, text="Porta").pack(side="right")

        # --- Desenho da estrutura
        area = ttk.Frame(self, padding=(12, 0))
        area.grid(row=1, column=0, sticky="n")
        self.canvas = tk.Canvas(area, bg="#15161a", highlightthickness=0, cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._clique_no_desenho)
        self.canvas.bind("<Motion>", self._mouse_no_desenho)
        self.canvas.bind("<Leave>", lambda e: self.canvas.delete("mouse"))
        ttk.Label(area, text="Clique no desenho para mover a garra até o ponto",
                  foreground=APAGADO, font=("Segoe UI", 9)).pack(pady=(4, 0))

        # --- Lado direito: posição, abas e PARAR
        lado = ttk.Frame(self, padding=(0, 0, 12, 0))
        lado.grid(row=1, column=1, sticky="n")

        caixa_pos = ttk.Frame(lado, style="Painel.TFrame", padding=10)
        caixa_pos.pack(fill="x")
        self.rotulo_pos = ttk.Label(caixa_pos, style="Pos.TLabel",
                                    text="X   0.0 mm\nZ   0.0 mm")
        self.rotulo_pos.pack(anchor="w")
        self.rotulo_alvo = ttk.Label(caixa_pos, style="Apagado.TLabel", text=" ")
        self.rotulo_alvo.pack(anchor="w")

        abas = ttk.Notebook(lado)
        abas.pack(fill="x", pady=8)
        abas.add(self._aba_manual(abas), text="Manual")
        abas.add(self._aba_programa(abas), text="Programa")

        tk.Button(lado, text="■  PARAR   (espaço)", bg=VERMELHO, fg="white",
                  activebackground="#ff5252", activeforeground="white", relief="flat",
                  font=("Segoe UI", 12, "bold"), pady=8,
                  command=self.parar).pack(fill="x")

        # --- Console
        console = ttk.Frame(self, padding=12)
        console.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.log = tk.Text(console, height=7, font=("Consolas", 9), state="disabled",
                           bg="#111215", fg="#cfcfcf", relief="flat", padx=6, pady=4)
        self.log.pack(fill="x")
        self.log.tag_config("enviado", foreground=AZUL)
        self.log.tag_config("erro", foreground=VERMELHO)
        self.log.tag_config("info", foreground=APAGADO)
        linha_cmd = ttk.Frame(console)
        linha_cmd.pack(fill="x", pady=(6, 0))
        ttk.Label(linha_cmd, text="Comando manual:").pack(side="left")
        self.entrada_cmd = ttk.Entry(linha_cmd)
        self.entrada_cmd.pack(side="left", fill="x", expand=True, padx=6)
        self.entrada_cmd.bind("<Return>", self._enviar_comando_digitado)
        ttk.Label(linha_cmd, text="ex.: x 100 · zr -20 · ? · h",
                  foreground=APAGADO, font=("Segoe UI", 9)).pack(side="left")

        self._desenhar()

    def _aba_manual(self, pai):
        aba = ttk.Frame(pai, style="Painel.TFrame", padding=10)

        ttk.Label(aba, text="Passo das setas (mm)", style="Painel.TLabel").pack(anchor="w")
        self.passo = tk.DoubleVar(value=10)
        linha = ttk.Frame(aba, style="Painel.TFrame")
        linha.pack(anchor="w", pady=(2, 6))
        for v in (1, 5, 10, 50, 100):
            ttk.Radiobutton(linha, text=str(v), value=v, variable=self.passo).pack(side="left", padx=2)

        setas = ttk.Frame(aba, style="Painel.TFrame")
        setas.pack(pady=4)
        ttk.Button(setas, text="▲", style="Seta.TButton",
                   command=lambda: self.mover_relativo("z", -1)).grid(row=0, column=1, pady=2)
        ttk.Button(setas, text="◀", style="Seta.TButton",
                   command=lambda: self.mover_relativo("x", +1)).grid(row=1, column=0, padx=2)
        ttk.Button(setas, text="⌂", style="Seta.TButton",
                   command=self.home).grid(row=1, column=1)
        ttk.Button(setas, text="▶", style="Seta.TButton",
                   command=lambda: self.mover_relativo("x", -1)).grid(row=1, column=2, padx=2)
        ttk.Button(setas, text="▼", style="Seta.TButton",
                   command=lambda: self.mover_relativo("z", +1)).grid(row=2, column=1, pady=2)
        ttk.Label(aba, text="Teclado: setas movem · espaço para", style="Apagado.TLabel").pack()

        ir = ttk.LabelFrame(aba, text="Ir para (mm)", padding=8)
        ir.pack(fill="x", pady=8)
        ttk.Label(ir, text="X", style="Painel.TLabel").grid(row=0, column=0, padx=(0, 4))
        self.entrada_x = ttk.Entry(ir, width=8)
        self.entrada_x.grid(row=0, column=1)
        ttk.Label(ir, text="Z", style="Painel.TLabel").grid(row=0, column=2, padx=(10, 4))
        self.entrada_z = ttk.Entry(ir, width=8)
        self.entrada_z.grid(row=0, column=3)
        ttk.Button(ir, text="Ir", style="Destaque.TButton",
                   command=self._ir_para_digitado).grid(row=0, column=4, padx=(8, 0))
        for campo in (self.entrada_x, self.entrada_z):
            campo.bind("<Return>", lambda e: self._ir_para_digitado())
        self.juntos = tk.BooleanVar(value=True)
        ttk.Checkbutton(ir, text="Mover X e Z ao mesmo tempo (linha reta)",
                        variable=self.juntos).grid(row=1, column=0, columnspan=5,
                                                   sticky="w", pady=(6, 0))

        vel = ttk.LabelFrame(aba, text="Velocidade", padding=8)
        vel.pack(fill="x")
        self.velocidade = tk.IntVar(value=2000)
        self.rotulo_vel = ttk.Label(vel, style="Painel.TLabel")
        self.rotulo_vel.pack(anchor="w")
        tk.Scale(vel, from_=400, to=2800, resolution=100, orient="horizontal",
                 variable=self.velocidade, showvalue=False, length=220,
                 bg=PAINEL, fg=TEXTO, troughcolor="#1a1b20", highlightthickness=0,
                 activebackground=ROXO, command=lambda v: self._mostrar_vel()).pack(fill="x")
        ttk.Button(vel, text="Aplicar velocidade", command=self._aplicar_velocidade).pack(fill="x", pady=(4, 0))
        self._mostrar_vel()

        ttk.Button(aba, text="Definir posição atual como zero",
                   command=lambda: self.garra.enviar("zero")).pack(fill="x", pady=(8, 0))
        return aba

    def _aba_programa(self, pai):
        aba = ttk.Frame(pai, style="Painel.TFrame", padding=10)
        ttk.Label(aba, text="Mova a garra e grave cada ponto.\nDepois clique em Executar.",
                  style="Apagado.TLabel").pack(anchor="w")

        self.lista = tk.Listbox(aba, height=9, font=("Consolas", 10), bg="#1a1b20", fg=TEXTO,
                                selectbackground=ROXO, relief="flat", highlightthickness=0,
                                activestyle="none")
        self.lista.pack(fill="x", pady=6)
        self.lista.bind("<Double-Button-1>", lambda e: self._ir_ponto_selecionado())

        b1 = ttk.Frame(aba, style="Painel.TFrame")
        b1.pack(fill="x")
        ttk.Button(b1, text="＋ Gravar ponto atual", style="Destaque.TButton",
                   command=self._gravar_ponto).pack(side="left", fill="x", expand=True)
        ttk.Button(b1, text="Remover", command=self._remover_ponto).pack(side="left", padx=(4, 0))

        b2 = ttk.Frame(aba, style="Painel.TFrame")
        b2.pack(fill="x", pady=4)
        ttk.Button(b2, text="↑", width=3, command=lambda: self._mover_na_lista(-1)).pack(side="left")
        ttk.Button(b2, text="↓", width=3, command=lambda: self._mover_na_lista(+1)).pack(side="left", padx=4)
        ttk.Button(b2, text="Limpar", command=self._limpar_pontos).pack(side="left")
        ttk.Button(b2, text="Abrir", command=self._abrir_programa).pack(side="right")
        ttk.Button(b2, text="Salvar", command=self._salvar_programa).pack(side="right", padx=4)

        self.repetir = tk.BooleanVar(value=False)
        ttk.Checkbutton(aba, text="Repetir sem parar", variable=self.repetir).pack(anchor="w", pady=4)
        self.botao_executar = ttk.Button(aba, text="▶  Executar programa", style="Destaque.TButton",
                                         command=self._executar_programa)
        self.botao_executar.pack(fill="x")
        ttk.Label(aba, text="Duplo clique num ponto: vai até ele", style="Apagado.TLabel").pack(pady=(4, 0))
        return aba

    # -------------------------------------------------------------- conexão
    def _atualizar_portas(self):
        encontradas = serial.tools.list_ports.comports()
        portas = [p.device for p in encontradas]
        self.combo_porta["values"] = portas
        if portas and not self.combo_porta.get():
            # A placa Anet usa o chip USB CH340: escolhe ela sozinho se estiver ligada
            ch340 = [p.device for p in encontradas if "CH340" in p.description]
            self.combo_porta.set(ch340[0] if ch340 else portas[0])

    def _conectar(self):
        if self.garra.conectado:
            self.executando = False
            self.garra.desconectar()
            self.botao_conectar.config(text="Conectar")
            return
        porta = self.combo_porta.get()
        if not porta:
            self._escrever_log("Nenhuma porta selecionada.", "erro")
            return
        self.estado.config(text="● Conectando (o Arduino reinicia)...", fg=LARANJA)
        self.update()
        try:
            self.garra.conectar(porta)
        except serial.SerialException as erro:
            self._escrever_log(f"Não conectou: {erro}", "erro")
            self._escrever_log("O Monitor Serial da Arduino IDE está aberto? Feche e tente de novo.", "info")
            return
        self.botao_conectar.config(text="Desconectar")
        self._escrever_log(f"Conectado em {porta}. Posição inicial = zero.", "info")
        self.garra.enviar("?")

    # ------------------------------------------------------------ movimento
    @staticmethod
    def _digitando(evento):
        return isinstance(evento.widget, (tk.Entry, tk.Listbox))

    def _tecla_seta(self, evento, eixo, sinal):
        if not self._digitando(evento):
            self.mover_relativo(eixo, sinal)

    def mover_relativo(self, eixo, sinal):
        # Não empilha movimentos quando a pessoa segura a tecla
        if self.garra.ocupado or self.executando:
            return
        passo = sinal * self.passo.get()
        if eixo == "x":
            self.alvo = (min(max(self.x_mm + passo, 0), self.curso_x), self.z_mm)
        else:
            self.alvo = (self.x_mm, min(max(self.z_mm + passo, 0), self.curso_z))
        self.z_planejado = self.alvo[1]
        self.garra.enviar(f"{eixo}r {passo:g}")

    def ir_para(self, x, z):
        """Juntos: X e Z em linha reta ao mesmo tempo (comando "m").
        Um de cada vez (ordem segura): se for descer, anda o X primeiro;
        se for subir, sobe o Z primeiro -> a garra não arrasta em nada no caminho."""
        x = min(max(x, 0), self.curso_x)
        z = min(max(z, 0), self.curso_z)
        if self.juntos.get():
            self.garra.enviar(f"m {x:.1f} {z:.1f}")
        elif z > self.z_planejado:
            self.garra.enviar(f"x {x:.1f}")
            self.garra.enviar(f"z {z:.1f}")
        else:
            self.garra.enviar(f"z {z:.1f}")
            self.garra.enviar(f"x {x:.1f}")
        self.z_planejado = z
        self.alvo = (x, z)

    def home(self):
        self.alvo = (0.0, 0.0)
        self.z_planejado = 0.0
        self.garra.enviar("h")

    def parar(self):
        self.executando = False
        self.garra.parar()
        self._atualizar_lista()

    def _ir_para_digitado(self):
        try:
            x = float(self.entrada_x.get().replace(",", ".") or self.x_mm)
            z = float(self.entrada_z.get().replace(",", ".") or self.z_mm)
        except ValueError:
            self._escrever_log("Digite números em X e Z.", "erro")
            return
        self.ir_para(x, z)

    def _pixel_para_mm(self, px, py):
        # O X cresce para a ESQUERDA: o zero fica na borda direita do desenho
        x = (MARGEM + self.curso_x * ESCALA - px) / ESCALA
        z = (py - MARGEM) / ESCALA
        return x, z

    def _mm_para_pixel(self, x, z):
        return MARGEM + (self.curso_x - x) * ESCALA, MARGEM + z * ESCALA

    def _clique_no_desenho(self, evento):
        if self.executando:
            return
        x, z = self._pixel_para_mm(evento.x, evento.y)
        self.ir_para(x, z)

    def _mouse_no_desenho(self, evento):
        x, z = self._pixel_para_mm(evento.x, evento.y)
        c = self.canvas
        c.delete("mouse")
        if 0 <= x <= self.curso_x and 0 <= z <= self.curso_z:
            c.create_text(evento.x + 12, evento.y - 12, anchor="w", tags="mouse",
                          text=f"X {x:.0f}  Z {z:.0f}", fill=AZUL, font=("Consolas", 9))

    def _mostrar_vel(self):
        v = self.velocidade.get()
        self.rotulo_vel.config(text=f"{v} passos/s  ≈ {v / 33.6:.0f} mm/s")

    def _aplicar_velocidade(self):
        v = self.velocidade.get()
        self.garra.enviar(f"vx {v}")
        self.garra.enviar(f"vz {v}")

    def _enviar_comando_digitado(self, _evento):
        comando = self.entrada_cmd.get().strip()
        if comando:
            self.garra.enviar(comando)
            self.entrada_cmd.delete(0, "end")

    # ------------------------------------------------------------- programa
    def _gravar_ponto(self):
        self.pontos.append([round(self.x_mm, 1), round(self.z_mm, 1)])
        self._atualizar_lista()
        self.lista.see("end")

    def _remover_ponto(self):
        sel = self.lista.curselection()
        if sel:
            del self.pontos[sel[0]]
            self._atualizar_lista()

    def _limpar_pontos(self):
        self.pontos.clear()
        self._atualizar_lista()

    def _mover_na_lista(self, direcao):
        sel = self.lista.curselection()
        if not sel:
            return
        i, j = sel[0], sel[0] + direcao
        if 0 <= j < len(self.pontos):
            self.pontos[i], self.pontos[j] = self.pontos[j], self.pontos[i]
            self._atualizar_lista()
            self.lista.selection_set(j)

    def _ir_ponto_selecionado(self):
        sel = self.lista.curselection()
        if sel and not self.executando:
            self.ir_para(*self.pontos[sel[0]])

    def _atualizar_lista(self):
        self.lista.delete(0, "end")
        for i, (x, z) in enumerate(self.pontos):
            marca = "▶" if self.executando and i == self.ponto_atual else " "
            self.lista.insert("end", f"{marca} {i + 1:2d}.  X {x:6.1f}   Z {z:6.1f}")
        self.botao_executar.config(text="⏸  Executando..." if self.executando
                                   else "▶  Executar programa")
        self._desenhar()

    def _executar_programa(self):
        if not self.pontos or self.executando or not self.garra.conectado:
            return
        self.executando = True
        self.ponto_atual = -1
        self._proximo_ponto()

    def _proximo_ponto(self):
        self.ponto_atual += 1
        if self.ponto_atual >= len(self.pontos):
            if self.repetir.get():
                self.ponto_atual = 0
            else:
                self.executando = False
                self._escrever_log("Programa terminado.", "info")
                self._atualizar_lista()
                return
        self.ir_para(*self.pontos[self.ponto_atual])
        self._atualizar_lista()

    def _salvar_programa(self):
        caminho = filedialog.asksaveasfilename(defaultextension=".json",
                                               filetypes=[("Programa da garra", "*.json")])
        if caminho:
            with open(caminho, "w", encoding="utf-8") as f:
                json.dump(self.pontos, f, indent=2)
            self._escrever_log(f"Programa salvo em {caminho}", "info")

    def _abrir_programa(self):
        caminho = filedialog.askopenfilename(filetypes=[("Programa da garra", "*.json")])
        if caminho:
            with open(caminho, encoding="utf-8") as f:
                self.pontos = [[float(x), float(z)] for x, z in json.load(f)]
            self._atualizar_lista()

    # ------------------------------------------------------ respostas/desenho
    def _laco(self):
        """Roda a cada 50 ms: trata respostas do Arduino e avança o programa."""
        while not self.linhas.empty():
            self._tratar_linha(self.linhas.get())

        if self.executando and not self.garra.ocupado:
            self._proximo_ponto()
        if not self.garra.ocupado and not self.executando:
            self.alvo = None
            self.z_planejado = self.z_mm

        self._atualizar_estado()
        self.after(50, self._laco)

    def _tratar_linha(self, linha):
        if linha.startswith("!! CONEXAO PERDIDA"):
            self.executando = False
            self.botao_conectar.config(text="Conectar")
            self._escrever_log(linha, "erro")
            self._escrever_log("Confira o cabo USB e clique em Conectar. "
                               "A posição foi perdida: leve a garra ao zero.", "info")
        elif linha.startswith("POS "):
            try:
                self.x_mm, self.z_mm, self.curso_x, self.curso_z = map(float, linha.split()[1:5])
            except ValueError:
                return
            self.rotulo_pos.config(text=f"X {self.x_mm:6.1f} mm\nZ {self.z_mm:6.1f} mm")
            self._desenhar()
        elif linha.startswith(">>"):
            self._escrever_log(linha, "enviado")
        elif linha.startswith(("X = ", "Z = ")) or linha == "ok":
            pass    # já aparece no painel de posição
        elif linha.startswith("!") or "desconhecido" in linha or "interrompido" in linha:
            self._escrever_log(linha, "erro")
        else:
            self._escrever_log(linha)

    def _atualizar_estado(self):
        if not self.garra.conectado:
            texto, cor = "● Desconectado", APAGADO
        elif self.executando:
            texto, cor = f"● Executando ponto {self.ponto_atual + 1}/{len(self.pontos)}", AZUL
        elif self.garra.ocupado:
            texto, cor = "● Movendo...", LARANJA
        else:
            texto, cor = "● Pronto", VERDE
        if self.estado.cget("text") != texto:
            self.estado.config(text=texto, fg=cor)
            self._desenhar()
        self.rotulo_alvo.config(text=f"indo para X {self.alvo[0]:.1f}  Z {self.alvo[1]:.1f}"
                                if self.alvo else " ")

    def _desenhar(self):
        c = self.canvas
        c.delete("all")
        largura = int(self.curso_x * ESCALA) + 2 * MARGEM
        altura = int(self.curso_z * ESCALA) + 2 * MARGEM
        c.config(width=largura, height=altura)
        esq, topo = MARGEM, MARGEM
        dir_ = MARGEM + self.curso_x * ESCALA
        base = MARGEM + self.curso_z * ESCALA

        # Grade em mm
        for mm in range(0, int(self.curso_x) + 1, GRADE_MM):
            px, _ = self._mm_para_pixel(mm, 0)
            c.create_line(px, topo, px, base, fill="#23252b")
            c.create_text(px, base + 12, text=str(mm), fill="#5d606a", font=("Consolas", 8))
        for mm in range(0, int(self.curso_z) + 1, GRADE_MM):
            _, py = self._mm_para_pixel(0, mm)
            c.create_line(esq, py, dir_, py, fill="#23252b")
            c.create_text(dir_ + 16, py, text=str(mm), fill="#5d606a", font=("Consolas", 8))
        c.create_rectangle(esq, topo, dir_, base, outline="#3a3d46")
        c.create_text(esq, base + 26, anchor="w", text="← X cresce para a esquerda",
                      fill="#5d606a", font=("Segoe UI", 8))
        c.create_text(dir_ + 30, topo - 16, anchor="e", text="ZERO", fill=VERDE,
                      font=("Segoe UI", 8, "bold"))

        # Pontos do programa, ligados na ordem
        coords = [self._mm_para_pixel(x, z) for x, z in self.pontos]
        if len(coords) > 1:
            c.create_line(*[v for p in coords for v in p], fill="#5a3a5a", dash=(4, 3))
        for i, (px, py) in enumerate(coords):
            ativo = self.executando and i == self.ponto_atual
            c.create_oval(px - 7, py - 7, px + 7, py + 7,
                          fill=AZUL if ativo else "#3a2a3a", outline=ROXO)
            c.create_text(px, py, text=str(i + 1), fill=TEXTO, font=("Segoe UI", 7, "bold"))

        # Alvo
        if self.alvo:
            ax, az = self._mm_para_pixel(*self.alvo)
            c.create_line(ax - 10, az, ax + 10, az, fill=LARANJA, width=2)
            c.create_line(ax, az - 10, ax, az + 10, fill=LARANJA, width=2)
            c.create_oval(ax - 6, az - 6, ax + 6, az + 6, outline=LARANJA, width=2)

        # Estrutura: cremalheira horizontal + vertical descendo até a garra
        px, pz = self._mm_para_pixel(self.x_mm, self.z_mm)
        c.create_line(esq, topo - 6, dir_ + 10, topo - 6, fill=ROXO, width=6)
        c.create_rectangle(dir_ + 2, topo - 18, dir_ + 22, topo + 6, fill="#555", outline="")
        c.create_line(px, topo - 6, px, pz, fill=ROXO, width=6)
        c.create_rectangle(px - 9, topo - 16, px + 9, topo + 4, fill="#555", outline="")
        # Garra
        c.create_rectangle(px - 12, pz, px + 12, pz + 10, fill="#444", outline="#777")
        c.create_line(px - 10, pz + 10, px - 10, pz + 24, fill="#aaa", width=3)
        c.create_line(px + 10, pz + 10, px + 10, pz + 24, fill="#aaa", width=3)
        c.create_oval(px - 3, pz - 3, px + 3, pz + 3, fill=VERDE, outline="")

    def _escrever_log(self, texto, tag=None):
        self.log.config(state="normal")
        self.log.insert("end", texto + "\n", tag)
        self.log.see("end")
        self.log.config(state="disabled")


if __name__ == "__main__":
    app = App()
    # "python controle_pc.py COM15" já conecta direto nessa porta
    if len(sys.argv) > 1:
        app.combo_porta.set(sys.argv[1])
        app.after(300, app._conectar)
    app.mainloop()
