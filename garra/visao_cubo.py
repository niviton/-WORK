"""
VisÃ£o da Garra - aproximar o eixo X do cubo laranja (didÃ¡tico)
==============================================================

A cÃ¢mera (GoPro em modo webcam) fica presa no carro do eixo X, olhando PARA A FRENTE,
na direÃ§Ã£o em que o X anda. Por isso, quando o X anda, o cubo sobe ou desce NA IMAGEM:
    cubo lÃ¡ em cima na imagem  -> longe da garra
    cubo embaixo na imagem     -> perto da garra

O programa:
  1. ACHA O CUBO pela cor (o laranja se destaca muito do chÃ£o/mesa).
     NÃ£o precisa de IA para isso: Ã© "limiarizaÃ§Ã£o de cor" no espaÃ§o HSV.
  2. Compara a ALTURA do cubo na imagem com a LINHA ALVO (verde) = ponto ideal
     para a garra pegar.
  3. MOVE O X para diminuir a diferenÃ§a, olha de novo, move de novo... sem parar.
     Erro grande -> movimento grande e rÃ¡pido; erro pequeno -> movimento curto.
     Ã‰ um controle PROPORCIONAL em malha fechada ("servo visual").
     Se o cubo sair do lugar, a garra vai atrÃ¡s dele sozinha.

  A LINHA LIMITE (vermelha) sÃ³ Ã© referÃªncia visual: com a garra no zero (X = 0),
  um cubo nessa altura estÃ¡ na distÃ¢ncia mÃ¡xima do trilho. Ela nÃ£o bloqueia nada,
  porque a cÃ¢mera anda junto com o X e a altura do cubo na imagem muda.
  Se o cubo estiver longe demais, a garra para sozinha no fim do trilho.

Mouse (na janela da cÃ¢mera):
  Clique ESQUERDO no cubo -> aprende a cor do cubo e liga o "seguir"
  Clique DIREITO          -> muda a LINHA ALVO (ponto ideal) para a altura clicada

Teclado:
  a  liga/desliga o "seguir"       espaÃ§o  PARAR
  d  linha LIMITE = altura atual do cubo
  c  calibrar de novo              m       mostrar a mÃ¡scara (o que o programa "vÃª")
  j  X +10 mm     l  X -10 mm      h       home          q  sair

Uso:
  python visao_cubo.py                  (cÃ¢mera 2 = GoPro, porta COM15)
  python visao_cubo.py --camera 0
  python visao_cubo.py --sem-garra      (sÃ³ testar a detecÃ§Ã£o, sem Arduino)
"""

import argparse
import json
import queue
import time
from pathlib import Path

import cv2
import numpy as np

from controle_pc import ConexaoGarra

CONFIG = Path(__file__).with_name("visao_config.json")
LARGURA = 960            # a imagem Ã© reduzida para esta largura (mais rÃ¡pido)
AREA_MINIMA = 300        # blobs menores que isso (pxÂ²) sÃ£o ignorados (ruÃ­do)
ESPERA_CAMERA = 0.8     # s: depois de mover, espera a imagem da GoPro "alcanÃ§ar" o movimento
MEDIDAS = 3              # mÃ©dia de N imagens antes de decidir (tira o tremido)
GANHO = 0.6              # fraÃ§Ã£o do erro corrigida por vez (1.0 = tudo de uma vez)
PASSO_MIN_MM = 0.5       # abaixo disso o motor nÃ£o consegue um passo Ãºtil
PASSO_MAX_MM = 150.0     # maior correÃ§Ã£o de uma vez sÃ³
DIST_CALIBRACAO_MM = 40.0

PADRAO = {
    "camera": 2,
    "alvo_y": 344,          # altura (px) do cubo no ponto ideal (medida nas suas fotos)
    "limite_y": 198,        # altura (px) do cubo na distÃ¢ncia mÃ¡xima, com X = 0 (sÃ³ referÃªncia)
    "hsv_min": [5, 120, 120],
    "hsv_max": [22, 255, 255],
    "px_por_mm": None,      # medido na calibraÃ§Ã£o: quantos px o cubo anda na imagem por mm de X
    "tolerancia_px": 5,
}


def carregar_config():
    cfg = dict(PADRAO)
    if CONFIG.exists():
        cfg.update(json.loads(CONFIG.read_text(encoding="utf-8")))
    return cfg


def salvar_config(cfg):
    CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- detecÃ§Ã£o
def achar_cubo(imagem, cfg, perto_de=None):
    """Devolve ((centro_x, centro_y, contorno), mascara) do blob laranja, ou (None, mascara)."""
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, np.array(cfg["hsv_min"]), np.array(cfg["hsv_max"]))
    # Tira pontinhos soltos e fecha buracos
    nucleo = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, nucleo)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidatos = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < AREA_MINIMA:
            continue
        m = cv2.moments(c)
        candidatos.append((area, m["m10"] / m["m00"], m["m01"] / m["m00"], c))
    if not candidatos:
        return None, mascara
    if perto_de is not None:
        # Mais de um objeto laranja: fica com o mais perto de onde o cubo estava
        escolhido = min(candidatos, key=lambda k: (k[1] - perto_de[0]) ** 2 + (k[2] - perto_de[1]) ** 2)
    else:
        escolhido = max(candidatos, key=lambda k: k[0])
    _, cx, cy, c = escolhido
    return (cx, cy, c), mascara


def aprender_cor(imagem, x, y, cfg):
    """Pega a cor mÃ©dia num quadradinho em volta do clique e monta a faixa HSV."""
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    amostra = hsv[max(0, y - 4):y + 5, max(0, x - 4):x + 5].reshape(-1, 3)
    h, s, v = (int(c) for c in np.median(amostra, axis=0))   # int puro: o JSON nÃ£o aceita numpy
    cfg["hsv_min"] = [max(0, h - 10), max(60, s - 70), max(60, v - 80)]
    cfg["hsv_max"] = [min(179, h + 10), 255, 255]
    return h, s, v


# ------------------------------------------------------------------ programa
class Visao:
    def __init__(self, args):
        self.cfg = carregar_config()
        if args.camera is not None:
            self.cfg["camera"] = args.camera

        self.linhas = queue.Queue()
        self.garra = ConexaoGarra(self.linhas.put)
        self.x_mm = 0.0
        self.curso_x = 390.0
        if not args.sem_garra:
            try:
                print(f"Conectando na garra ({args.porta})...", flush=True)
                self.garra.conectar(args.porta)
                self.garra.enviar("?")
                print("Garra conectada.", flush=True)
            except Exception as erro:
                print(f"Sem garra: {erro}  -> sÃ³ a detecÃ§Ã£o vai funcionar", flush=True)

        self.seguir = True            # tenta sempre deixar o cubo na linha alvo
        self.calibrando = False
        self.etapa = 0
        self.pronto_em = 0.0          # quando a imagem jÃ¡ mostra o fim do Ãºltimo movimento
        self.medidas = []
        self.calib = {}
        self.px_por_mm_atual = self.cfg["px_por_mm"]   # vai sendo ajustado durante o movimento
        self.antes_do_movimento = None                 # (x_mm, y_px) antes do Ãºltimo movimento
        self.contra_sentido = 0                        # movimentos seguidos que afastaram o cubo
        self.ultimo_cubo = None
        self.mostrar_mascara = False
        self.mensagem = "Procurando o cubo..."
        self.frame = None

    # --- eventos
    def mouse(self, evento, x, y, _flags, _param):
        if self.frame is None:
            return
        if evento == cv2.EVENT_LBUTTONDOWN:
            h, s, v = aprender_cor(self.frame, x, y, self.cfg)
            salvar_config(self.cfg)
            self.ultimo_cubo = (x, y)
            self.seguir = True
            self.mensagem = f"Cor aprendida (H{h} S{s} V{v}). Seguindo o cubo."
        elif evento == cv2.EVENT_RBUTTONDOWN:
            self.cfg["alvo_y"] = y
            salvar_config(self.cfg)
            self.mensagem = f"Linha ALVO (ponto ideal) em y = {y}px."

    def tecla(self, k):
        if k == ord(" "):
            self.parar("PARADO. Tecla A volta a seguir o cubo.")
        elif k == ord("a"):
            self.seguir = not self.seguir
            if not self.seguir:
                self.garra.parar()
            self.mensagem = "Seguindo o cubo." if self.seguir else "Seguir DESLIGADO."
        elif k == ord("c"):
            self.iniciar_calibracao()
        elif k == ord("d"):
            if self.ultimo_cubo:
                self.cfg["limite_y"] = int(self.ultimo_cubo[1])
                salvar_config(self.cfg)
                self.mensagem = f"Linha LIMITE (distÃ¢ncia mÃ¡xima) em y = {self.cfg['limite_y']}px."
        elif k == ord("j"):
            self.garra.enviar("xr 10")
        elif k == ord("l"):
            self.garra.enviar("xr -10")
        elif k == ord("h"):
            self.seguir = False
            self.garra.enviar("h")
            self.mensagem = "Home. Seguir DESLIGADO (tecla A liga)."
        elif k == ord("m"):
            self.mostrar_mascara = not self.mostrar_mascara
        elif k == ord("i") and self.cfg["px_por_mm"]:
            self.garra.parar()
            self.inverter_sentido("Sentido do X invertido (tecla I).")

    def parar(self, msg):
        self.seguir = False
        self.calibrando = False
        self.garra.parar()
        self.mensagem = msg

    # --- respostas do Arduino
    def ler_garra(self):
        while not self.linhas.empty():
            linha = self.linhas.get()
            if linha.startswith("POS "):
                partes = linha.split()
                self.x_mm, self.curso_x = float(partes[1]), float(partes[3])
            elif linha.startswith("!!") or "desconhecido" in linha:
                print(linha, flush=True)

    def imagem_estavel(self):
        """True quando a garra parou E a cÃ¢mera jÃ¡ teve tempo de mostrar isso."""
        if self.garra.ocupado:
            self.pronto_em = time.time() + ESPERA_CAMERA
            return False
        return time.time() >= self.pronto_em

    def medir(self, cubo):
        """Junta MEDIDAS alturas seguidas do cubo; devolve a mÃ©dia quando completar."""
        self.medidas.append(cubo[1])
        if len(self.medidas) < MEDIDAS:
            return None
        media = sum(self.medidas) / len(self.medidas)
        self.medidas = []
        return media

    def mover(self, mm):
        self.garra.enviar(f"xr {mm:.2f}")
        self.pronto_em = time.time() + ESPERA_CAMERA

    # --- calibraÃ§Ã£o: mede quantos px o cubo anda na imagem por mm de X
    def iniciar_calibracao(self):
        if not self.garra.conectado:
            self.mensagem = "Calibrar precisa da garra conectada."
            return
        self.calibrando, self.etapa, self.medidas = True, 0, []
        self.mensagem = "Calibrando: medindo a posiÃ§Ã£o do cubo..."

    def passo_calibracao(self, cubo):
        if not self.imagem_estavel():
            return
        if cubo is None:
            self.parar("CalibraÃ§Ã£o: nÃ£o vejo o cubo.")
            return
        y = self.medir(cubo)
        if y is None:
            return
        if self.etapa == 0:
            self.calib = {"y0": y, "x0": self.x_mm}
            # Anda para o lado que tem espaÃ§o no trilho
            dist = DIST_CALIBRACAO_MM if self.x_mm + DIST_CALIBRACAO_MM <= self.curso_x else -DIST_CALIBRACAO_MM
            self.mover(dist)
            self.etapa = 1
            self.mensagem = f"Calibrando: movendo X {dist:+.0f} mm e vendo o cubo andar..."
        else:
            dx_mm = self.x_mm - self.calib["x0"]
            dpx = y - self.calib["y0"]
            if abs(dx_mm) < 5 or abs(dpx) < 3:
                self.parar("CalibraÃ§Ã£o falhou: o X ou o cubo quase nÃ£o andou.")
                return
            self.cfg["px_por_mm"] = self.px_por_mm_atual = dpx / dx_mm
            salvar_config(self.cfg)
            self.calibrando = False
            self.mensagem = (f"Calibrado: o cubo anda {abs(self.cfg['px_por_mm']):.2f} px por mm de X. "
                             "Seguindo o cubo.")

    def reaprender_px_por_mm(self, y):
        """Perspectiva: perto, 1 mm de X vira MAIS pixels do que longe.
        EntÃ£o, a cada movimento, compara o quanto o X andou com o quanto o cubo andou
        na imagem e ajusta a conversÃ£o aos poucos (mÃ©dia entre o valor velho e o novo)."""
        anterior = self.antes_do_movimento
        self.antes_do_movimento = None
        if anterior is None:
            return
        dx_mm = self.x_mm - anterior[0]
        dpx = y - anterior[1]
        if abs(dx_mm) < 2 or abs(dpx) < 2:
            return                                  # movimento pequeno demais para medir bem
        novo = dpx / dx_mm
        if novo * self.px_por_mm_atual <= 0:
            # O cubo andou na imagem para o lado CONTRÃRIO do esperado.
            # Uma vez pode ser ruÃ­do (ou alguÃ©m mexeu no cubo); duas seguidas = sentido invertido.
            self.contra_sentido += 1
            if self.contra_sentido >= 2:
                self.inverter_sentido("Sentido do X estava invertido: corrigido sozinho.")
                self.px_por_mm_atual = self.cfg["px_por_mm"] = novo
                salvar_config(self.cfg)
            return
        self.contra_sentido = 0
        self.px_por_mm_atual = 0.5 * self.px_por_mm_atual + 0.5 * novo

    def inverter_sentido(self, msg):
        self.cfg["px_por_mm"] = -self.cfg["px_por_mm"]
        self.px_por_mm_atual = -self.px_por_mm_atual
        self.contra_sentido = 0
        salvar_config(self.cfg)
        self.mensagem = msg
        print(msg, flush=True)

    # --- seguir: controle proporcional em malha fechada, o tempo todo
    def passo_seguir(self, cubo):
        if not self.imagem_estavel():
            return
        if cubo is None:
            self.medidas = []
            self.mensagem = "Procurando o cubo..."
            return
        if not self.cfg["px_por_mm"]:
            self.iniciar_calibracao()       # primeira vez: calibra sozinho
            return
        y = self.medir(cubo)
        if y is None:
            return
        self.reaprender_px_por_mm(y)

        erro_px = self.cfg["alvo_y"] - y
        erro_mm = erro_px / self.px_por_mm_atual
        if abs(erro_px) <= self.cfg["tolerancia_px"]:
            self.mensagem = f"NO PONTO!  erro {erro_px:+.1f} px ({erro_mm:+.1f} mm)"
            return

        mm = max(-PASSO_MAX_MM, min(PASSO_MAX_MM, GANHO * erro_mm))
        if abs(mm) < PASSO_MIN_MM:
            mm = PASSO_MIN_MM if mm > 0 else -PASSO_MIN_MM
        # NÃ£o adianta pedir alÃ©m do trilho
        if (self.x_mm <= 0.5 and mm < 0) or (self.x_mm >= self.curso_x - 0.5 and mm > 0):
            self.mensagem = "X no fim do trilho: nÃ£o dÃ¡ para chegar mais no cubo."
            return
        self.antes_do_movimento = (self.x_mm, y)
        self.mover(mm)
        self.mensagem = f"Aproximando: erro {erro_px:+.0f} px ({erro_mm:+.1f} mm) -> X {mm:+.1f} mm"

    # --- desenho
    def desenhar(self, img, cubo):
        h, w = img.shape[:2]
        alvo, limite = int(self.cfg["alvo_y"]), int(self.cfg["limite_y"])
        # Linha limite (vermelha, tracejada) e linha alvo (verde)
        for x in range(0, w, 24):
            cv2.line(img, (x, limite), (x + 12, limite), (60, 60, 230), 2)
        cv2.putText(img, "LIMITE", (w - 90, limite - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 230), 2)
        cv2.line(img, (0, alvo), (w, alvo), (80, 220, 80), 2)
        cv2.putText(img, "ALVO (ponto ideal)", (w - 200, alvo - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (80, 220, 80), 2)

        if cubo:
            cx, cy, contorno = cubo
            cv2.drawContours(img, [contorno], -1, (0, 165, 255), 2)
            cv2.circle(img, (int(cx), int(cy)), 6, (0, 165, 255), -1)
            cv2.arrowedLine(img, (int(cx), int(cy)), (int(cx), alvo), (255, 255, 255), 2, tipLength=0.15)
            erro_px = alvo - cy
            texto = f"erro {erro_px:+.0f} px"
            if self.px_por_mm_atual:
                texto += f" = {erro_px / self.px_por_mm_atual:+.1f} mm de X"
            cv2.putText(img, texto, (int(cx) + 12, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if self.calibrando:
            estado, cor = "CALIBRANDO", (255, 180, 0)
        elif self.seguir:
            estado, cor = "SEGUINDO", (0, 200, 255)
        else:
            estado, cor = "PARADO", (200, 200, 200)
        cv2.rectangle(img, (0, h - 56), (w, h), (20, 20, 20), -1)
        conexao = "garra OK" if self.garra.conectado else "SEM GARRA"
        cv2.putText(img, f"[{estado}]  X = {self.x_mm:.1f} mm   ({conexao})", (10, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
        cv2.putText(img, self.mensagem, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        cv2.putText(img, "A seguir on/off  espaco PARAR  C calibrar  I inverte  J/L X+-10  H home  M mascara  Q sair",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # --- laÃ§o principal
    def rodar(self):
        cam = cv2.VideoCapture(self.cfg["camera"], cv2.CAP_DSHOW)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # imagem sempre a mais nova possÃ­vel
        if not cam.isOpened():
            print(f"NÃ£o abriu a cÃ¢mera {self.cfg['camera']}. Tente --camera 0, 1, 2...", flush=True)
            return

        janela = "Visao da Garra"
        cv2.namedWindow(janela)
        cv2.setMouseCallback(janela, self.mouse)

        fechada_seguidas = 0
        while True:
            ok, bruto = cam.read()
            if not ok:
                print("A cÃ¢mera parou de mandar imagens.", flush=True)
                break
            escala = LARGURA / bruto.shape[1]
            self.frame = cv2.resize(bruto, None, fx=escala, fy=escala)

            self.ler_garra()
            cubo, mascara = achar_cubo(self.frame, self.cfg, self.ultimo_cubo)
            if cubo:
                self.ultimo_cubo = cubo[:2]

            if self.garra.conectado:
                if self.calibrando:
                    self.passo_calibracao(cubo)
                elif self.seguir:
                    self.passo_seguir(cubo)

            tela = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR) if self.mostrar_mascara else self.frame.copy()
            self.desenhar(tela, cubo)
            cv2.imshow(janela, tela)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k != 255:
                self.tecla(k)
            # No Windows essa verificaÃ§Ã£o Ã s vezes dÃ¡ "fechada" num frame sÃ³ por engano:
            # sÃ³ sai se a janela aparecer fechada em vÃ¡rios frames seguidos
            fechada = cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1
            fechada_seguidas = fechada_seguidas + 1 if fechada else 0
            if fechada_seguidas >= 10:
                break

        self.garra.parar()
        cam.release()
        cv2.destroyAllWindows()
        self.garra.desconectar()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Aproxima a garra do cubo laranja usando a cÃ¢mera.")
    p.add_argument("--camera", type=int, default=None, help="Ã­ndice da cÃ¢mera (GoPro costuma ser 2)")
    p.add_argument("--porta", default="COM15", help="porta serial do Arduino")
    p.add_argument("--sem-garra", action="store_true", help="sÃ³ a cÃ¢mera, sem mover a garra")
    Visao(p.parse_args()).rodar()
