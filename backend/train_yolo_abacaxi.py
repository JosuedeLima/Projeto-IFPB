"""
Script de Treinamento e Inferência - Detector de Abacaxi para Robô de Adubação
Arquitetura: YOLOv8 (Detecção de Objetos)

Detecta dois objetos:
  - Classe 0: pe_abacaxi   → a planta com folhagem (gatilho para adubar)
  - Classe 1: olho_abacaxi → o fruto no topo da folhagem (alvo da adubação)

Lógica do robô:
  1. Câmera captura frame enquanto caminha pela plantação
  2. Se detectar "pe_abacaxi" E "olho_abacaxi" → move bico e aduba no olho
  3. Se detectar apenas "pe_abacaxi" sem olho visível → registra mas não aduba
  4. Se não detectar nada → continua caminhando

Pré-requisitos:
  pip install ultralytics opencv-python pyyaml numpy Pillow

Uso:
  # Treinar o modelo com seu dataset:
  python train_yolo_abacaxi.py --modo treinar

  # Rodar inferência em tempo real (câmera do robô):
  python train_yolo_abacaxi.py --modo robo

  # Testar em uma imagem específica:
  python train_yolo_abacaxi.py --modo testar --imagem caminho/para/imagem.jpg

Estrutura esperada do dataset (formato YOLOv8):
  dataset/
    images/
      train/   ← imagens de treino (.jpg, .png)
      val/     ← imagens de validação
    labels/
      train/   ← anotações .txt correspondentes
      val/
    data.yaml  ← gerado automaticamente por este script
"""

import os
import sys
import argparse
import yaml
import time
import numpy as np
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================================
# CONFIGURAÇÕES GERAIS
# ============================================================================

BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "dataset"  # dataset/ está na raiz do projeto
MODEL_DIR   = BASE_DIR / "modelo_yolo"
RUNS_DIR    = BASE_DIR / "runs"

# Classes do detector — ordem alinhada com as anotações do Roboflow
# (classe 0 = pe_abacaxi, classe 1 = olho_abacaxi)
CLASSES = {
    0: "pe_abacaxi",    # Planta com folhagem → gatilho para verificar olho
    1: "olho_abacaxi",  # Fruto no topo da folhagem → alvo da adubação
}

# Parâmetros de treinamento
EPOCHS      = 100       # YOLOv8 converge bem com 100 épocas
IMG_SIZE    = 640       # Tamanho padrão do YOLOv8
BATCH_SIZE  = 8         # Ajuste conforme a VRAM disponível
MODEL_BASE  = "yolov8n.pt"  # nano=rápido; trocar por yolov8s.pt para mais precisão

# Parâmetros de inferência do robô
CONF_THRESHOLD = 0.50   # Confiança mínima para aceitar detecção (50%)
IOU_THRESHOLD  = 0.45   # Threshold de IoU para NMS
CAMERA_ID      = 0      # ID da câmera (0 = câmera padrão do sistema)
REQUIRED_CONSECUTIVE_FRAMES = 3  # Frames consecutivos para confirmar adubação
TARGET_STABILITY_RADIUS_PX = 40  # Tolerância de deslocamento do olho entre frames
ADUBACAO_COOLDOWN_FRAMES = 20    # Bloqueio curto para evitar adubar repetidamente


# ============================================================================
# GERAÇÃO DO data.yaml
# ============================================================================

def gerar_data_yaml() -> Path:
    """
    Gera o arquivo data.yaml exigido pelo YOLOv8 para treinamento.
    O YOLOv8 usa esse arquivo para saber onde estão as imagens e as classes.
    """
    yaml_path = DATASET_DIR / "data.yaml"

    # Detecta a estrutura de pastas real (suporta layouts antigos e novos)
    if (DATASET_DIR / "images" / "train").exists() and (DATASET_DIR / "images" / "val").exists():
        train_path = "images/train"
        val_path = "images/val"
    elif (DATASET_DIR / "train" / "images").exists():
        train_path = "train/images"
        if (DATASET_DIR / "valid" / "images").exists():
            val_path = "valid/images"
        elif (DATASET_DIR / "val" / "images").exists():
            val_path = "val/images"
        elif (DATASET_DIR / "images" / "val").exists():
            # Fallback: alguns pipelines geram apenas dataset/images/val
            val_path = "images/val"
        else:
            raise FileNotFoundError(
                "Split de validação não encontrado. Crie 'dataset/valid/images', "
                "'dataset/val/images' ou 'dataset/images/val' para medir generalização real."
            )
    else:
        raise FileNotFoundError(
            "Split de treino não encontrado. Esperado: 'dataset/images/train' "
            "ou 'dataset/train/images'."
        )

    data = {
        "path":  str(DATASET_DIR.resolve()),
        "train": train_path,
        "val":   val_path,
        "nc":    len(CLASSES),
        "names": list(CLASSES.values()),
    }

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    print(f"[OK] data.yaml gerado com sucesso.")
    print(f"     Treino: {train_path}")
    print(f"     Val:    {val_path}")
    return yaml_path


# ============================================================================
# VERIFICAÇÃO DO DATASET
# ============================================================================

def verificar_dataset() -> bool:
    """
    Verifica se o dataset está no formato correto para o YOLOv8.

    Formato esperado:
      dataset/images/train/*.jpg
      dataset/images/val/*.jpg
      dataset/labels/train/*.txt   ← anotações YOLO
      dataset/labels/val/*.txt

    Cada .txt tem uma linha por objeto:
      <classe> <x_centro> <y_centro> <largura> <altura>
    Todos os valores normalizados entre 0 e 1.

    Exemplo de linha para um olho_abacaxi:
      1 0.512 0.234 0.085 0.110
    """
    print("\n[INFO] Verificando estrutura do dataset...")

    # Tenta detectar a estrutura (prefere layout moderno: images/ + labels/)
    train_img_path = None
    val_img_path = None
    pastas_necessarias = []

    if (DATASET_DIR / "images" / "train").exists():
        train_img_path = DATASET_DIR / "images" / "train"
        val_img_path = DATASET_DIR / "images" / "val"
        pastas_necessarias = [
            DATASET_DIR / "images" / "train",
            DATASET_DIR / "images" / "val",
            DATASET_DIR / "labels" / "train",
            DATASET_DIR / "labels" / "val",
        ]
    elif (DATASET_DIR / "train" / "images").exists():
        train_img_path = DATASET_DIR / "train" / "images"
        if (DATASET_DIR / "valid" / "images").exists():
            val_img_path = DATASET_DIR / "valid" / "images"
            pastas_necessarias = [
                DATASET_DIR / "train" / "images",
                DATASET_DIR / "train" / "labels",
                DATASET_DIR / "valid" / "images",
                DATASET_DIR / "valid" / "labels",
            ]
        elif (DATASET_DIR / "val" / "images").exists():
            val_img_path = DATASET_DIR / "val" / "images"
            pastas_necessarias = [
                DATASET_DIR / "train" / "images",
                DATASET_DIR / "train" / "labels",
                DATASET_DIR / "val" / "images",
                DATASET_DIR / "val" / "labels",
            ]
        elif (DATASET_DIR / "images" / "val").exists():
            # Fallback para estrutura híbrida: treino em train/, val em images/
            val_img_path = DATASET_DIR / "images" / "val"
            pastas_necessarias = [
                DATASET_DIR / "train" / "images",
                DATASET_DIR / "train" / "labels",
                DATASET_DIR / "images" / "val",
                DATASET_DIR / "labels" / "val",
            ]
        else:
            print("  [FALTA] Split de validação (valid/, val/ ou images/val)")
            pastas_necessarias = [
                DATASET_DIR / "train" / "images",
                DATASET_DIR / "train" / "labels",
            ]
            ok = False
    else:
        print("  [FALTA] Split de treino (images/train ou train/images)")
        return False

    ok = locals().get("ok", True)
    for pasta in pastas_necessarias:
        if not pasta.exists():
            print(f"  [FALTA] {pasta}")
            ok = False
        else:
            n_arquivos = len(list(pasta.iterdir()))
            print(f"  [OK] {pasta.relative_to(DATASET_DIR.parent)} ({n_arquivos} arquivos)")

    if not ok:
        print("\n[ERRO] Dataset incompleto. Crie as pastas e adicione imagens anotadas.")
        print("       Use Roboflow ou LabelImg para anotar as imagens.")
        print("       Cada imagem precisa de um .txt correspondente com as caixas.")
        return False

    if val_img_path is None:
        print("\n[ERRO] Split de validação não encontrado.")
        print("       Esperado: 'dataset/valid/images', 'dataset/val/images' ou 'dataset/images/val'.")
        return False

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs_train = [p for p in train_img_path.iterdir() if p.suffix.lower() in exts]
    imgs_val   = [p for p in val_img_path.iterdir() if p.suffix.lower() in exts] if val_img_path.exists() else []

    if len(imgs_train) == 0:
        print("\n[ERRO] Nenhuma imagem encontrada em images/train/")
        return False

    print(f"\n  Imagens de treino:    {len(imgs_train)}")
    print(f"  Imagens de validação: {len(imgs_val)}")

    if len(imgs_train) < 20:
        print("\n[AVISO] Poucas imagens de treino. YOLOv8 funciona melhor com 100+ imagens.")
        print("        O Transfer Learning compensará parcialmente, mas colete mais fotos.")

    return True


# ============================================================================
# TREINAMENTO
# ============================================================================

def treinar():
    """
    Treina o modelo YOLOv8 com o dataset de abacaxi.

    O YOLOv8 usa Transfer Learning automaticamente a partir dos pesos
    pré-treinados no COCO dataset (yolov8n.pt), adaptando para nossas
    duas classes: pe_abacaxi e olho_abacaxi.

    Data augmentation aplicado internamente pelo YOLOv8:
      - Mosaic (combina 4 imagens)
      - Flip horizontal/vertical
      - Escala, rotação, translação
      - Ajuste de HSV (matiz, saturação, brilho)
      - MixUp
    """
    print("\n" + "=" * 60)
    print("TREINAMENTO - YOLOv8 Detector de Abacaxi")
    print("=" * 60)

    if not verificar_dataset():
        sys.exit(1)

    try:
        yaml_path = gerar_data_yaml()
    except FileNotFoundError as exc:
        print(f"\n[ERRO] {exc}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERRO] Ultralytics não instalado.")
        print("       Execute: pip install ultralytics")
        sys.exit(1)

    print(f"\n[INFO] Carregando modelo base: {MODEL_BASE}")
    print(f"       (pré-treinado no COCO → Transfer Learning para abacaxi)")
    model = YOLO(MODEL_BASE)

    print(f"\n[INFO] Iniciando treinamento...")
    print(f"       Épocas:   {EPOCHS}")
    print(f"       Imagem:   {IMG_SIZE}x{IMG_SIZE}")
    print(f"       Batch:    {BATCH_SIZE}")
    print(f"       Classes:  {list(CLASSES.values())}")

    start_time = time.time()

    model.train(
        data      = str(yaml_path),
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH_SIZE,
        name      = "abacaxi_detector",
        project   = str(RUNS_DIR),
        patience  = 0,
        workers   = 0,
        save      = True,
        plots     = True,
        verbose   = True,
        # Augmentation ajustada para imagens de plantação
        flipud    = 0.5,    # Flip vertical (útil para vista aérea/drone)
        fliplr    = 0.5,    # Flip horizontal
        degrees   = 45.0,   # Rotação até 45°
        translate = 0.1,
        scale     = 0.5,
        hsv_h     = 0.015,  # Variação de matiz (pequena — folhas são verdes)
        hsv_s     = 0.7,    # Variação de saturação (iluminação do campo)
        hsv_v     = 0.4,    # Variação de brilho (sol, sombra, nublado)
        mosaic    = 1.0,
    )

    end_time = time.time()
    duracao_total = end_time - start_time
    minutos = int(duracao_total // 60)
    segundos = int(duracao_total % 60)

    print("\n" + "=" * 60)
    print(f"TREINAMENTO CONCLUÍDO!")
    print(f"Tempo total: {minutos}m {segundos}s")
    print("=" * 60)

    # Busca dinâmica pela pasta de run mais recente (evita usar pesos desatualizados de runs anteriores)
    candidatos = list(RUNS_DIR.glob("abacaxi_detector*"))
    if candidatos:
        candidatos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        melhor_modelo = candidatos[0] / "weights" / "best.pt"
        print(f"[INFO] Usando pesos da pasta de run mais recente: {candidatos[0].name}")
    else:
        melhor_modelo = RUNS_DIR / "abacaxi_detector" / "weights" / "best.pt"

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    destino = MODEL_DIR / "detector_abacaxi.pt"

    import shutil
    if melhor_modelo.exists():
        shutil.copy(melhor_modelo, destino)
        print(f"\n[SALVO] Melhor modelo salvo em: {destino}")
    else:
        print(f"\n[AVISO] Modelo não encontrado em {melhor_modelo}")

    print("\n[INFO] Gráficos salvos em: runs/abacaxi_detector/")
    print("       (loss, precision, recall, mAP — abra results.png)")
    return destino


# ============================================================================
# LÓGICA DO ROBÔ — INFERÊNCIA EM TEMPO REAL
# ============================================================================

class RoboAdubador:
    """
    Controlador de inferência para o robô de adubação.

    Fluxo por frame:
      1. Captura frame da câmera
      2. Roda detecção YOLOv8
      3. Analisa detecções:
         - Pe E Olho detectados  → calcula posição e aduba
         - Apenas Pe detectado   → planta jovem/ângulo ruim, registra
         - Nada detectado        → continua caminhando
    """

    def __init__(self, caminho_modelo: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            print("[ERRO] Execute: pip install ultralytics")
            sys.exit(1)

        print(f"\n[INFO] Carregando modelo: {caminho_modelo}")
        self.model = YOLO(caminho_modelo)
        self.classes = CLASSES

        self.total_frames       = 0
        self.total_adubacoes    = 0
        self.total_pes_sem_olho = 0
        self.frames_confirmacao_olho = 0
        self.ultimo_centro_olho = None
        self.cooldown_frames_restantes = 0

        print("[OK] Robô pronto para operar.")

    def detectar_frame(self, frame, conf: float | None = None, imgsz: int | None = None) -> dict:
        """
        Executa apenas a detecção YOLO (sem lógica de adubação).
        Usado pelo frontend web para validação de acurácia em tempo real.
        """
        conf = CONF_THRESHOLD if conf is None else conf
        kwargs = {"conf": conf, "iou": IOU_THRESHOLD, "verbose": False}
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        resultados = self.model(frame, **kwargs)

        deteccoes = []
        for resultado in resultados:
            boxes = resultado.boxes
            if boxes is None:
                continue
            for box in boxes:
                classe_id = int(box.cls[0])
                confianca = float(box.conf[0])
                coords = box.xyxy[0].tolist()
                cx = (coords[0] + coords[2]) / 2
                cy = (coords[1] + coords[3]) / 2
                deteccoes.append({
                    "classe":    self.classes[classe_id],
                    "classe_id": classe_id,
                    "confianca": confianca,
                    "bbox":      coords,
                    "centro":    (cx, cy),
                })

        altura, largura = frame.shape[:2]
        return {
            "deteccoes": deteccoes,
            "largura":   largura,
            "altura":    altura,
        }

    def analisar_frame(self, frame) -> dict:
        """
        Analisa um frame e retorna a decisão de adubação.

        Retorna dict com:
          - acao:          "adubar" | "aguardar" | "caminhar"
          - posicao_olho:  (x, y) em pixels ou None
          - deteccoes:     lista de objetos detectados
          - frame_anotado: frame com caixas desenhadas
          - mensagem:      descrição textual da decisão
        """
        resultados = self.model(
            frame,
            conf    = CONF_THRESHOLD,
            iou     = IOU_THRESHOLD,
            verbose = False,
        )

        deteccoes = []
        pes       = []
        olhos     = []

        for resultado in resultados:
            boxes = resultado.boxes
            if boxes is None:
                continue

            for box in boxes:
                classe_id = int(box.cls[0])
                confianca = float(box.conf[0])
                coords    = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                cx = (coords[0] + coords[2]) / 2
                cy = (coords[1] + coords[3]) / 2

                det = {
                    "classe":    self.classes[classe_id],
                    "classe_id": classe_id,
                    "confianca": confianca,
                    "bbox":      coords,
                    "centro":    (cx, cy),
                }
                deteccoes.append(det)

                if classe_id == 1:   # pe_abacaxi
                    pes.append(det)
                elif classe_id == 0:  # olho_abacaxi
                    olhos.append(det)

        frame_anotado = resultados[0].plot() if resultados else frame

        # ── LÓGICA DE DECISÃO ──────────────────────────────────────────────
        if pes and olhos:
            # Encontrou pé E olho → aduba no olho de maior confiança
            melhor_olho  = max(olhos, key=lambda d: d["confianca"])
            posicao_olho = melhor_olho["centro"]

            if self.cooldown_frames_restantes > 0:
                self.cooldown_frames_restantes -= 1
                self.frames_confirmacao_olho = 0
                self.ultimo_centro_olho = posicao_olho
                return {
                    "acao":          "aguardar",
                    "posicao_olho":  None,
                    "deteccoes":     deteccoes,
                    "frame_anotado": frame_anotado,
                    "mensagem":      f"Alvo detectado, aguardando cooldown "
                                     f"({self.cooldown_frames_restantes} frame(s)).",
                }

            if self.ultimo_centro_olho is None:
                self.frames_confirmacao_olho = 1
            else:
                dx = posicao_olho[0] - self.ultimo_centro_olho[0]
                dy = posicao_olho[1] - self.ultimo_centro_olho[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist <= TARGET_STABILITY_RADIUS_PX:
                    self.frames_confirmacao_olho += 1
                else:
                    self.frames_confirmacao_olho = 1

            self.ultimo_centro_olho = posicao_olho

            if self.frames_confirmacao_olho < REQUIRED_CONSECUTIVE_FRAMES:
                return {
                    "acao":          "aguardar",
                    "posicao_olho":  None,
                    "deteccoes":     deteccoes,
                    "frame_anotado": frame_anotado,
                    "mensagem":      f"Confirmando alvo ({self.frames_confirmacao_olho}/"
                                     f"{REQUIRED_CONSECUTIVE_FRAMES} frames).",
                }

            return {
                "acao":          "adubar",
                "posicao_olho":  posicao_olho,
                "deteccoes":     deteccoes,
                "frame_anotado": frame_anotado,
                "mensagem":      f"Olho em ({posicao_olho[0]:.0f}, {posicao_olho[1]:.0f})px "
                                 f"(conf: {melhor_olho['confianca']:.0%})",
            }

        elif pes and not olhos:
            # Pé encontrado, olho não visível (planta jovem ou ângulo ruim)
            self.frames_confirmacao_olho = 0
            self.ultimo_centro_olho = None
            return {
                "acao":          "aguardar",
                "posicao_olho":  None,
                "deteccoes":     deteccoes,
                "frame_anotado": frame_anotado,
                "mensagem":      f"{len(pes)} pé(s) detectado(s), olho não visível.",
            }

        else:
            # Nenhum pé → continua caminhando
            self.frames_confirmacao_olho = 0
            self.ultimo_centro_olho = None
            return {
                "acao":          "caminhar",
                "posicao_olho":  None,
                "deteccoes":     deteccoes,
                "frame_anotado": frame_anotado,
                "mensagem":      "Nenhum pé de abacaxi detectado.",
            }

    def executar_acao(self, decisao: dict):
        """
        Executa a ação física do robô.
        Integre com ROS2, GPIO, etc. nos stubs abaixo.
        """
        acao = decisao["acao"]

        if acao == "adubar":
            pos = decisao["posicao_olho"]
            print(f"  [ADUBAR]   Bico → ({pos[0]:.0f}, {pos[1]:.0f})px | {decisao['mensagem']}")
            self._mover_bico(pos)
            self._acionar_dosador()
            self.total_adubacoes += 1
            self.cooldown_frames_restantes = ADUBACAO_COOLDOWN_FRAMES
            self.frames_confirmacao_olho = 0

        elif acao == "aguardar":
            print(f"  [AGUARDAR] {decisao['mensagem']}")
            if "olho não visível" in decisao["mensagem"]:
                self.total_pes_sem_olho += 1

        else:
            print(f"  [CAMINHAR] {decisao['mensagem']}")

    def _mover_bico(self, posicao: tuple):
        """
        Stub: integrar com servomotor/braço robótico via ROS2 / GPIO.
        Converter posicao (x_pixel, y_pixel) para coordenadas físicas
        usando homografia ou câmera de profundidade.

        Exemplo futuro (ROS2):
            from geometry_msgs.msg import Point
            pub.publish(Point(x=posicao[0], y=posicao[1], z=0))
        """
        pass

    def _acionar_dosador(self):
        """
        Stub: integrar com bomba/válvula via GPIO (Raspberry Pi / Arduino).

        Exemplo futuro (RPi):
            import RPi.GPIO as GPIO
            GPIO.output(PINO_DOSADOR, GPIO.HIGH)
            time.sleep(TEMPO_DOSAGEM)
            GPIO.output(PINO_DOSADOR, GPIO.LOW)
        """
        time.sleep(0.1)  # Simulação do tempo de adubação

    def relatorio_sessao(self):
        print("\n" + "=" * 50)
        print("RELATÓRIO DA SESSÃO")
        print("=" * 50)
        print(f"  Frames analisados:       {self.total_frames}")
        print(f"  Adubações realizadas:    {self.total_adubacoes}")
        print(f"  Pés sem olho detectado:  {self.total_pes_sem_olho}")
        print("=" * 50)


# ============================================================================
# MODO ROBÔ — CÂMERA EM TEMPO REAL
# ============================================================================

def modo_robo(caminho_modelo: str):
    """
    Roda o robô em loop contínuo lendo frames da câmera.
    Pressione 'q' para encerrar.
    """
    import cv2

    robo = RoboAdubador(caminho_modelo)

    print(f"\n[INFO] Abrindo câmera (ID={CAMERA_ID})...")
    cap = cv2.VideoCapture(CAMERA_ID)

    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a câmera ID={CAMERA_ID}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

    print("[INFO] Robô em operação. Pressione 'q' para encerrar.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[AVISO] Frame inválido, tentando novamente...")
                continue

            robo.total_frames += 1
            decisao = robo.analisar_frame(frame)

            if robo.total_frames % 10 == 0:
                print(f"Frame {robo.total_frames:05d} | {decisao['acao'].upper()} "
                      f"| {decisao['mensagem']}")

            robo.executar_acao(decisao)

            try:
                import cv2 as _cv2
                _cv2.imshow("Robô Adubador - Abacaxi", decisao["frame_anotado"])
                if _cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except Exception:
                pass  # Ambiente headless (ex: Raspberry Pi sem display)

    except KeyboardInterrupt:
        print("\n[INFO] Encerrado pelo usuário.")
    finally:
        cap.release()
        try:
            import cv2 as _cv2
            _cv2.destroyAllWindows()
        except Exception:
            pass
        robo.relatorio_sessao()


# ============================================================================
# MODO TESTE — IMAGEM ESTÁTICA
# ============================================================================

def modo_testar(caminho_modelo: str, caminho_imagem: str):
    """Testa o modelo em uma imagem estática e salva o resultado anotado."""
    import cv2

    if not os.path.exists(caminho_imagem):
        print(f"[ERRO] Imagem não encontrada: {caminho_imagem}")
        sys.exit(1)

    robo = RoboAdubador(caminho_modelo)
    frame = cv2.imread(caminho_imagem)

    print(f"\n[INFO] Analisando: {caminho_imagem}")
    decisao = robo.analisar_frame(frame)

    print(f"\n  Ação decidida : {decisao['acao'].upper()}")
    print(f"  {decisao['mensagem']}")
    print(f"\n  Detecções ({len(decisao['deteccoes'])}):")
    for det in decisao["deteccoes"]:
        print(f"    - {det['classe']:<15} | conf: {det['confianca']:.0%} "
              f"| centro: ({det['centro'][0]:.0f}, {det['centro'][1]:.0f})px")

    saida = Path(caminho_imagem).stem + "_resultado.jpg"
    cv2.imwrite(saida, decisao["frame_anotado"])
    print(f"\n[SALVO] Imagem anotada em: {saida}")


# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 - Detector de Abacaxi para Robô de Adubação (IFPB)"
    )
    parser.add_argument(
        "--modo",
        choices=["treinar", "robo", "testar"],
        required=True,
        help="treinar: treina o modelo | robo: tempo real | testar: imagem estática",
    )
    parser.add_argument(
        "--modelo",
        default=str(MODEL_DIR / "detector_abacaxi.pt"),
        help="Caminho para o modelo treinado (.pt)",
    )
    parser.add_argument(
        "--imagem",
        default=None,
        help="Caminho da imagem (apenas para --modo testar)",
    )
    args = parser.parse_args()

    if args.modo == "treinar":
        treinar()

    elif args.modo == "robo":
        if not os.path.exists(args.modelo):
            print(f"[ERRO] Modelo não encontrado: {args.modelo}")
            print("       Execute: python train_yolo_abacaxi.py --modo treinar")
            sys.exit(1)
        modo_robo(args.modelo)

    elif args.modo == "testar":
        if args.imagem is None:
            print("[ERRO] Informe a imagem com --imagem caminho/para/foto.jpg")
            sys.exit(1)
        if not os.path.exists(args.modelo):
            print(f"[ERRO] Modelo não encontrado: {args.modelo}")
            print("       Execute: python train_yolo_abacaxi.py --modo treinar")
            sys.exit(1)
        modo_testar(args.modelo, args.imagem)


if __name__ == "__main__":
    main()
