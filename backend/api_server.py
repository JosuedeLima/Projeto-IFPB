"""
Servidor web para validação do modelo em tempo real.

Expõe API REST e serve o frontend estático.
A câmera roda no navegador (PC ou celular); os frames são enviados
para inferência YOLO e as caixas (pe_abacaxi / olho_abacaxi) retornam em JSON.

Uso:
  python api_server.py
  python api_server.py --modelo modelo_yolo/detector_abacaxi.pt --porta 8000

Acesse: http://localhost:8000
Para testar no celular na mesma rede: http://<IP-do-PC>:8000
"""

from __future__ import annotations

import argparse
import csv
import socket
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
MODEL_DIR = BASE_DIR / "modelo_yolo"
DEFAULT_MODEL = MODEL_DIR / "detector_abacaxi.pt"
IMAGES_DIR = BASE_DIR / "images"

# Validador web usa limiar mais baixo que o robô (modelo em treino costuma ter conf baixa)
WEB_CONF_DEFAULT = 0.005

from train_yolo_abacaxi import CLASSES, CONF_THRESHOLD, IMG_SIZE, RoboAdubador  # noqa: E402

robo: RoboAdubador | None = None
modelo_carregado: str | None = None
diagnostico_modelo: dict = {}
metricas_treino: dict = {}


def _avaliar_modelo(detector: RoboAdubador) -> dict:
    """Testa o modelo em uma imagem local para estimar qualidade."""
    candidatos = []
    if IMAGES_DIR.exists():
        candidatos = sorted(
            list(IMAGES_DIR.glob("*.jpg"))
            + list(IMAGES_DIR.glob("*.png"))
            + list(IMAGES_DIR.glob("*.jpeg"))
        )
    if not candidatos:
        return {"testado": False, "motivo": "Nenhuma imagem em images/ para teste."}

    frame = cv2.imread(str(candidatos[0]))
    if frame is None:
        return {"testado": False, "motivo": f"Falha ao ler {candidatos[0].name}."}

    resultado = detector.detectar_frame(frame, conf=0.001)
    if not resultado["deteccoes"]:
        return {
            "testado": True,
            "imagem": candidatos[0].name,
            "max_confianca": 0.0,
            "deteccoes_acima_50": 0,
            "alerta": "Modelo não detectou nada nem com confiança 0.1%. Retreine o modelo.",
        }

    max_conf = max(d["confianca"] for d in resultado["deteccoes"])
    acima_50 = sum(1 for d in resultado["deteccoes"] if d["confianca"] >= 0.5)
    alerta = None
    if max_conf < 0.25:
        alerta = (
            f"Confiança máxima apenas {max_conf:.1%} em {candidatos[0].name}. "
            "Use o slider baixo no frontend ou retreine com mais imagens."
        )
    return {
        "testado": True,
        "imagem": candidatos[0].name,
        "max_confianca": round(max_conf, 4),
        "deteccoes_acima_50": acima_50,
        "alerta": alerta,
    }


def _extrair_metricas_treino(caminho_modelo: Path) -> dict:
    """Lê results.csv do run correspondente ao modelo e retorna as métricas reais."""
    # O modelo fica em runs/<nome_run>/weights/best.pt ou last.pt
    # results.csv está em runs/<nome_run>/results.csv
    runs_dir = caminho_modelo.parent.parent  # weights/ -> run_dir
    results_csv = runs_dir / "results.csv"
    args_yaml = runs_dir / "args.yaml"

    metricas: dict = {
        "disponivel": False,
        "run": runs_dir.name,
    }

    # Lê total de imagens de treino e validação do data.yaml
    data_yaml_path: Path | None = None
    if args_yaml.exists():
        with open(args_yaml, encoding="utf-8") as f:
            for line in f:
                if line.startswith("data:"):
                    data_yaml_path = Path(line.split(":", 1)[1].strip())
                    break

    if data_yaml_path and data_yaml_path.exists():
        dataset_base = data_yaml_path.parent
        train_imgs = dataset_base / "train" / "images"
        val_imgs = dataset_base / "valid" / "images"
        test_imgs = dataset_base / "test" / "images"
        n_train = len(list(train_imgs.glob("*.*"))) if train_imgs.exists() else 0
        n_val = len(list(val_imgs.glob("*.*"))) if val_imgs.exists() else 0
        n_test = len(list(test_imgs.glob("*.*"))) if test_imgs.exists() else 0
        metricas["dataset"] = {
            "total": n_train + n_val + n_test,
            "treino": n_train,
            "validacao": n_val,
            "teste": n_test,
        }

    if not results_csv.exists():
        metricas["motivo"] = "results.csv não encontrado."
        return metricas

    # Lê CSV e encontra a época com melhor mAP@50
    melhor: dict | None = None
    total_epocas = 0
    with open(results_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Remove espaços dos cabeçalhos
        reader.fieldnames = [h.strip() for h in (reader.fieldnames or [])]
        for row in reader:
            total_epocas += 1
            try:
                map50 = float(row.get("metrics/mAP50(B)", 0) or 0)
                if melhor is None or map50 > melhor["map50"]:
                    melhor = {
                        "epoca": int(float(row.get("epoch", 0))),
                        "map50": round(map50, 4),
                        "map50_95": round(float(row.get("metrics/mAP50-95(B)", 0) or 0), 4),
                        "precisao": round(float(row.get("metrics/precision(B)", 0) or 0), 4),
                        "recall": round(float(row.get("metrics/recall(B)", 0) or 0), 4),
                        "val_box_loss": round(float(row.get("val/box_loss", 0) or 0), 4),
                        "val_cls_loss": round(float(row.get("val/cls_loss", 0) or 0), 4),
                    }
            except (ValueError, TypeError):
                continue

    if melhor:
        metricas["disponivel"] = True
        metricas["total_epocas"] = total_epocas
        metricas["melhor_epoca"] = melhor

    return metricas


def carregar_modelo(caminho: Path) -> None:
    global robo, modelo_carregado, diagnostico_modelo, metricas_treino
    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {caminho}\n"
            "Execute: python train_yolo_abacaxi.py --modo treinar"
        )
    robo = RoboAdubador(str(caminho))
    modelo_carregado = str(caminho.resolve())
    diagnostico_modelo = _avaliar_modelo(robo)
    metricas_treino = _extrair_metricas_treino(caminho)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Garante modelo carregado mesmo com `uvicorn api_server:app`."""
    global robo
    if robo is None and DEFAULT_MODEL.exists():
        carregar_modelo(DEFAULT_MODEL)
    yield


app = FastAPI(
    title="Validador Abacaxi — IFPB",
    description="Detecção em tempo real para validação do modelo YOLOv8",
    version="1.0.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {
        "ok": robo is not None,
        "modelo": modelo_carregado,
        "classes": list(CLASSES.values()),
        "conf_robo": CONF_THRESHOLD,
        "conf_web_padrao": WEB_CONF_DEFAULT,
        "diagnostico": diagnostico_modelo,
        "metricas_treino": metricas_treino,
    }


@app.post("/api/detect")
async def detect(
    imagem: UploadFile = File(...),
    conf: float = Query(default=WEB_CONF_DEFAULT, ge=0.001, le=0.99),
):
    if robo is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    raw = await imagem.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Imagem vazia.")

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Formato de imagem inválido.")

    t0 = time.perf_counter()
    resultado = robo.detectar_frame(frame, conf=conf, imgsz=IMG_SIZE)
    tempo_ms = (time.perf_counter() - t0) * 1000

    # Monta lista completa
    todas = []
    for det in resultado["deteccoes"]:
        todas.append({
            "classe": det["classe"],
            "classe_id": det["classe_id"],
            "confianca": det["confianca"],
            "bbox": det["bbox"],
            "centro": list(det["centro"]),
        })

    # Mantém apenas a detecção de maior confiança por classe (máx. 1 por classe = 2 boxes)
    melhor_por_classe: dict = {}
    for det in todas:
        cid = det["classe_id"]
        if cid not in melhor_por_classe or det["confianca"] > melhor_por_classe[cid]["confianca"]:
            melhor_por_classe[cid] = det
    deteccoes = list(melhor_por_classe.values())

    return {
        "deteccoes": deteccoes,
        "largura": resultado["largura"],
        "altura": resultado["altura"],
        "tempo_ms": round(tempo_ms, 1),
        "conf_usada": conf,
    }


@app.get("/")
def index():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend não encontrado.")
    return FileResponse(index_path)


if FRONTEND_DIR.exists():
    # html=False: nunca lista diretórios nem expõe estrutura de pastas
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=False), name="static")


def porta_disponivel(host: str, porta: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, porta))
            return True
        except OSError:
            return False


def main():
    parser = argparse.ArgumentParser(description="Servidor web — validação do detector de abacaxi")
    parser.add_argument("--modelo", default=str(DEFAULT_MODEL), help="Caminho do .pt treinado")
    parser.add_argument("--host", default="0.0.0.0", help="Host (0.0.0.0 para acesso na rede local)")
    parser.add_argument("--porta", type=int, default=8000, help="Porta HTTP")
    args = parser.parse_args()

    try:
        carregar_modelo(Path(args.modelo))
    except FileNotFoundError as exc:
        print(f"[ERRO] {exc}")
        sys.exit(1)

    if not porta_disponivel(args.host, args.porta):
        print(f"\n[ERRO] Porta {args.porta} já está em uso.")
        print("       Feche a instância anterior do servidor (Ctrl+C no terminal)")
        print("       ou inicie em outra porta: python api_server.py --porta 8001")
        sys.exit(1)

    import uvicorn

    print("\n" + "=" * 56)
    print("  Validador web — Detector de Abacaxi (IFPB)")
    print("=" * 56)
    print(f"  Modelo : {modelo_carregado}")
    print(f"  Local  : http://127.0.0.1:{args.porta}")
    print(f"  Rede   : http://<seu-ip>:{args.porta}  (celular na mesma Wi-Fi)")
    if diagnostico_modelo.get("alerta"):
        print(f"\n  [AVISO] {diagnostico_modelo['alerta']}")
    elif diagnostico_modelo.get("testado"):
        print(
            f"\n  [OK] Teste rápido em {diagnostico_modelo.get('imagem')}: "
            f"conf. máx. {diagnostico_modelo.get('max_confianca', 0):.1%}"
        )
    print("=" * 56 + "\n")

    uvicorn.run(app, host=args.host, port=args.porta, log_level="info")


if __name__ == "__main__":
    main()
