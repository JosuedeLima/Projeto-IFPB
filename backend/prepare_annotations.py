"""
prepare_annotations.py
Valida se as anotações YOLOv8 (.txt) estão corretas antes do treinamento.

FORMATO YOLO TXT:
    Cada imagem tem um arquivo .txt correspondente com uma linha por objeto:
        <classe_id> <x_centro> <y_centro> <largura> <altura>

    Todos os valores normalizados entre 0 e 1.

    Classes:
        0 → pe_abacaxi   (folhagem da planta)
        1 → olho_abacaxi (fruto no topo)

    Exemplo (pe + olho na mesma imagem):
        0 0.500 0.600 0.800 0.700
        1 0.510 0.210 0.150 0.180

USO:
    python prepare_annotations.py
"""

import os
import sys
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR    = Path(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = BASE_DIR / "dataset"

CLASSES = {0: "pe_abacaxi", 1: "olho_abacaxi"}
EXTENSOES_IMG = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def validar_txt(txt_path: Path) -> tuple[bool, list[str]]:
    """
    Valida um arquivo .txt de anotação YOLO.
    Retorna (ok, lista_de_erros).
    """
    erros = []
    linhas = txt_path.read_text(encoding='utf-8').strip().splitlines()

    if not linhas:
        return True, []  # Arquivo vazio = imagem sem objetos (válido)

    for i, linha in enumerate(linhas, start=1):
        partes = linha.strip().split()
        
        # Este projeto usa detecção por bbox: exatamente 5 valores por linha.
        if len(partes) != 5:
            erros.append(
                f"Linha {i}: formato inválido (encontrado {len(partes)} valores, esperado 5). "
                "Exporte labels no formato YOLO de detecção (bbox)."
            )
            continue

        try:
            classe_id = int(partes[0])
            coords = [float(v) for v in partes[1:]]
        except ValueError:
            erros.append(f"Linha {i}: valor não numérico → '{linha}'")
            continue

        if classe_id not in CLASSES:
            erros.append(f"Linha {i}: classe_id={classe_id} inválida (use 0 ou 1)")

        for v in coords:
            # Pequena tolerância numérica para exportadores.
            if not (-1e-3 <= v <= 1 + 1e-3):
                erros.append(f"Linha {i}: valor {v:.4f} fora do intervalo [0,1]")

    return len(erros) == 0, erros


def validar_split(split: str) -> dict:
    """Valida todas as anotações de um split e retorna estatísticas."""
    # Tenta dois formatos comuns: dataset/split/images ou dataset/images/split
    img_dir = DATASET_DIR / split / "images"
    lbl_dir = DATASET_DIR / split / "labels"
    
    if not img_dir.exists():
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split

    print(f"\n{'='*55}")
    print(f"  Split: [{split.upper()}]")
    print(f"{'='*55}")

    stats = {"pe": 0, "olho": 0, "vazios": 0,
             "erros": 0, "imgs_sem_label": 0, "total_imgs": 0}

    if not img_dir.exists():
        print(f"  [FALTA] {img_dir}")
        return stats
    if not lbl_dir.exists():
        print(f"  [FALTA] {lbl_dir}")
        return stats

    imagens = sorted([p for p in img_dir.iterdir()
                      if p.suffix.lower() in EXTENSOES_IMG])
    stats["total_imgs"] = len(imagens)

    print(f"  Imagens encontradas: {len(imagens)}")

    for img_path in imagens:
        txt_path = lbl_dir / (img_path.stem + ".txt")

        if not txt_path.exists():
            print(f"  [SEM LABEL] {img_path.name}")
            stats["imgs_sem_label"] += 1
            continue

        ok, erros = validar_txt(txt_path)

        if not ok:
            stats["erros"] += 1
            print(f"  [ERRO] {img_path.name}:")
            for e in erros:
                print(f"         → {e}")
            continue

        # Contar objetos por classe
        linhas = txt_path.read_text().strip().splitlines()
        if not linhas:
            stats["vazios"] += 1
        else:
            for linha in linhas:
                partes = linha.strip().split()
                if partes:
                    cid = int(partes[0])
                    if cid == 0:
                        stats["pe"] += 1
                    elif cid == 1:
                        stats["olho"] += 1

    # Resumo do split
    anotados = stats["total_imgs"] - stats["imgs_sem_label"] - stats["erros"]
    print(f"\n  ✔ Anotados corretamente : {anotados}")
    print(f"  ✘ Com erro              : {stats['erros']}")
    print(f"  ○ Sem arquivo .txt      : {stats['imgs_sem_label']}")
    print(f"  ○ .txt vazio (sem obj)  : {stats['vazios']}")
    print(f"\n  Objetos detectáveis:")
    print(f"     {CLASSES[0]:<12} (0): {stats['pe']:>4} bbox(es)")
    print(f"     {CLASSES[1]:<12} (1): {stats['olho']:>4} bbox(es)")

    if stats["pe"] == 0:
        print("\n  [ATENÇÃO] Nenhum 'pe_abacaxi' anotado! "
              "O robô nunca ativará a adubagem.")
    if stats["olho"] == 0:
        print("\n  [ATENÇÃO] Nenhum 'olho_abacaxi' anotado! "
              "O robô não saberá onde adubar.")

    return stats


def main():
    print("Validador de Anotações YOLOv8 - Projeto Abacaxi IFPB")
    print("-" * 55)
    print(f"  Classes: 0=pe_abacaxi | 1=olho_abacaxi")

    stats_train = validar_split("train")
    stats_val   = validar_split("val")

    total_erros = stats_train["erros"] + stats_val["erros"]
    sem_label   = stats_train["imgs_sem_label"] + stats_val["imgs_sem_label"]

    print(f"\n{'='*55}")
    if total_erros == 0 and sem_label == 0:
        print("  [OK] Dataset válido! Execute:")
        print("       python train_yolo_abacaxi.py --modo treinar")
    else:
        print("  [PENDENTE] Corrija os problemas acima.")
        if sem_label:
            print(f"  → {sem_label} imagem(ns) sem arquivo .txt de anotação")
        if total_erros:
            print(f"  → {total_erros} arquivo(s) com formato inválido")
        print("\n  Dica: Use Roboflow (https://roboflow.com) para anotar")
        print("  e exporte em formato 'YOLOv8' para gerar os .txt corretos.")
    print("=" * 55)


if __name__ == "__main__":
    main()
