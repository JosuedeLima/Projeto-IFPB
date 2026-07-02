"""
consolidar_e_retreinar.py

Consolida todas as imagens dos splits train/valid/test em uma pasta temporária,
refaz o split estratificado usando prepare_dataset_robust e retreina o modelo.
"""
import shutil
import sys
import subprocess
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR.parent / "dataset"
POOL_IMAGES = BASE_DIR.parent / "_pool_images"
POOL_LABELS = BASE_DIR.parent / "_pool_labels"

SPLITS = ["train", "valid", "test"]

def consolidar():
    POOL_IMAGES.mkdir(parents=True, exist_ok=True)
    POOL_LABELS.mkdir(parents=True, exist_ok=True)

    total_imgs = 0
    total_lbls = 0

    for split in SPLITS:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"

        if img_dir.exists():
            for f in img_dir.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    dest = POOL_IMAGES / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)
                        total_imgs += 1

        if lbl_dir.exists():
            for f in lbl_dir.iterdir():
                if f.suffix.lower() == ".txt":
                    dest = POOL_LABELS / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)
                        total_lbls += 1

    print(f"[OK] Pool consolidado: {total_imgs} imagens, {total_lbls} labels")
    return total_imgs

def limpar_splits_antigos():
    """Remove os splits antigos para o prepare_dataset_robust gerar os novos."""
    for split in ["images", "labels"]:
        for sub in ["train", "val"]:
            p = DATASET_DIR / split / sub
            if p.exists():
                shutil.rmtree(p)
                print(f"[LIMPO] {p.relative_to(DATASET_DIR.parent)}")

def main():
    print("\n" + "=" * 60)
    print("  Consolidação e Retreat — Todas as 21 imagens")
    print("=" * 60)

    n = consolidar()
    if n == 0:
        print("[ERRO] Nenhuma imagem nova encontrada nos splits.")
        sys.exit(1)

    limpar_splits_antigos()

    print("\n[INFO] Executando prepare_dataset_robust.py com todas as imagens...")
    result = subprocess.run(
        [
            sys.executable,
            str(BASE_DIR / "prepare_dataset_robust.py"),
            "--images-dir", str(POOL_IMAGES),
            "--labels-dir", str(POOL_LABELS),
            "--dataset-dir", str(DATASET_DIR),
            "--val-ratio", "0.2",
            "--seed", "42",
        ],
        cwd=str(BASE_DIR.parent),
        capture_output=False,
    )

    if result.returncode != 0:
        print("[ERRO] prepare_dataset_robust.py falhou.")
        sys.exit(1)

    # Limpar pool temporário
    shutil.rmtree(POOL_IMAGES, ignore_errors=True)
    shutil.rmtree(POOL_LABELS, ignore_errors=True)
    print("\n[INFO] Pool temporário removido.")

    print("\n[INFO] Iniciando treinamento com o dataset completo...")
    result = subprocess.run(
        [sys.executable, str(BASE_DIR / "train_yolo_abacaxi.py"), "--modo", "treinar"],
        cwd=str(BASE_DIR.parent),
        capture_output=False,
    )

    if result.returncode != 0:
        print("[ERRO] Treinamento falhou.")
        sys.exit(1)

    print("\n[CONCLUÍDO] Modelo retreinado com todas as 21 imagens!")

if __name__ == "__main__":
    main()
