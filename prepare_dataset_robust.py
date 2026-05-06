"""
prepare_dataset_robust.py

Pipeline para preparar dataset YOLO de forma robusta:
1) Limpa/converte labels inválidos (mantém apenas bbox: 5 valores por linha).
2) Cria split train/val estratificado por perfil de classes da imagem.
3) Gera relatório de qualidade antes do treino.

Uso comum:
  python prepare_dataset_robust.py --from-images

Quando já existem labels em dataset/labels_raw:
  python prepare_dataset_robust.py --images-dir images --labels-dir dataset/labels_raw
"""

import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


CLASSES = {0: "pe_abacaxi", 1: "olho_abacaxi"}
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_COORD = -1e-3
MAX_COORD = 1 + 1e-3


@dataclass
class LabelStats:
    total_files: int = 0
    files_with_errors: int = 0
    total_lines: int = 0
    kept_lines: int = 0
    dropped_lines: int = 0
    images_without_label: int = 0
    images_empty_after_clean: int = 0
    class_pe: int = 0
    class_olho: int = 0


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {images_dir}")
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS])


def _parse_bbox_line(line: str) -> tuple[bool, int | None, str | None]:
    parts = line.strip().split()
    if len(parts) != 5:
        return False, None, "numero_de_campos_invalido"

    try:
        cid = int(parts[0])
        coords = [float(v) for v in parts[1:]]
    except ValueError:
        return False, None, "valor_nao_numerico"

    if cid not in CLASSES:
        return False, None, "classe_invalida"

    for c in coords:
        if not (MIN_COORD <= c <= MAX_COORD):
            return False, None, "coordenada_fora_de_intervalo"

    return True, cid, None


def clean_labels_for_images(images: list[Path], labels_dir: Path, cleaned_labels_dir: Path) -> tuple[dict[str, list[int]], LabelStats]:
    cleaned_labels_dir.mkdir(parents=True, exist_ok=True)
    stats = LabelStats(total_files=len(images))
    image_classes: dict[str, list[int]] = {}

    for img in images:
        raw_txt = labels_dir / f"{img.stem}.txt"
        dst_txt = cleaned_labels_dir / f"{img.stem}.txt"

        if not raw_txt.exists():
            stats.images_without_label += 1
            dst_txt.write_text("", encoding="utf-8")
            image_classes[img.name] = []
            continue

        lines = raw_txt.read_text(encoding="utf-8").splitlines()
        stats.total_lines += len(lines)

        cleaned: list[str] = []
        local_classes: list[int] = []
        file_had_error = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            ok, cid, _ = _parse_bbox_line(stripped)
            if not ok:
                stats.dropped_lines += 1
                file_had_error = True
                continue

            cleaned.append(stripped)
            local_classes.append(cid if cid is not None else -1)
            stats.kept_lines += 1
            if cid == 0:
                stats.class_pe += 1
            elif cid == 1:
                stats.class_olho += 1

        if file_had_error:
            stats.files_with_errors += 1

        if len(cleaned) == 0:
            stats.images_empty_after_clean += 1

        dst_txt.write_text("\n".join(cleaned), encoding="utf-8")
        image_classes[img.name] = local_classes

    return image_classes, stats


def _bucket_for_image(class_ids: list[int]) -> str:
    uniq = set(class_ids)
    if uniq == {0, 1}:
        return "pe_e_olho"
    if uniq == {0}:
        return "so_pe"
    if uniq == {1}:
        return "so_olho"
    return "sem_objetos"


def stratified_split(images: list[Path], image_classes: dict[str, list[int]], val_ratio: float, seed: int) -> tuple[list[Path], list[Path], dict[str, int]]:
    random.seed(seed)
    groups: dict[str, list[Path]] = defaultdict(list)

    for img in images:
        bucket = _bucket_for_image(image_classes.get(img.name, []))
        groups[bucket].append(img)

    train: list[Path] = []
    val: list[Path] = []
    bucket_counts = {k: len(v) for k, v in groups.items()}

    for _, group_imgs in groups.items():
        random.shuffle(group_imgs)
        n_val = int(round(len(group_imgs) * val_ratio))
        if len(group_imgs) >= 2 and n_val == 0:
            n_val = 1
        if n_val >= len(group_imgs) and len(group_imgs) > 1:
            n_val = len(group_imgs) - 1
        val.extend(group_imgs[:n_val])
        train.extend(group_imgs[n_val:])

    if not train and val:
        train.append(val.pop())
    if not val and len(train) > 1:
        val.append(train.pop())

    random.shuffle(train)
    random.shuffle(val)
    return train, val, bucket_counts


def copy_split(images: list[Path], cleaned_labels_dir: Path, dst_images: Path, dst_labels: Path) -> None:
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    for img in images:
        shutil.copy2(img, dst_images / img.name)
        src_txt = cleaned_labels_dir / f"{img.stem}.txt"
        if src_txt.exists():
            shutil.copy2(src_txt, dst_labels / src_txt.name)
        else:
            (dst_labels / f"{img.stem}.txt").write_text("", encoding="utf-8")


def write_data_yaml(dataset_dir: Path) -> None:
    data_yaml = dataset_dir / "data.yaml"
    content = (
        "names:\n"
        "- pe_abacaxi\n"
        "- olho_abacaxi\n"
        "nc: 2\n"
        f"path: {str(dataset_dir.resolve())}\n"
        "train: images/train\n"
        "val: images/val\n"
    )
    data_yaml.write_text(content, encoding="utf-8")


def quality_report(
    stats: LabelStats,
    train: list[Path],
    val: list[Path],
    image_classes: dict[str, list[int]],
    bucket_counts: dict[str, int],
) -> str:
    train_counter = Counter(_bucket_for_image(image_classes.get(img.name, [])) for img in train)
    val_counter = Counter(_bucket_for_image(image_classes.get(img.name, [])) for img in val)

    report_lines = [
        "",
        "=" * 70,
        "RELATORIO DE QUALIDADE DO DATASET",
        "=" * 70,
        f"Arquivos de label processados: {stats.total_files}",
        f"Arquivos com erros de formato: {stats.files_with_errors}",
        f"Linhas totais de anotacao: {stats.total_lines}",
        f"Linhas mantidas: {stats.kept_lines}",
        f"Linhas descartadas: {stats.dropped_lines}",
        f"Imagens sem label: {stats.images_without_label}",
        f"Imagens sem objetos apos limpeza: {stats.images_empty_after_clean}",
        "",
        "Contagem de objetos (apos limpeza):",
        f"  - pe_abacaxi (0): {stats.class_pe}",
        f"  - olho_abacaxi (1): {stats.class_olho}",
        "",
        "Perfis de imagem no dataset:",
        f"  - pe_e_olho: {bucket_counts.get('pe_e_olho', 0)}",
        f"  - so_pe: {bucket_counts.get('so_pe', 0)}",
        f"  - so_olho: {bucket_counts.get('so_olho', 0)}",
        f"  - sem_objetos: {bucket_counts.get('sem_objetos', 0)}",
        "",
        f"Split final -> train: {len(train)} | val: {len(val)}",
        "Distribuicao train por perfil:",
        f"  - pe_e_olho: {train_counter.get('pe_e_olho', 0)}",
        f"  - so_pe: {train_counter.get('so_pe', 0)}",
        f"  - so_olho: {train_counter.get('so_olho', 0)}",
        f"  - sem_objetos: {train_counter.get('sem_objetos', 0)}",
        "Distribuicao val por perfil:",
        f"  - pe_e_olho: {val_counter.get('pe_e_olho', 0)}",
        f"  - so_pe: {val_counter.get('so_pe', 0)}",
        f"  - so_olho: {val_counter.get('so_olho', 0)}",
        f"  - sem_objetos: {val_counter.get('sem_objetos', 0)}",
        "=" * 70,
        "",
    ]
    return "\n".join(report_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara dataset YOLO robusto para projeto de adubacao.")
    parser.add_argument("--images-dir", default="images", help="Pasta de imagens brutas.")
    parser.add_argument("--labels-dir", default="dataset/labels_raw", help="Pasta de labels de origem (.txt).")
    parser.add_argument("--dataset-dir", default="dataset", help="Pasta de saida do dataset YOLO.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Proporcao para validacao.")
    parser.add_argument("--seed", type=int, default=42, help="Seed do split.")
    parser.add_argument(
        "--from-images",
        action="store_true",
        help="Se informado, cria labels vazios para imagens sem labels de origem.",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    dataset_dir = Path(args.dataset_dir)

    images = list_images(images_dir)
    if not images:
        raise RuntimeError(f"Nenhuma imagem encontrada em: {images_dir}")

    if not labels_dir.exists():
        if not args.from_images:
            print(
                "[AVISO] Pasta de labels de origem nao encontrada: "
                f"{labels_dir}. Prosseguindo com labels vazios (equivalente a --from-images)."
            )
        labels_dir.mkdir(parents=True, exist_ok=True)

    cleaned_labels_dir = dataset_dir / "labels_cleaned"
    image_classes, stats = clean_labels_for_images(images, labels_dir, cleaned_labels_dir)

    train, val, bucket_counts = stratified_split(
        images=images,
        image_classes=image_classes,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    out_train_images = dataset_dir / "images" / "train"
    out_val_images = dataset_dir / "images" / "val"
    out_train_labels = dataset_dir / "labels" / "train"
    out_val_labels = dataset_dir / "labels" / "val"

    for p in [out_train_images, out_val_images, out_train_labels, out_val_labels]:
        if p.exists():
            shutil.rmtree(p)

    copy_split(train, cleaned_labels_dir, out_train_images, out_train_labels)
    copy_split(val, cleaned_labels_dir, out_val_images, out_val_labels)
    write_data_yaml(dataset_dir)

    report = quality_report(stats, train, val, image_classes, bucket_counts)
    print(report)
    (dataset_dir / "quality_report.txt").write_text(report, encoding="utf-8")

    print("[OK] Dataset preparado com sucesso.")
    print(f"     data.yaml: {dataset_dir / 'data.yaml'}")
    print(f"     relatorio: {dataset_dir / 'quality_report.txt'}")


if __name__ == "__main__":
    main()
