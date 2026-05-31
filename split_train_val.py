import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

# TODO: set your paths here
RGB_DIR = Path(r"D:\multimodel\data\datasets\all\rgb2")
GRAY_DIR = Path(r"D:\multimodel\data\datasets\all\gray")
LABEL_DIR = Path(r"D:\multimodel\data\datasets\all\labels")
OUTPUT_DIR = Path(r"D:\multimodel\data\datasets\all\output")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SHUFFLE = True
SEED = 42

RGB_EXT = ".jpg"
GRAY_EXT = ".jpg"
LABEL_EXT = ".txt"


def list_stems(folder: Path, ext: str) -> List[str]:
    return sorted([p.stem for p in folder.glob(f"*{ext}")])


def ensure_matching(stems: Iterable[str], folder: Path, ext: str) -> None:
    missing = [s for s in stems if not (folder / f"{s}{ext}").exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {folder}: {missing[:10]} (showing up to 10)")


def split_items(items: List[str], train_ratio: float, val_ratio: float) -> Tuple[List[str], List[str], List[str]]:
    if SHUFFLE:
        random.seed(SEED)
        random.shuffle(items)
    train_idx = int(len(items) * train_ratio)
    val_idx = train_idx + int(len(items) * val_ratio)
    return items[:train_idx], items[train_idx:val_idx], items[val_idx:]


def copy_subset(stems: List[str], subset_dir: Path) -> None:
    (subset_dir / "rgb").mkdir(parents=True, exist_ok=True)
    (subset_dir / "gray").mkdir(parents=True, exist_ok=True)
    (subset_dir / "labels").mkdir(parents=True, exist_ok=True)

    for s in stems:
        shutil.copy2(RGB_DIR / f"{s}{RGB_EXT}", subset_dir / "rgb" / f"{s}{RGB_EXT}")
        shutil.copy2(GRAY_DIR / f"{s}{GRAY_EXT}", subset_dir / "gray" / f"{s}{GRAY_EXT}")
        shutil.copy2(LABEL_DIR / f"{s}{LABEL_EXT}", subset_dir / "labels" / f"{s}{LABEL_EXT}")


def main() -> None:
    rgb_stems = list_stems(RGB_DIR, RGB_EXT)
    if not rgb_stems:
        raise RuntimeError(f"No RGB images found in {RGB_DIR}")

    ensure_matching(rgb_stems, GRAY_DIR, GRAY_EXT)
    ensure_matching(rgb_stems, LABEL_DIR, LABEL_EXT)

    train_stems, val_stems, test_stems = split_items(rgb_stems, TRAIN_RATIO, VAL_RATIO)

    copy_subset(train_stems, OUTPUT_DIR / "train")
    copy_subset(val_stems, OUTPUT_DIR / "val")
    copy_subset(test_stems, OUTPUT_DIR / "test")

    print(
        f"Total: {len(rgb_stems)}; Train: {len(train_stems)}; Val: {len(val_stems)}; "
        f"Test: {len(test_stems)}"
    )


if __name__ == "__main__":
    main()
