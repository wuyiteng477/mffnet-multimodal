#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
该脚本用于双相机 RGB + 灰度/荧光 TIFF 图像的数据整理。由于两个相机不是严格同步拍摄，但拍摄顺序一致，因此脚本分别按文件修改时间从早到晚排序，再按顺序一一配对，并同步重命名，方便后续图像配准和多模态目标检测训练。
"""

import csv
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np


RGB_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
GRAY_EXTS = {".tif", ".tiff"}


# ====== 用户配置区（直接修改这里即可） ======
RGB_INPUT_DIR = r"D:\test2\rgb_input"
GRAY_INPUT_DIR = r"D:\test2\tif_input"
RGB_OUTPUT_DIR = r"D:\test2\rgb_output"
GRAY_OUTPUT_DIR = r"D:\test2\tif_output"
START_INDEX = 1
DIGITS = 6
JPEG_QUALITY = 95
OVERWRITE = False
DRY_RUN = False
# ==========================================


def collect_image_files(input_dir: Path, exts: set) -> List[Path]:
    files = []
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def sort_by_mtime(files: List[Path]) -> List[Path]:
    return sorted(files, key=lambda p: (p.stat().st_mtime, p.name))


def check_output_dir(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    if any(output_dir.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"Output directory is not empty and overwrite is False: {output_dir}"
            )


def convert_and_save_rgb_as_jpg(
    src_path: Path, dst_path: Path, jpeg_quality: int
) -> bool:
    try:
        data = np.fromfile(str(src_path), dtype=np.uint8)
    except Exception as exc:
        print(f"[ERROR] Failed to read RGB bytes: {src_path} ({exc})")
        return False

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] Failed to read RGB image: {src_path}")
        return False

    ok = cv2.imwrite(
        str(dst_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    if not ok:
        print(f"[ERROR] Failed to write RGB JPG: {dst_path}")
        return False

    return True


def copy_and_rename_gray_tiff(src_path: Path, dst_path: Path) -> bool:
    try:
        shutil.copy2(src_path, dst_path)
    except Exception as exc:
        print(f"[ERROR] Failed to copy Gray TIFF: {src_path} -> {dst_path} ({exc})")
        return False

    return True


def write_pairing_log(csv_path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "index",
        "rgb_original_path",
        "gray_original_path",
        "rgb_output_path",
        "gray_output_path",
        "rgb_original_name",
        "gray_original_name",
        "new_rgb_name",
        "new_gray_name",
        "rgb_mtime",
        "gray_mtime",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(sep=" ")


def main() -> None:
    rgb_input_dir = Path(RGB_INPUT_DIR)
    gray_input_dir = Path(GRAY_INPUT_DIR)
    rgb_output_dir = Path(RGB_OUTPUT_DIR)
    gray_output_dir = Path(GRAY_OUTPUT_DIR)

    if not rgb_input_dir.exists() or not rgb_input_dir.is_dir():
        raise RuntimeError(f"RGB input directory not found: {rgb_input_dir}")
    if not gray_input_dir.exists() or not gray_input_dir.is_dir():
        raise RuntimeError(f"Gray input directory not found: {gray_input_dir}")

    check_output_dir(rgb_output_dir, OVERWRITE)
    check_output_dir(gray_output_dir, OVERWRITE)

    rgb_files = sort_by_mtime(collect_image_files(rgb_input_dir, RGB_EXTS))
    gray_files = sort_by_mtime(collect_image_files(gray_input_dir, GRAY_EXTS))

    print(f"RGB image count: {len(rgb_files)}")
    print(f"Gray TIFF count: {len(gray_files)}")

    pair_count = min(len(rgb_files), len(gray_files))
    if len(rgb_files) != len(gray_files):
        print(
            "[WARNING] RGB/Gray counts differ. Only the smaller count will be processed."
        )

    print(f"Pair count to process: {pair_count}")

    log_rows: List[dict] = []
    current_index = START_INDEX

    for i in range(pair_count):
        rgb_path = rgb_files[i]
        gray_path = gray_files[i]

        base_name = f"{current_index:0{DIGITS}d}"
        new_rgb_name = f"{base_name}.jpg"
        new_gray_name = f"{base_name}.tif"

        rgb_out = rgb_output_dir / new_rgb_name
        gray_out = gray_output_dir / new_gray_name

        rgb_mtime = _format_mtime(rgb_path)
        gray_mtime = _format_mtime(gray_path)

        print(f"RGB:  {rgb_path.name} -> {new_rgb_name}")
        print(f"Gray: {gray_path.name} -> {new_gray_name}")
        print(f"RGB mtime:  {rgb_mtime}")
        print(f"Gray mtime: {gray_mtime}")

        success = True
        if not DRY_RUN:
            success = convert_and_save_rgb_as_jpg(rgb_path, rgb_out, JPEG_QUALITY)
            if success:
                success = copy_and_rename_gray_tiff(gray_path, gray_out)

        if not success:
            print("[WARNING] Skipped this pair due to error.")
            continue

        log_rows.append(
            {
                "index": current_index,
                "rgb_original_path": str(rgb_path),
                "gray_original_path": str(gray_path),
                "rgb_output_path": str(rgb_out),
                "gray_output_path": str(gray_out),
                "rgb_original_name": rgb_path.name,
                "gray_original_name": gray_path.name,
                "new_rgb_name": new_rgb_name,
                "new_gray_name": new_gray_name,
                "rgb_mtime": rgb_mtime,
                "gray_mtime": gray_mtime,
            }
        )

        current_index += 1

    log_path = Path("pairing_log.csv")
    write_pairing_log(log_path, log_rows)
    print(f"Pairing log saved: {log_path}")


if __name__ == "__main__":
    main()
