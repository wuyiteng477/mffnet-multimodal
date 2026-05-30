"""
This script batch-converts synchronized, renamed grayscale/fluorescence PNG images
into 3-channel JPG images for downstream image registration and multimodal detection
training. The original PNG files are not modified.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

# Fill these paths before running. Use absolute paths.
DEFAULT_INPUT_PNG_DIR = r"D:\multimodel\data\APPLY_RGB\png"
DEFAULT_OUTPUT_JPG_DIR = r"D:\multimodel\data\APPLY_RGB\jpg"


@dataclass
class ConvertLogRow:
    original_path: str
    output_path: str
    original_dtype: str
    original_shape: str
    output_dtype: str
    output_shape: str
    status: str
    message: str


def collect_png_files(input_png_dir: str) -> List[str]:
    exts = (".png",)
    files = []
    for name in os.listdir(input_png_dir):
        if name.lower().endswith(exts):
            files.append(os.path.join(input_png_dir, name))
    files.sort()
    return files


def check_output_dir(output_jpg_dir: str, overwrite: bool) -> None:
    if not os.path.exists(output_jpg_dir):
        os.makedirs(output_jpg_dir, exist_ok=True)
        return

    if not overwrite:
        existing = [f for f in os.listdir(output_jpg_dir) if os.path.isfile(os.path.join(output_jpg_dir, f))]
        if existing:
            raise RuntimeError(
                f"Output directory is not empty: {output_jpg_dir}. Set --overwrite to proceed."
            )


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img

    if img.dtype == np.uint16 or img.dtype == np.int16:
        normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    if np.issubdtype(img.dtype, np.integer) or np.issubdtype(img.dtype, np.floating):
        normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    raise ValueError(f"Unsupported dtype: {img.dtype}")


def ensure_three_channels(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3:
        if img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            return img
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported shape for channel conversion: {img.shape}")


def save_as_jpg(output_path: str, img: np.ndarray, jpeg_quality: int) -> bool:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    return bool(cv2.imwrite(output_path, img, params))


def write_convert_log(log_path: str, rows: List[ConvertLogRow]) -> None:
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "original_path",
                "output_path",
                "original_dtype",
                "original_shape",
                "output_dtype",
                "output_shape",
                "status",
                "message",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.original_path,
                    row.output_path,
                    row.original_dtype,
                    row.original_shape,
                    row.output_dtype,
                    row.output_shape,
                    row.status,
                    row.message,
                ]
            )


def convert_one(
    input_path: str,
    output_path: str,
    jpeg_quality: int,
    overwrite: bool,
) -> Tuple[bool, ConvertLogRow]:
    if os.path.exists(output_path) and not overwrite:
        row = ConvertLogRow(
            original_path=input_path,
            output_path=output_path,
            original_dtype="",
            original_shape="",
            output_dtype="",
            output_shape="",
            status="skipped",
            message="output exists and overwrite is False",
        )
        return False, row

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        row = ConvertLogRow(
            original_path=input_path,
            output_path=output_path,
            original_dtype="",
            original_shape="",
            output_dtype="",
            output_shape="",
            status="failed",
            message="read failed",
        )
        return False, row

    original_dtype = str(img.dtype)
    original_shape = str(img.shape)

    try:
        img_uint8 = convert_to_uint8(img)
        img_3ch = ensure_three_channels(img_uint8)
    except Exception as exc:
        row = ConvertLogRow(
            original_path=input_path,
            output_path=output_path,
            original_dtype=original_dtype,
            original_shape=original_shape,
            output_dtype="",
            output_shape="",
            status="failed",
            message=str(exc),
        )
        return False, row

    if not (img_3ch.ndim == 3 and img_3ch.shape[2] == 3 and img_3ch.dtype == np.uint8):
        row = ConvertLogRow(
            original_path=input_path,
            output_path=output_path,
            original_dtype=original_dtype,
            original_shape=original_shape,
            output_dtype=str(img_3ch.dtype),
            output_shape=str(img_3ch.shape),
            status="failed",
            message="output validation failed",
        )
        return False, row

    ok = save_as_jpg(output_path, img_3ch, jpeg_quality)
    if not ok:
        row = ConvertLogRow(
            original_path=input_path,
            output_path=output_path,
            original_dtype=original_dtype,
            original_shape=original_shape,
            output_dtype=str(img_3ch.dtype),
            output_shape=str(img_3ch.shape),
            status="failed",
            message="save failed",
        )
        return False, row

    row = ConvertLogRow(
        original_path=input_path,
        output_path=output_path,
        original_dtype=original_dtype,
        original_shape=original_shape,
        output_dtype=str(img_3ch.dtype),
        output_shape=str(img_3ch.shape),
        status="success",
        message="",
    )
    return True, row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert renamed grayscale/fluorescence PNG images to 3-channel JPG. "
            "Original PNG files are not modified."
        )
    )
    parser.add_argument("--input_png_dir", default=DEFAULT_INPUT_PNG_DIR)
    parser.add_argument("--output_jpg_dir", default=DEFAULT_OUTPUT_JPG_DIR)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    check_output_dir(args.output_jpg_dir, args.overwrite)

    png_files = collect_png_files(args.input_png_dir)
    print(f"Found {len(png_files)} PNG files.")

    success_count = 0
    fail_count = 0
    rows: List[ConvertLogRow] = []

    for input_path in png_files:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_name = f"{base_name}.jpg"
        output_path = os.path.join(args.output_jpg_dir, output_name)

        ok, row = convert_one(
            input_path=input_path,
            output_path=output_path,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
        )
        rows.append(row)

        print(
            " | ".join(
                [
                    f"src={os.path.basename(input_path)}",
                    f"dst={os.path.basename(output_path)}",
                    f"src_dtype={row.original_dtype}",
                    f"src_shape={row.original_shape}",
                    f"dst_dtype={row.output_dtype}",
                    f"dst_shape={row.output_shape}",
                    f"status={row.status}",
                    f"msg={row.message}",
                ]
            )
        )

        if ok:
            success_count += 1
        else:
            fail_count += 1

    log_path = os.path.join(args.output_jpg_dir, "convert_log.csv")
    write_convert_log(log_path, rows)

    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
