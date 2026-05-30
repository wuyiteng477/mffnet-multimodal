"""
This script converts the RGB->fluorescence registration workflow from a notebook into
an executable Python program. It detects chessboard corners, estimates a homography
from RGB to fluorescence coordinates, and optionally saves registered RGB images and
visualizations. Fluorescence images are expected to be 3-channel JPG files.

Example:
  python final_peizhun1007_jpg.py \
    --rgb_dir /mnt/SSD/work/tomato/paired/images_rgb \
    --fluo_dir /mnt/SSD/work/tomato/paired/images_gray_jpg \
    --output_dir /mnt/SSD/work/tomato/paired/registration_result \
    --registered_rgb_dir /mnt/SSD/work/tomato/paired/images_rgb_registered \
    --vis_dir /mnt/SSD/work/tomato/paired/registration_visualization \
    --pattern_size 9,6 \
    --save_overlay \
    --save_visual_registered_rgb \
    --overwrite
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# Edit these defaults to run without CLI arguments.
DEFAULT_MODE = "full"
DEFAULT_CALIB_RGB_DIR = r"D:\multimodel\data\CALIB_RGB\jpg"
DEFAULT_CALIB_FLUO_DIR = r"D:\multimodel\data\CALIB_GRAY"
DEFAULT_APPLY_RGB_DIR = r"D:\multimodel\data\apply_rgbtest"
DEFAULT_APPLY_FLUO_DIR = r"D:\multimodel\data\apply_graytest"
DEFAULT_OUTPUT_DIR = r"D:\multimodel\data\APPLY_RGB\apply"
DEFAULT_HOMOGRAPHY_PATH = r"D:\multimodel\data\APPLY_RGB\apply\homography_rgb_to_fluo.npy"
DEFAULT_REGISTERED_RGB_DIR = None
DEFAULT_VIS_DIR = None
DEFAULT_PATTERN_SIZE = (9, 6)
DEFAULT_RANSAC_THRESH = 5.0
DEFAULT_JPEG_QUALITY = 95
DEFAULT_OVERWRITE = True
DEFAULT_SAVE_OVERLAY = True
DEFAULT_SAVE_REGISTERED_RGB = True
DEFAULT_SAVE_VISUAL_REGISTERED_RGB = True
DEFAULT_DRY_RUN = False


@dataclass
class CornerLogRow:
    filename: str
    rgb_path: str
    fluo_path: str
    rgb_corner_found: bool
    fluo_corner_found: bool
    used_for_homography: bool
    message: str


@dataclass
class RegistrationLogRow:
    filename: str
    rgb_path: str
    fluo_path: str
    registered_rgb_train_path: str
    registered_rgb_visual_path: str
    overlay_path: str
    status: str
    message: str


def parse_pattern_size(value: str) -> Tuple[int, int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("pattern_size must be like 9,6")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pattern_size must be two integers") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Register RGB images to fluorescence JPG images using chessboard corners "
            "and a homography."
        )
    )
    parser.add_argument("--mode", choices=["calibrate", "apply", "full"], default=DEFAULT_MODE)
    parser.add_argument(
        "--calib_rgb_dir",
        "--rgb_dir",
        dest="calib_rgb_dir",
        default=DEFAULT_CALIB_RGB_DIR,
        help="Chessboard RGB folder (.jpg)",
    )
    parser.add_argument(
        "--calib_fluo_dir",
        "--fluo_dir",
        dest="calib_fluo_dir",
        default=DEFAULT_CALIB_FLUO_DIR,
        help="Chessboard fluorescence folder (.jpg)",
    )
    parser.add_argument(
        "--apply_rgb_dir",
        default=DEFAULT_APPLY_RGB_DIR,
        help="Non-chessboard RGB folder (.jpg)",
    )
    parser.add_argument(
        "--apply_fluo_dir",
        default=DEFAULT_APPLY_FLUO_DIR,
        help="Non-chessboard fluorescence folder (.jpg)",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output root folder")
    parser.add_argument("--homography_path", default=DEFAULT_HOMOGRAPHY_PATH)
    parser.add_argument("--registered_rgb_dir", default=DEFAULT_REGISTERED_RGB_DIR, help="Registered RGB output folder")
    parser.add_argument("--vis_dir", default=DEFAULT_VIS_DIR, help="Visualization output folder")
    parser.add_argument("--pattern_size", type=parse_pattern_size, default=DEFAULT_PATTERN_SIZE)
    parser.add_argument("--ransac_thresh", type=float, default=DEFAULT_RANSAC_THRESH)
    parser.add_argument("--jpeg_quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--overwrite", action="store_true", default=DEFAULT_OVERWRITE)
    parser.add_argument("--save_overlay", action="store_true", default=DEFAULT_SAVE_OVERLAY)
    parser.add_argument(
        "--save_registered_rgb",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_REGISTERED_RGB,
        help="Save registered RGB images for training",
    )
    parser.add_argument(
        "--save_visual_registered_rgb",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SAVE_VISUAL_REGISTERED_RGB,
        help="Save visualization RGB images with a prefix",
    )
    parser.add_argument("--dry_run", action="store_true", default=DEFAULT_DRY_RUN)
    return parser.parse_args()


def list_jpg_files(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.jpg")))


def check_dir_empty_or_create(path: str, overwrite: bool) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return

    if not overwrite:
        has_content = any(os.scandir(path))
        if has_content:
            raise RuntimeError(f"Directory not empty: {path}. Use --overwrite to proceed.")


def collect_pairs(rgb_dir: str, fluo_dir: str) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    rgb_files = list_jpg_files(rgb_dir)
    pairs: List[Tuple[str, str, str]] = []
    missing: List[str] = []

    for rgb_path in rgb_files:
        filename = os.path.basename(rgb_path)
        stem = os.path.splitext(filename)[0]
        fluo_path = os.path.join(fluo_dir, f"{stem}.jpg")
        if not os.path.exists(fluo_path):
            missing.append(filename)
            continue
        pairs.append((filename, rgb_path, fluo_path))

    return pairs, missing


def find_corners(
    image_path: str,
    pattern_size: Tuple[int, int],
) -> Tuple[bool, Optional[np.ndarray], str]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return False, None, "read failed"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        return False, None, "corner not found"

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined, ""


def estimate_homography(
    rgb_points: List[np.ndarray],
    fluo_points: List[np.ndarray],
    ransac_thresh: float,
) -> np.ndarray:
    all_rgb = np.vstack(rgb_points).reshape(-1, 2)
    all_fluo = np.vstack(fluo_points).reshape(-1, 2)
    H, mask = cv2.findHomography(all_rgb, all_fluo, cv2.RANSAC, ransac_thresh)
    if H is None:
        raise RuntimeError("Homography estimation failed.")
    if mask is not None:
        inlier_ratio = float(np.sum(mask)) / float(len(mask))
        print(f"Homography inlier ratio: {inlier_ratio:.2%}")
    return H


def save_homography(output_dir: str, H: np.ndarray) -> Tuple[str, str]:
    npy_path = os.path.join(output_dir, "homography_rgb_to_fluo.npy")
    txt_path = os.path.join(output_dir, "homography_rgb_to_fluo.txt")
    np.save(npy_path, H)
    np.savetxt(txt_path, H, fmt="%.8f")
    return npy_path, txt_path


def save_jpg(path: str, image: np.ndarray, jpeg_quality: int) -> bool:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    return bool(cv2.imwrite(path, image, params))


def write_corner_log(path: str, rows: List[CornerLogRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "rgb_path",
                "fluo_path",
                "rgb_corner_found",
                "fluo_corner_found",
                "used_for_homography",
                "message",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.filename,
                    row.rgb_path,
                    row.fluo_path,
                    row.rgb_corner_found,
                    row.fluo_corner_found,
                    row.used_for_homography,
                    row.message,
                ]
            )


def write_registration_log(path: str, rows: List[RegistrationLogRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "rgb_path",
                "fluo_path",
                "registered_rgb_train_path",
                "registered_rgb_visual_path",
                "overlay_path",
                "status",
                "message",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.filename,
                    row.rgb_path,
                    row.fluo_path,
                    row.registered_rgb_train_path,
                    row.registered_rgb_visual_path,
                    row.overlay_path,
                    row.status,
                    row.message,
                ]
            )


def load_homography(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise RuntimeError(f"Homography file not found: {path}")
    return np.load(path)


def register_pairs(
    pairs: List[Tuple[str, str, str]],
    H: np.ndarray,
    registered_rgb_dir: str,
    vis_dir: str,
    save_registered_rgb: bool,
    save_visual_registered_rgb: bool,
    save_overlay: bool,
    jpeg_quality: int,
    dry_run: bool,
) -> Tuple[List[RegistrationLogRow], int, int]:
    rows: List[RegistrationLogRow] = []
    saved = 0
    skipped = 0

    for filename, rgb_path, fluo_path in pairs:
        if dry_run:
            rows.append(
                RegistrationLogRow(
                    filename=filename,
                    rgb_path=rgb_path,
                    fluo_path=fluo_path,
                    registered_rgb_train_path="",
                    registered_rgb_visual_path="",
                    overlay_path="",
                    status="dry_run",
                    message="dry_run enabled",
                )
            )
            continue

        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        fluo_img = cv2.imread(fluo_path, cv2.IMREAD_COLOR)
        if rgb_img is None or fluo_img is None:
            rows.append(
                RegistrationLogRow(
                    filename=filename,
                    rgb_path=rgb_path,
                    fluo_path=fluo_path,
                    registered_rgb_train_path="",
                    registered_rgb_visual_path="",
                    overlay_path="",
                    status="failed",
                    message="read failed",
                )
            )
            skipped += 1
            continue

        registered_rgb = cv2.warpPerspective(rgb_img, H, (fluo_img.shape[1], fluo_img.shape[0]))

        train_path = ""
        vis_rgb_path = ""
        overlay_path = ""
        status = "success"
        message = ""

        if save_registered_rgb:
            train_path = os.path.join(registered_rgb_dir, filename)
            if not save_jpg(train_path, registered_rgb, jpeg_quality):
                status = "failed"
                message = "failed to save registered RGB"

        if save_visual_registered_rgb:
            vis_rgb_path = os.path.join(vis_dir, f"registered_rgb_{filename}")
            if not save_jpg(vis_rgb_path, registered_rgb, jpeg_quality):
                status = "failed"
                message = "failed to save visual registered RGB"

        if save_overlay:
            overlay = cv2.addWeighted(registered_rgb, 0.7, fluo_img, 0.3, 0)
            overlay_path = os.path.join(vis_dir, f"overlay_{filename}")
            if not save_jpg(overlay_path, overlay, jpeg_quality):
                status = "failed"
                message = "failed to save overlay"

        if status == "success":
            saved += 1
        else:
            skipped += 1

        rows.append(
            RegistrationLogRow(
                filename=filename,
                rgb_path=rgb_path,
                fluo_path=fluo_path,
                registered_rgb_train_path=train_path,
                registered_rgb_visual_path=vis_rgb_path,
                overlay_path=overlay_path,
                status=status,
                message=message,
            )
        )

    return rows, saved, skipped


def main() -> None:
    args = parse_args()

    registered_rgb_dir = args.registered_rgb_dir or os.path.join(args.output_dir, "registered_rgb_for_train")
    vis_dir = args.vis_dir or os.path.join(args.output_dir, "visualization")

    check_dir_empty_or_create(args.output_dir, args.overwrite)
    check_dir_empty_or_create(registered_rgb_dir, args.overwrite)
    check_dir_empty_or_create(vis_dir, args.overwrite)

    H: Optional[np.ndarray] = None
    homography_npy = ""
    homography_txt = ""
    corner_rows: List[CornerLogRow] = []
    corner_pairs = 0

    if args.mode in ("calibrate", "full"):
        if not os.path.isdir(args.calib_rgb_dir):
            raise RuntimeError(f"Chessboard RGB directory not found: {args.calib_rgb_dir}")
        if not os.path.isdir(args.calib_fluo_dir):
            raise RuntimeError(f"Chessboard fluorescence directory not found: {args.calib_fluo_dir}")

        calib_rgb_files = list_jpg_files(args.calib_rgb_dir)
        calib_fluo_files = list_jpg_files(args.calib_fluo_dir)
        if not calib_rgb_files:
            raise RuntimeError("No chessboard RGB JPG files found.")
        if not calib_fluo_files:
            raise RuntimeError("No chessboard fluorescence JPG files found.")

        calib_pairs, missing = collect_pairs(args.calib_rgb_dir, args.calib_fluo_dir)
        for name in missing:
            print(f"Warning: missing chessboard fluorescence image for {name}")

        rgb_points: List[np.ndarray] = []
        fluo_points: List[np.ndarray] = []

        for filename, rgb_path, fluo_path in calib_pairs:
            rgb_ok, rgb_corners, rgb_msg = find_corners(rgb_path, args.pattern_size)
            fluo_ok, fluo_corners, fluo_msg = find_corners(fluo_path, args.pattern_size)
            used = bool(rgb_ok and fluo_ok)
            message = ""
            if not rgb_ok:
                message = rgb_msg
            if not fluo_ok:
                message = f"{message}; {fluo_msg}".strip("; ")

            if used:
                rgb_points.append(rgb_corners)
                fluo_points.append(fluo_corners)

            corner_rows.append(
                CornerLogRow(
                    filename=filename,
                    rgb_path=rgb_path,
                    fluo_path=fluo_path,
                    rgb_corner_found=rgb_ok,
                    fluo_corner_found=fluo_ok,
                    used_for_homography=used,
                    message=message,
                )
            )

        if not rgb_points:
            raise RuntimeError("No valid corner pairs found. Check pattern_size and image quality.")

        H = estimate_homography(rgb_points, fluo_points, args.ransac_thresh)
        print("Homography H (RGB -> Fluorescence):")
        print(H)

        homography_npy, homography_txt = save_homography(args.output_dir, H)
        corner_pairs = len(rgb_points)

        corner_log_path = os.path.join(args.output_dir, "corner_detection_log.csv")
        write_corner_log(corner_log_path, corner_rows)

    if args.mode in ("apply", "full"):
        if not os.path.isdir(args.apply_rgb_dir):
            raise RuntimeError(f"Apply RGB directory not found: {args.apply_rgb_dir}")
        if not os.path.isdir(args.apply_fluo_dir):
            raise RuntimeError(f"Apply fluorescence directory not found: {args.apply_fluo_dir}")

        apply_rgb_files = list_jpg_files(args.apply_rgb_dir)
        apply_fluo_files = list_jpg_files(args.apply_fluo_dir)
        if not apply_rgb_files:
            raise RuntimeError("No apply RGB JPG files found.")
        if not apply_fluo_files:
            raise RuntimeError("No apply fluorescence JPG files found.")

        if H is None:
            if not args.homography_path:
                raise RuntimeError("--homography_path is required in apply mode.")
            H = load_homography(args.homography_path)
            homography_npy = args.homography_path

        apply_pairs, missing = collect_pairs(args.apply_rgb_dir, args.apply_fluo_dir)
        for name in missing:
            print(f"Warning: missing apply fluorescence image for {name}")

        registration_rows, saved, skipped = register_pairs(
            apply_pairs,
            H,
            registered_rgb_dir,
            vis_dir,
            args.save_registered_rgb,
            args.save_visual_registered_rgb,
            args.save_overlay,
            args.jpeg_quality,
            args.dry_run,
        )

        registration_log_path = os.path.join(args.output_dir, "registration_log.csv")
        write_registration_log(registration_log_path, registration_rows)

        print("\nSummary:")
        print(f"Apply RGB images: {len(apply_rgb_files)}")
        print(f"Apply fluorescence images: {len(apply_fluo_files)}")
        print(f"Paired images: {len(apply_pairs)}")
        print(f"Saved pairs: {saved}")
        print(f"Skipped pairs: {skipped}")
        if homography_npy:
            print(f"Homography path: {homography_npy}")
        if homography_txt:
            print(f"Homography text: {homography_txt}")
        print(f"Registered RGB dir: {registered_rgb_dir}")
        print(f"Visualization dir: {vis_dir}")
        print(f"Registration log: {registration_log_path}")

    if args.mode == "calibrate":
        print("\nSummary:")
        print(f"Corner success pairs: {corner_pairs}")
        if homography_npy:
            print(f"Homography saved: {homography_npy}")
        if homography_txt:
            print(f"Homography text: {homography_txt}")


if __name__ == "__main__":
    main()
