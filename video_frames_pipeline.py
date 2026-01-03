"""
video_frames_pipeline.py

Usage (step-by-step):

1) Extract N frames (evenly spaced across the video):
   python video_frames_pipeline.py extract \
       --video input.mp4 \
       --out_dir frames_raw \
       --num 120

   (Run your own processing on frames_raw -> frames_processed)

2) Assemble processed frames into a video:
   python video_frames_pipeline.py assemble \
       --frames_dir frames_processed \
       --out output.mp4 \
       --fps 30
   # Tip: if you want to match the original video's FPS automatically:
   # python video_frames_pipeline.py assemble --frames_dir frames_processed --out output.mp4 --ref_video input.mp4
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional

import cv2


def _natural_key(p: Path) -> Tuple:
    """
    Sort helper: extracts a numeric chunk from filenames like frame_000123.png
    so they sort in capture order.
    """
    s = p.stem
    nums = re.findall(r"\d+", s)
    return (int(nums[-1]) if nums else -1, s)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_video_meta(video_path: Path) -> Tuple[int, float, Tuple[int, int]]:
    """
    Returns (frame_count, fps, (width, height)).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Some containers don't report frame count reliably; we still try.
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    cap.release()
    return frame_count, fps, (width, height)


def _compute_sample_indices(total_frames: int, num_samples: int) -> List[int]:
    """
    Compute exactly num_samples indices covering [0, total_frames-1] as evenly as possible.
    If total_frames == 0, raises. If num_samples <= 0, raises.
    Caps num_samples to total_frames if needed.
    """
    if total_frames <= 0:
        raise ValueError("Video appears to have zero frames (or metadata missing).")
    if num_samples <= 0:
        raise ValueError("num_samples must be >= 1")

    num_samples = min(num_samples, total_frames)

    if num_samples == 1:
        return [total_frames // 2]

    # Even spacing with integer rounding, guarantees first=0 and last=total_frames-1
    indices = [round(i * (total_frames - 1) / (num_samples - 1)) for i in range(num_samples)]
    # Remove any duplicates while preserving order
    seen = set()
    uniq = []
    for idx in indices:
        if idx not in seen:
            uniq.append(idx)
            seen.add(idx)
    return uniq


def extract_frames(video_path: Path, out_dir: Path, num_frames: int, prefix: str = "frame") -> List[Path]:
    """
    Extracts `num_frames` evenly spaced frames from `video_path` to `out_dir`.
    Filenames are zero-padded PNGs like frame_000001.png.
    Returns a list of saved file paths in chronological order.
    """
    _ensure_dir(out_dir)

    # Get total frames; fall back to scanning if metadata is unreliable
    meta_frame_count, _, _ = _get_video_meta(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # If reported frame_count is 0, count by reading.
    if meta_frame_count == 0:
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        meta_frame_count = count

    target_indices = _compute_sample_indices(meta_frame_count, num_frames)
    target_set = set(target_indices)

    saved_paths: List[Path] = []
    frame_idx = 0
    write_ordinal = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in target_set:
            # Zero-pad to 6 digits for stable sorting
            filename = f"{prefix}_{write_ordinal:06d}.png"
            out_path = out_dir / filename
            ok2 = cv2.imwrite(str(out_path), frame)
            if not ok2:
                cap.release()
                raise RuntimeError(f"Failed to write image: {out_path}")
            saved_paths.append(out_path)
            write_ordinal += 1
        frame_idx += 1

    cap.release()

    if len(saved_paths) == 0:
        raise RuntimeError("No frames were saved (check inputs).")

    return saved_paths


def frames_to_video(
    frames_dir: Path,
    output_path: Path,
    fps: Optional[float] = None,
    codec: str = "mp4v",
    ref_video: Optional[Path] = None,
) -> None:
    """
    Assemble all images in frames_dir into a video at output_path.
    - fps: if None, and ref_video is provided, uses ref_video's FPS; else defaults to 30.
    - codec: 'mp4v' is a safe default for .mp4. Alternatives: 'avc1', 'XVID'.
    - Resizes frames on the fly to the size of the first frame to avoid writer errors.
    """
    images = sorted(
        [p for p in frames_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=_natural_key,
    )
    if not images:
        raise RuntimeError(f"No images found in {frames_dir}")

    # Determine FPS
    if fps is None and ref_video is not None and ref_video.exists():
        _, ref_fps, _ = _get_video_meta(ref_video)
        fps = ref_fps if ref_fps > 0 else 30.0
    if fps is None:
        fps = 30.0

    # Read first image to get size
    first = cv2.imread(str(images[0]))
    if first is None:
        raise RuntimeError(f"Could not read first image: {images[0]}")
    h, w = first.shape[:2]
    size = (w, h)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    if not out.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter with codec='{codec}'. "
            f"Try another codec (e.g., 'avc1' or 'XVID') or ensure proper codecs are installed."
        )

    # Write frames (resize if needed)
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            out.release()
            raise RuntimeError(f"Could not read image: {img_path}")
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        out.write(frame)

    out.release()


def main():
    parser = argparse.ArgumentParser(description="Extract frames and assemble videos.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_extract = subparsers.add_parser("extract", help="Extract evenly spaced frames from a video.")
    p_extract.add_argument("--video", required=True, type=Path, help="Path to input video (e.g., input.mp4)")
    p_extract.add_argument("--out_dir", required=True, type=Path, help="Directory to save frames")
    p_extract.add_argument("--num", required=True, type=int, help="Number of frames to extract")
    p_extract.add_argument("--prefix", default="frame", help="Filename prefix (default: frame)")

    p_assemble = subparsers.add_parser("assemble", help="Assemble frames into a video.")
    p_assemble.add_argument("--frames_dir", required=True, type=Path, help="Directory containing processed frames")
    p_assemble.add_argument("--out", required=True, type=Path, help="Output video path (e.g., output.mp4)")
    p_assemble.add_argument("--fps", type=float, default=None, help="Frames-per-second for output video")
    p_assemble.add_argument("--codec", type=str, default="mp4v", help="FourCC codec (default: mp4v)")
    p_assemble.add_argument("--ref_video", type=Path, default=None, help="Reference video to copy FPS from (optional)")

    args = parser.parse_args()

    if args.cmd == "extract":
        saved = extract_frames(args.video, args.out_dir, args.num, prefix=args.prefix)
        print(f"Saved {len(saved)} frames to: {args.out_dir.resolve()}")

    elif args.cmd == "assemble":
        frames_to_video(args.frames_dir, args.out, fps=args.fps, codec=args.codec, ref_video=args.ref_video)
        print(f"Wrote video: {args.out.resolve()}")


if __name__ == "__main__":
    main()