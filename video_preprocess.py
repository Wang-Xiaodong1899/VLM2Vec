import argparse
import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
from decord import VideoReader, cpu
from tqdm import tqdm


VIDEO_EXTENSIONS = [
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".flv",
    ".wmv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".3gp",
    ".m4v",
    ".3g2",
    ".mts",
]


def _to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _frame_dir_name(video_relpath: str) -> str:
    return os.path.splitext(video_relpath)[0].replace("/", "_")


def _resolve_video_path(video_root: str, video_relpath: str) -> str:
    if os.path.isabs(video_relpath):
        return video_relpath

    cand = os.path.join(video_root, video_relpath)
    if os.path.exists(cand):
        return cand

    base = os.path.splitext(video_relpath)[0]
    for ext in VIDEO_EXTENSIONS:
        cand2 = os.path.join(video_root, base + ext)
        if os.path.exists(cand2):
            return cand2

    return cand


def _iter_video_relpaths(data: object) -> Iterable[str]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "video" in item:
                yield item["video"]
    elif isinstance(data, dict):
        if "video" in data and isinstance(data["video"], str):
            yield data["video"]
        for v in data.values():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "video" in item:
                        yield item["video"]


def extract_1fps_frames(
    video_path: str,
    out_dir: str,
    target_fps: float = 1.0,
    overwrite: bool = False,
    jpg_quality: int = 95,
    num_threads: int = 1,
) -> Tuple[int, float]:
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(out_dir, "meta.json")
    if (not overwrite) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta.get("num_frames", 0)), float(meta.get("original_fps", 0.0))

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=num_threads)
    total_frames = len(vr)
    original_fps = float(vr.get_avg_fps())
    sample_step = max(1, round(original_fps / float(target_fps)))

    raw_indices = list(range(0, total_frames, sample_step))

    for sampled_idx, raw_idx in enumerate(raw_indices):
        frame = vr[raw_idx].asnumpy()
        frame_bgr = _to_bgr(frame)
        out_path = os.path.join(out_dir, f"{sampled_idx:06d}.jpg")
        cv2.imwrite(out_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "video_path": video_path,
                "target_fps": float(target_fps),
                "original_fps": original_fps,
                "sample_step": int(sample_step),
                "total_frames": int(total_frames),
                "num_frames": int(len(raw_indices)),
                "sampled_to_original": raw_indices,
            },
            f,
            ensure_ascii=False,
        )

    return len(raw_indices), original_fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--out_root", required=True, help="video_frame_basedir")
    parser.add_argument("--target_fps", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jpg_quality", type=int, default=95)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    videos = sorted(set(_iter_video_relpaths(data)))
    if args.limit and args.limit > 0:
        videos = videos[: args.limit]

    os.makedirs(args.out_root, exist_ok=True)

    bad: List[Dict[str, str]] = []
    for video_rel in tqdm(videos, desc="extract_1fps_frames"):
        video_path = _resolve_video_path(args.video_root, video_rel)
        if not os.path.exists(video_path):
            bad.append({"video": video_rel, "error": "video_not_found", "path": video_path})
            continue

        frame_dir = os.path.join(args.out_root, _frame_dir_name(video_rel))
        try:
            extract_1fps_frames(
                video_path=video_path,
                out_dir=frame_dir,
                target_fps=args.target_fps,
                overwrite=args.overwrite,
                jpg_quality=args.jpg_quality,
                num_threads=args.num_threads,
            )
        except Exception as e:
            bad.append({"video": video_rel, "error": repr(e), "path": video_path})

    if bad:
        bad_path = os.path.join(args.out_root, "extract_bad.json")
        with open(bad_path, "w", encoding="utf-8") as f:
            json.dump(bad, f, ensure_ascii=False, indent=2)
        print(f"Bad videos: {len(bad)}. Saved to {bad_path}")


if __name__ == "__main__":
    main()
