import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from decord import VideoReader, cpu
import cv2


def get_frame_indices(
    total_frames: int,
    original_fps: float,
    target_fps: float,
    num_frm: int,
    multiple: int = 1,
) -> List[int]:
    sample_fps = max(1, round(original_fps / target_fps))
    frame_idx = [i for i in range(0, total_frames, sample_fps)]
    if len(frame_idx) < num_frm:
        while len(frame_idx) % multiple != 0:
            frame_idx.append(0)
        return frame_idx
    scale = 1.0 * len(frame_idx) / num_frm
    uniform_idx = [round((i + 1) * scale - 1) for i in range(num_frm)]
    frame_idx = [frame_idx[i] for i in uniform_idx]
    return frame_idx


def _clamp_int_list(values: Sequence[Any], low: int, high: int) -> List[int]:
    out: List[int] = []
    for v in values:
        iv = int(v)
        if iv < low:
            iv = low
        if iv > high:
            iv = high
        out.append(iv)
    return out


class VideoITGClipKeyframeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        json_path: str,
        video_root: str = "",
        interval_s: int = 5,
        target_fps: float = 1.0,
        clip_index_base: int = 0,
        max_keyframes: Optional[int] = None,
        max_clip_frames: Optional[int] = None,
        return_frames: bool = True,
        return_sampled_to_original: bool = False,
        num_threads: int = 1,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {json_path}, got {type(data)}")

        self.data = data
        self.json_path = json_path
        self.video_root = video_root
        self.interval_s = int(interval_s)
        self.target_fps = float(target_fps)
        self.clip_index_base = int(clip_index_base)
        self.max_keyframes = max_keyframes
        self.max_clip_frames = max_clip_frames
        self.return_frames = bool(return_frames)
        self.return_sampled_to_original = bool(return_sampled_to_original)
        self.num_threads = int(num_threads)

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_video_path(self, video: str) -> str:
        if os.path.isabs(video):
            return video
        if self.video_root:
            return os.path.join(self.video_root, video)
        return video

    def _sampled_to_original_indices(self, vr: VideoReader) -> List[int]:
        total_frames = len(vr)
        original_fps = float(vr.get_avg_fps())
        return get_frame_indices(total_frames, original_fps, self.target_fps, num_frm=total_frames)

    def _clip_sampled_positions(self, clip_num: int, sampled_len: int) -> List[int]:
        clip_idx = int(clip_num) - self.clip_index_base
        start = clip_idx * self.interval_s
        end = start + self.interval_s
        if sampled_len <= 0:
            return []
        start = max(0, min(start, sampled_len))
        end = max(0, min(end, sampled_len))
        return list(range(start, end))

    def getitem(self, index: int) -> Dict[str, Any]:
        item = self.data[index]
        video_rel = item["video"]
        video_path = self._resolve_video_path(video_rel)

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=self.num_threads)
        sampled_to_original = self._sampled_to_original_indices(vr)
        sampled_len = len(sampled_to_original)
        if sampled_len == 0:
            raise RuntimeError(f"No frames sampled from {video_path}")

        clip_num_field = item.get("clip_num", [])
        if isinstance(clip_num_field, list) and len(clip_num_field) > 0:
            clip_num = int(clip_num_field[0])
        else:
            clip_num = None

        frame_num_field = item.get("frame_num", [])
        if not isinstance(frame_num_field, list):
            frame_num_field = []

        sampled_key_positions = _clamp_int_list(frame_num_field, 0, sampled_len - 1)
        if self.max_keyframes is not None:
            sampled_key_positions = sampled_key_positions[: int(self.max_keyframes)]

        sampled_clip_positions: List[int] = []
        if clip_num is not None:
            sampled_clip_positions = self._clip_sampled_positions(clip_num, sampled_len)
            if self.max_clip_frames is not None:
                sampled_clip_positions = sampled_clip_positions[: int(self.max_clip_frames)]
        print(sampled_key_positions)
        print(sampled_clip_positions)
        orig_key_indices = [sampled_to_original[p] for p in sampled_key_positions]
        orig_clip_indices = [sampled_to_original[p] for p in sampled_clip_positions]

        # save frames to .jpg file
        # keyframes = vr.get_batch(orig_key_indices).asnumpy()
        # clipframes = vr.get_batch(orig_clip_indices).asnumpy()
        # print(keyframes.shape)
        # print(clipframes.shape)
        def _to_bgr(img: np.ndarray) -> np.ndarray:
            if img.ndim == 3 and img.shape[-1] == 3:
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img

        # for i, frame in enumerate(keyframes):
        #     cv2.imwrite(f"key_{i}.jpg", _to_bgr(frame))
        # for i, frame in enumerate(clipframes):
        #     cv2.imwrite(f"clip_{i}.jpg", _to_bgr(frame))


        result: Dict[str, Any] = {
            "id": item.get("id"),
            "video": video_rel,
            "video_path": video_path,
            "question": item.get("question"),
            "answer": item.get("answer"),
            "existence": item.get("existence"),
            "motion": item.get("motion"),
            "clip_num": clip_num_field,
            "frame_num": frame_num_field,
            "sampled_len": sampled_len,
            "sampled_key_positions": sampled_key_positions,
            "sampled_clip_positions": sampled_clip_positions,
            "orig_key_indices": orig_key_indices,
            "orig_clip_indices": orig_clip_indices,
        }

        if self.return_sampled_to_original:
            result["sampled_to_original"] = sampled_to_original

        if self.return_frames:
            if len(orig_key_indices) > 0:
                keyframes = vr.get_batch(orig_key_indices).asnumpy()
            else:
                keyframes = np.zeros((0, 1, 1, 3), dtype=np.uint8)

            if len(orig_clip_indices) > 0:
                clipframes = vr.get_batch(orig_clip_indices).asnumpy()
            else:
                clipframes = np.zeros((0, 1, 1, 3), dtype=np.uint8)

            result["keyframes"] = keyframes
            result["clipframes"] = clipframes

        return result

    def __getitem__(self, index: int) -> Dict[str, Any]:
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception:
                index = int(torch.randint(low=0, high=len(self), size=(1,)).item())
        raise RuntimeError("Too many bad samples.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default="test.json")
    parser.add_argument("--video_root", default="/mnt/bn/wxd-video-understanding/wangxd/data")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    ds = VideoITGClipKeyframeDataset(args.json_path, video_root=args.video_root)
    sample = ds[args.index]
    print(
        {
            "id": sample["id"],
            "video": sample["video"],
            "clip_num": sample["clip_num"],
            "frame_num": sample["frame_num"],
            "sampled_len": sample["sampled_len"],
            "orig_clip_indices": sample["orig_clip_indices"],
            "orig_key_indices": sample["orig_key_indices"],
            "clipframes_shape": None if "clipframes" not in sample else list(sample["clipframes"].shape),
            "keyframes_shape": None if "keyframes" not in sample else list(sample["keyframes"].shape),
        }
    )