import os
from typing import Any, Dict, List, Optional

import datasets
import random

from src.data.dataset.base_pair_dataset import AutoPairDataset, add_metainfo_hook, MULTIMODAL_FEATURES, RESOLUTION_MAPPING
from src.model.processor import process_input_text
from src.utils.vision_utils.vision_utils import load_frames, sample_frames


TASK_INST_QRY = "Based on the question, identify the key content in the video."
TASK_INST_TGT = "Understand the content of the provided video."


def _resolve_frame_dir(frame_basedir: str, video: str) -> Optional[str]:
    video_noext = os.path.splitext(video)[0]
    normalized = video_noext.replace("/", "_")
    candidates = [
        os.path.join(frame_basedir, normalized),
        os.path.join(frame_basedir, video_noext),
        os.path.join(frame_basedir, video.replace("/", "_")),
        os.path.join(frame_basedir, os.path.basename(video_noext)),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    return None


def _pick_by_indices(frames: List[str], indices: List[int]) -> List[str]:
    out = []
    for idx in indices:
        if 0 <= idx < len(frames):
            out.append(frames[idx])
    return out


def _clip_indices(clip_num: Any, interval_s: int, max_idx: int, max_context_clip: int = 5) -> List[int]:
    if clip_num is None:
        return []

    if isinstance(clip_num, str):
        if clip_num.lower() == "all":
            return list(range(0, max_idx + 1))
        return []

    if isinstance(clip_num, (int, float)):
        clip_list = [int(clip_num)]
    elif isinstance(clip_num, list):
        clip_list = [int(x) for x in clip_num if isinstance(x, (int, float)) or (isinstance(x, str) and str(x).isdigit())]
    else:
        return []

    indices: List[int] = []
    for c in clip_list:
        # random get a left_context from nums [0,1,2,...max_context_clip]
        left_context = random.randint(0, max_context_clip)
        right_context = random.randint(0, max_context_clip)
        start_clip = c - left_context
        end_clip = c + right_context
        for k in range(start_clip, end_clip + 1):
            start = k * interval_s
            end = start + interval_s
            indices.extend(list(range(start, end)))

    indices = sorted(set(i for i in indices if 0 <= i <= max_idx))
    return indices


def _normalize_int_list(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, list):
        out = []
        for v in values:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    return []


@add_metainfo_hook
def data_prepare_videoitg_clip_keyframe(batch_dict: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
    model_backbone = kwargs["model_backbone"]
    image_resolution = kwargs["image_resolution"]
    frame_basedir = kwargs["video_frame_basedir"]

    interval_s = int(kwargs.get("interval_s", 5))
    query_num_frames = int(kwargs.get("query_num_frames", interval_s))
    pos_num_frames = int(kwargs.get("pos_num_frames", kwargs.get("num_frames", 8)))
    pos_with_question = bool(kwargs.get("pos_with_question", True))

    default_res = RESOLUTION_MAPPING.get(image_resolution, (224, 224))
    if default_res is None:
        default_res = (224, 224)

    query_texts, query_images, pos_texts, pos_images, neg_texts, neg_images = [], [], [], [], [], []

    for data_id, video, question, frame_num, clip_num in zip(
        batch_dict.get("id", []),
        batch_dict.get("video", []),
        batch_dict.get("question", []),
        batch_dict.get("frame_num", []),
        batch_dict.get("clip_num", []),
    ):
        frame_dir = _resolve_frame_dir(frame_basedir, video)
        if frame_dir is None:
            continue

        frames = load_frames(frame_dir)
        if len(frames) == 0:
            continue

        max_idx = len(frames) - 1
        key_indices = _normalize_int_list(frame_num)
        key_indices = sorted(set(i for i in key_indices if 0 <= i <= max_idx))
        key_paths = _pick_by_indices(frames, key_indices)
        if len(key_paths) == 0:
            continue
        if len(key_paths) >= pos_num_frames:
            key_paths = sample_frames(key_paths, num_segments=pos_num_frames)
        else:
            while len(key_paths) < pos_num_frames:
                key_paths.append(key_paths[-1])

        clip_indices = _clip_indices(clip_num, interval_s=interval_s, max_idx=max_idx)
        clip_paths = _pick_by_indices(frames, clip_indices)
        if len(clip_paths) == 0:
            clip_paths = frames

        if query_num_frames == 0:
            clip_paths = []
        elif len(clip_paths) >= query_num_frames:
            clip_paths = sample_frames(clip_paths, num_segments=query_num_frames)
        else:
            while len(clip_paths) < query_num_frames:
                clip_paths.append(clip_paths[-1])

        query_text = process_input_text(TASK_INST_QRY, model_backbone, text=question, add_video_token=True)
        if pos_with_question:
            pos_text = process_input_text(TASK_INST_TGT, model_backbone, text=question, add_video_token=True)
        else:
            pos_text = process_input_text(TASK_INST_TGT, model_backbone, add_video_token=True)

        query_frames = {
            "bytes": [None] * len(clip_paths),
            "paths": clip_paths,
            "resolutions": [default_res] * len(clip_paths),
        }
        pos_frames = {
            "bytes": [None] * len(key_paths),
            "paths": key_paths,
            "resolutions": [default_res] * len(key_paths),
        }

        query_texts.append(query_text)
        query_images.append(query_frames)
        pos_texts.append(pos_text)
        pos_images.append(pos_frames)
        neg_texts.append([])
        neg_images.append([{"bytes": [b""], "paths": [""], "resolutions": [(224, 224)]}])

    return {
        "query_text": query_texts,
        "query_image": query_images,
        "pos_text": pos_texts,
        "pos_image": pos_images,
        "neg_text": neg_texts,
        "neg_image": neg_images,
    }


DATASET_PARSER_NAME = "videoitg_clip_keyframe"


@AutoPairDataset.register(DATASET_PARSER_NAME)
def load_videoitg_clip_keyframe_dataset(model_args, data_args, training_args, *args, **kwargs):
    dataset_name = kwargs.get("dataset_name", DATASET_PARSER_NAME)
    assert "dataset_path" in kwargs, "`dataset_path` should be given for loading videoitg dataset."
    assert "video_frame_basedir" in kwargs, "`video_frame_basedir` should be given for loading videoitg dataset."

    dataset_path = kwargs["dataset_path"]
    dataset = datasets.load_dataset("json", split="train", data_files=dataset_path, streaming=False)

    num_sample_per_subset = kwargs.get("num_sample_per_subset", getattr(data_args, "num_sample_per_subset", None))
    if num_sample_per_subset is not None and num_sample_per_subset < dataset.num_rows:
        num_rows = int(num_sample_per_subset)
        dataset = dataset.select(range(num_rows))
    num_rows = dataset.num_rows

    num_shards = training_args.dataloader_num_workers if training_args.dataloader_num_workers > 0 else 1
    dataset = dataset.to_iterable_dataset(num_shards=num_shards)

    kwargs["model_backbone"] = model_args.model_backbone
    kwargs["image_resolution"] = data_args.image_resolution
    kwargs["video_frame_basedir"] = kwargs["video_frame_basedir"]
    kwargs["global_dataset_name"] = f"{DATASET_PARSER_NAME}/{dataset_name}"

    remove_columns = list(dataset.column_names)
    dataset = dataset.map(
        lambda x: data_prepare_videoitg_clip_keyframe(x, **kwargs),
        batched=True,
        batch_size=128,
        drop_last_batch=True,
        features=MULTIMODAL_FEATURES,
        remove_columns=remove_columns,
    )

    setattr(dataset, "num_rows", num_rows)
    return dataset