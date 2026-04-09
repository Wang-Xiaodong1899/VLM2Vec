import os
import argparse
from decord import VideoReader, cpu
import cv2
from tqdm import tqdm

def get_frame_indices(total_frames, original_fps, target_fps, num_frm, multiple=1):
    """
    获取帧索引，每秒提取一帧
    """
    sample_fps = max(1, round(original_fps / target_fps))
    frame_idx = [i for i in range(0, total_frames, sample_fps)]
    if len(frame_idx) < num_frm:
        while len(frame_idx) % multiple != 0:
            frame_idx.append(0)
        # If we have fewer frames than num_frm, just return all the frames
        return frame_idx 
    scale = 1.0 * len(frame_idx) / num_frm
    uniform_idx = [round((i + 1) * scale - 1) for i in range(num_frm)]
    frame_idx = [frame_idx[i] for i in uniform_idx]
    return frame_idx

def extract_clip_frames(video_path, clip_index, output_dir, interval_s=5):
    """
    根据clip_index提取视频片段帧并保存为图像
    
    Args:
        video_path: 视频文件路径
        clip_index: clip下标（从0开始）
        output_dir: 输出目录
        interval_s: 每个clip的时长（秒），默认为5秒
    """
    # 读取视频
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    original_fps = vr.get_avg_fps()
    target_fps = 1  # 每秒一帧
    
    # 获取所有帧索引（每秒一帧）
    indices = get_frame_indices(total_frames, original_fps, target_fps, num_fram=total_frames)
    
    # 计算clip对应的帧范围
    start_frame_idx = clip_index * interval_s  # 起始秒数
    end_frame_idx = (clip_index + 1) * interval_s  # 结束秒数
    
    # 确保不超出范围
    start_frame_idx = min(start_frame_idx, len(indices) - 1)
    end_frame_idx = min(end_frame_idx, len(indices))
    
    if start_frame_idx >= len(indices):
        print(f"Warning: clip_index {clip_index} is out of range. Video has {len(indices)} frames.")
        return
    
    # 提取对应clip的帧
    clip_frames_indices = indices[start_frame_idx:end_frame_idx]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 提取并保存帧
    for i, frame_idx in enumerate(clip_frames_indices):
        try:
            # 获取帧
            frame = vr[frame_idx].asnumpy()
            
            # 转换为RGB（如果需要）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 保存图像
            output_path = os.path.join(output_dir, f"clip_{clip_index}_frame_{i:03d}.jpg")
            cv2.imwrite(output_path, frame_rgb)
            
            print(f"Saved frame {i} to {output_path}")
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {str(e)}")
            continue
    
    print(f"Successfully extracted {len(clip_frames_indices)} frames for clip {clip_index}")
    print(f"Time range: {start_frame_idx}-{end_frame_idx} seconds")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from video clip based on clip index')
    parser.add_argument('--video_path', required=True, help='Path to the video file')
    parser.add_argument('--clip_index', type=int, required=True, help='Clip index (starting from 0)')
    parser.add_argument('--output_dir', required=True, help='Output directory for frames')
    parser.add_argument('--interval_s', type=int, default=5, help='Duration of each clip in seconds (default: 5)')
    
    args = parser.parse_args()
    
    extract_clip_frames(args.video_path, args.clip_index, args.output_dir, args.interval_s)

if __name__ == '__main__':
    main()
