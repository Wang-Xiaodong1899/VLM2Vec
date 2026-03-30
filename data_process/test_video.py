from decord import VideoReader, cpu
import cv2

def get_frame_indices(total_frames, original_fps, target_fps, num_frm, multiple=1):
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

def process_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    original_fps = vr.get_avg_fps()
    target_fps = 1
    
    indices = get_frame_indices(total_frames, original_fps, target_fps, num_frm=total_frames)
    frame_features = []
    
    # for start_idx in range(0, len(indices), batch_size):
    #     end_idx = min(start_idx + batch_size, len(indices))
    #     batch_indices = indices[start_idx:end_idx]
    #     batch_frames = vr.get_batch(batch_indices).asnumpy()
    for idx, i in enumerate(indices):
        frame = vr[i].asnumpy()
        cv2.imwrite(f"frames3/frame_{idx}.jpg", frame)

# process_video("ytb_PJY98RKW-tA.mp4")
# process_video("ytb_tCTcOwSUons.mp4")
# process_video("v_E15Q3Z9J-Zg.mp4")
process_video("v_ITVfOVR34Jo.mp4")
