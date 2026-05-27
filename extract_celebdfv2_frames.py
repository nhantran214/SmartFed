import os
import cv2

SRC_VIDEO_ROOT = 'data/celebdfv2_ori'
DST_ROOT = 'CelebDFv2_ori_frames'
LIST_FILE = os.path.join(SRC_VIDEO_ROOT, "List_of_testing_videos.txt")
NUM_FRAMES = 10  # Số frame muốn lấy mẫu từ mỗi video

# Tạo thư mục nếu chưa tồn tại
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_frames_from_video(video_path, dst_folder, num_frames=5, prefix=None):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"[Warning] Video {video_path} is empty or cannot be read.")
        cap.release()
        return
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frame_indices = list(sorted(set(frame_indices)))
    saved = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"{prefix}_frame{idx:04d}.jpg" if prefix else f"frame{idx:04d}.jpg"
        cv2.imwrite(os.path.join(dst_folder, fname), frame)
        saved += 1
    cap.release()
    print(f"Saved {saved} frames from {video_path} to {dst_folder}")

def get_val_video_set():
    val_videos = set()
    with open(LIST_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, rel_path = line.split()
            video_name = os.path.splitext(os.path.basename(rel_path))[0]
            val_videos.add(video_name)
    return val_videos

def main():
    # Thư mục val
    val_ai_dir = os.path.join(DST_ROOT, 'val', 'ai')
    val_nature_dir = os.path.join(DST_ROOT, 'val', 'nature')
    ensure_dir(val_ai_dir)
    ensure_dir(val_nature_dir)
    # Thư mục train
    train_ai_dir = os.path.join(DST_ROOT, 'train', 'ai')
    train_nature_dir = os.path.join(DST_ROOT, 'train', 'nature')
    ensure_dir(train_ai_dir)
    ensure_dir(train_nature_dir)

    val_videos = get_val_video_set()

    # Xử lý val
    with open(LIST_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, rel_path = line.split()
            label = int(label)
            video_name = os.path.splitext(os.path.basename(rel_path))[0]
            video_path = os.path.join(SRC_VIDEO_ROOT, rel_path)
            if not os.path.exists(video_path):
                print(f"[Warning] Video not found: {video_path}")
                continue
            if label == 0:
                dst_folder = val_ai_dir
            else:
                dst_folder = val_nature_dir
            extract_frames_from_video(video_path, dst_folder, NUM_FRAMES, prefix=video_name)

    # Xử lý train
    for root, dirs, files in os.walk(SRC_VIDEO_ROOT):
        for file in files:
            if not file.lower().endswith('.mp4'):
                continue
            video_name = os.path.splitext(file)[0]
            if video_name in val_videos:
                continue  # Bỏ qua video đã dùng cho val
            video_path = os.path.join(root, file)
            # Xác định nhãn dựa vào folder cha
            parent = os.path.basename(os.path.dirname(video_path)).lower()
            if 'synth' in parent:
                dst_folder = train_ai_dir
            elif 'real' in parent or 'youtube' in parent:
                dst_folder = train_nature_dir
            else:
                print(f"[Warning] Unknown label for video: {video_path}")
                continue
            extract_frames_from_video(video_path, dst_folder, NUM_FRAMES, prefix=video_name)

if __name__ == '__main__':
    main() 