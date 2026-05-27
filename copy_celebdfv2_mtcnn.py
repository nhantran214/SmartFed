import os
import shutil

SRC_ROOT = "data/Celeb-DF-v2-mtcnn"
DST_ROOT = "CelebDFv2_mtcnn"

VAL_LIST = os.path.join(SRC_ROOT, "List_of_testing_videos.txt")
SYNTHESIS_DIR = os.path.join(SRC_ROOT, "Celeb-synthesis")
REAL_DIR = os.path.join(SRC_ROOT, "Celeb-real")
YOUTUBE_DIR = os.path.join(SRC_ROOT, "YouTube-real")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def copy_images(src_folder, dst_folder, exclude_files=None):
    if not os.path.exists(src_folder):
        print(f"Warning: {src_folder} does not exist.")
        return
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), src_folder)
                if exclude_files and rel_path in exclude_files:
                    continue
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_folder, file)
                shutil.copy2(src_path, dst_path)

def get_val_image_set():
    val_image_set = set()
    with open(VAL_LIST, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            label, rel_path = parts
            if rel_path.endswith(".mp4"):
                rel_path = rel_path[:-4]
            src_folder = os.path.join(SRC_ROOT, rel_path)
            if not os.path.exists(src_folder):
                continue
            for root, dirs, files in os.walk(src_folder):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        rel_file = os.path.relpath(os.path.join(root, file), src_folder)
                        val_image_set.add((rel_path, rel_file))
    return val_image_set

def main():
    # 1. Xử lý val
    val_ai_dir = os.path.join(DST_ROOT, "val", "ai")
    val_nature_dir = os.path.join(DST_ROOT, "val", "nature")
    ensure_dir(val_ai_dir)
    ensure_dir(val_nature_dir)

    print("Copying validation images...")
    val_folders = set()
    with open(VAL_LIST, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            label, rel_path = parts
            if rel_path.endswith(".mp4"):
                rel_path = rel_path[:-4]
            src_folder = os.path.join(SRC_ROOT, rel_path)
            val_folders.add(rel_path)
            if label == "0":
                dst_folder = val_ai_dir
            elif label == "1":
                dst_folder = val_nature_dir
            else:
                continue
            print(f"Copying from {src_folder} to {dst_folder}")
            copy_images(src_folder, dst_folder)

    # 2. Lấy set các ảnh val để loại trừ khi copy train
    print("Indexing validation images for exclusion from train...")
    val_image_set = get_val_image_set()

    # 3. Xử lý train/ai (Celeb-synthesis, trừ các folder đã dùng cho val)
    train_ai_dir = os.path.join(DST_ROOT, "train", "ai")
    ensure_dir(train_ai_dir)
    print("Copying train/ai images (excluding val)...")
    for folder in os.listdir(SYNTHESIS_DIR):
        rel_folder = f'Celeb-synthesis/{folder}'
        src_folder = os.path.join(SYNTHESIS_DIR, folder)
        if os.path.isdir(src_folder) and rel_folder not in val_folders:
            print(f"Copying from {src_folder} to {train_ai_dir}")
            exclude_files = {f for (rel, f) in val_image_set if rel == rel_folder}
            copy_images(src_folder, train_ai_dir, exclude_files=exclude_files)

    # 4. Xử lý train/nature (Celeb-real + YouTube-real, trừ các folder đã dùng cho val)
    train_nature_dir = os.path.join(DST_ROOT, "train", "nature")
    ensure_dir(train_nature_dir)
    print("Copying train/nature images (excluding val)...")
    for real_dir, prefix in [(REAL_DIR, "Celeb-real"), (YOUTUBE_DIR, "YouTube-real")]:
        for folder in os.listdir(real_dir):
            rel_folder = f'{prefix}/{folder}'
            src_folder = os.path.join(real_dir, folder)
            if os.path.isdir(src_folder) and rel_folder not in val_folders:
                print(f"Copying from {src_folder} to {train_nature_dir}")
                exclude_files = {f for (rel, f) in val_image_set if rel == rel_folder}
                copy_images(src_folder, train_nature_dir, exclude_files=exclude_files)

    print("Done copying and organizing dataset.")

if __name__ == "__main__":
    main()