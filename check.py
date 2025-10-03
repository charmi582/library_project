from pathlib import Path

# === 請改成你的路徑 ===
IMAGES_DIR = Path(r"C:\Users\user\Desktop\dataset\images\val")
LABELS_DIR = Path(r"C:\Users\user\Desktop\dataset\label\val")

# 支援的影像副檔名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def delete_unlabeled(images_dir: Path, labels_dir: Path):
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    label_stems = {p.stem for p in labels_dir.glob("*.txt")}

    to_delete = [p for p in images if p.stem not in label_stems]

    print(f"\n📂 {images_dir}  vs  {labels_dir}")
    print(f"找到圖片 {len(images)}，標籤 {len(label_stems)}")
    print(f"準備刪除『有圖沒標籤』：{len(to_delete)} 檔")

    for p in to_delete:
        try:
            p.unlink()
            print("  🗑️ 已刪除 ->", p.name)
        except Exception as e:
            print("  ⛔ 無法刪除 ->", p.name, "|", e)

if __name__ == "__main__":
    delete_unlabeled(IMAGES_DIR, LABELS_DIR)
