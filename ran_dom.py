import os
import random
import shutil

# === 設定參數 ===
source_folder = r'C:\Users\user\Desktop\test\picture'    # 原始圖片資料夾
output_folder = 'shuffled_images'    # 打亂後輸出資料夾
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']  # 可支援的圖片格式

# 建立輸出資料夾
os.makedirs(output_folder, exist_ok=True)

# 收集所有圖片檔案
all_images = [
    f for f in os.listdir(source_folder)
    if os.path.splitext(f)[1].lower() in image_extensions
]

# 打亂圖片順序
random.shuffle(all_images)

# 複製並重新命名圖片
for idx, filename in enumerate(all_images):
    ext = os.path.splitext(filename)[1].lower()
    new_name = f'{idx + 1:04d}{ext}'  # 例如：0001.jpg
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(output_folder, new_name)
    shutil.copy2(src_path, dst_path)

print(f"✅ 共處理 {len(all_images)} 張圖片，打亂後儲存於：{output_folder}/")
