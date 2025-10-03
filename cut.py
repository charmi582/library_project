import os
import random
import shutil

# ==== 參數設定 ====
image_dir = r"C:\Users\user\Desktop\shuffled_images"     # 原始圖片資料夾
label_dir = r"C:\Users\user\Desktop\labelimg"     # 原始標籤資料夾 (YOLO .txt)
output_dir = "dataset"       # 輸出資料夾
train_ratio = 0.7            # 訓練集比例
val_ratio = 0.2              # 驗證集比例 (剩下就是測試集)
random.seed(42)              # 固定隨機種子，結果可重現

# ==== 建立 YOLO 資料夾結構 ====
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# ==== 讀取所有圖片檔案 ====
images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(images)

# ==== 計算切割點 ====
total = len(images)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_files = images[:train_end]
val_files = images[train_end:val_end]
test_files = images[val_end:]

# ==== 定義檔案複製函數 ====
def copy_files(file_list, split):
    for img_file in file_list:
        # 圖片來源 & 目的地
        src_img = os.path.join(image_dir, img_file)
        dst_img = os.path.join(output_dir, 'images', split, img_file)
        shutil.copy(src_img, dst_img)

        # 對應標籤檔
        label_file = os.path.splitext(img_file)[0] + '.txt'
        src_label = os.path.join(label_dir, label_file)
        dst_label = os.path.join(output_dir, 'labels', split, label_file)

        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)
        else:
            print(f"[警告] 找不到標籤檔：{label_file}")

# ==== 開始複製 ====
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print(f"✅ 分割完成！共 {total} 張圖片：")
print(f"  訓練集：{len(train_files)} 張")
print(f"  驗證集：{len(val_files)} 張")
print(f"  測試集：{len(test_files)} 張")
