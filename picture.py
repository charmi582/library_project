import cv2
import os

# === 設定參數 ===
video_path = r'D:\4.mp4'        # 替換成你的影片路徑
output_folder = 'frames_every_4'     # 輸出影像的資料夾
frame_interval = 3                   # 每3幀取出1張圖

# 建立輸出資料夾
os.makedirs(output_folder, exist_ok=True)

# 讀取影片
cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f'frame_{saved_count:04d}.jpg')
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ 完成：共儲存 {saved_count} 張影像（每3幀取1張）至「{output_folder}」資料夾。")
