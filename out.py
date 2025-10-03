from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import os
import shutil, subprocess
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# ===== 使用者輸入初始時間 =====
print("如果您想自訂時間請按 1，若使用預設 08:00:00 請按 0")
choose = int(input().strip())

print("請輸入西元年")
year = int(input().strip())
print("請輸入月")
month = int(input().strip())
print("請輸入日期")
day = int(input().strip())

if choose == 1:
    print("請輸入小時 (0-23)")
    hour = int(input().strip())
    print("請輸入分鐘 (0-59)")
    minute = int(input().strip())
    print("請輸入秒 (0-59，直接按 Enter 使用 0)")
    sec_in = input().strip()
    second = int(sec_in) if sec_in != "" else 0
else:
    hour, minute, second = 8, 0, 0  # 預設 08:00:00

INITIAL_TIME = datetime(year, month, day, hour, minute, second)
print("初始時間：", INITIAL_TIME.strftime("%Y-%m-%d %H:%M:%S"))

# === 模型與影片路徑 ===
model = YOLO(r"D:\best.pt")
VIDEO_PATH = r"D:\5110.mp4"                     # ← 改這裡：1.mp4 / 2.mp4 / 3.mp4 / 4.mp4
cap = cv2.VideoCapture(VIDEO_PATH)

TRACK_CLASSES = [0, 1]  # 0=person, 1=takebook（請依你的模型確認）

# === 檢查影片是否成功打開 ===
if not cap.isOpened():
    raise RuntimeError("無法開啟輸入影片，請確認路徑是否正確。")

# === 取得輸入影像資訊 ===
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1:
    fps = 40.0  # 保守預設
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === 依檔名選 ROI（使用 if/elif） ===
base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]  # "1"、"2"、"3"、"4"

use_roi = True
if base_name == "ch1":
    ROI_XYWH = (216, 276, 1805, 972)       # ← 你的 ROI for 1.mp4
elif base_name == "ch2":
    ROI_XYWH = (420, 0, 1612, 1292)       # ← TODO: 換成你的 ROI for 2.mp4
elif base_name == "ch3":
    ROI_XYWH = (223, 14, 1469, 1377)       # ← TODO: 換成你的 ROI for 3.mp4
elif base_name == "ch4":
    ROI_XYWH = (622, 275, 1617, 1159)       # ← TODO: 換成你的 ROI for 4.mp4
else:
    # 後備方案：整張畫面
    ROI_XYWH = (0, 0, width, height)
    print(f"警告：檔名 {base_name} 未定義 ROI，改用全畫面。")

# （可選）安全檢查，避免 ROI 超界
x, y, w, h = ROI_XYWH
assert 0 <= x < width and 0 <= y < height, "ROI 左上角超出影像範圍"
assert x + w <= width and y + h <= height, "ROI 寬高超出影像範圍"

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = max(1, (boxA[2]-boxA[0])) * max(1, (boxA[3]-boxA[1]))
    boxBArea = max(1, (boxB[2]-boxB[0])) * max(1, (boxB[3]-boxB[1]))
    return interArea / float(boxAArea + boxBArea - interArea)

# ROI 規則：框中心在 ROI 內（穩定）
def box_in_roi_center(box, roi_xywh):
    if not use_roi:
        return True
    x, y, w, h = roi_xywh
    cx = 0.5 * (box[0] + box[2])
    cy = 0.5 * (box[1] + box[3])
    return (x <= cx <= x + w) and (y <= cy <= y + h)

roi_filter = lambda b: box_in_roi_center(b, ROI_XYWH)

# === 先用 MJPG 寫 AVI（檔名帶上影片代號，避免覆蓋） ===
avi_path = f"output_tracking_{base_name}.avi"
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))
if not out.isOpened():
    raise RuntimeError("VideoWriter 初始化失敗（AVI/MJPG）。")

# === 追蹤與統計 ===
box_history = defaultdict(list)  # 每個 ID 的歷史框（簡易 IoU tracker 產生的 ID）
label_by_id = {}                 # ID -> 類別名稱（字串）
class_by_id = {}                 # ID -> 類別 ID（數字）
next_object_id = 0
frame_idx = 0

# 全程累計（唯一 ID）
seen_person_ids   = set()
seen_takebook_ids = set()

# 以「日期 + 小時」為桶聚合（每列：date, hour, person, takebook）
# key: (date_str, hour_int) -> Counter({'person': x, 'takebook': y})
hour_buckets = defaultdict(lambda: Counter())

# === 平均框函數 ===
def average_box(box_list):
    return np.mean(np.array(box_list), axis=0)

# === 主迴圈 ===
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    # 由初始時間 + fps 推算當前幀的時間戳
    timestamp = INITIAL_TIME + timedelta(seconds=frame_idx / float(fps))
    date_str = timestamp.date().isoformat()
    hour_int = timestamp.hour

    # --- YOLO 偵測 ---
    results = model(frame, conf=0.6)
    current_boxes, current_clsids, current_labels = [], [], []

    for box in results[0].boxes:
        if box.conf[0] < 0.2:
            continue
        cls_id = int(box.cls[0].item())
        if cls_id in TRACK_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            b = [float(x1), float(y1), float(x2), float(y2)]
            label = model.names[cls_id]

            # 只保留 ROI 內的偵測
            if roi_filter(b):
                current_boxes.append(b)
                current_clsids.append(cls_id)
                current_labels.append(label)

    # === 以簡易 IoU 進行賦值追蹤 ===
    new_object_ids = {}
    matched_ids = set()

    for cur_box, cls_id, label in zip(current_boxes, current_clsids, current_labels):
        matched = False
        for obj_id, history_boxes in box_history.items():
            if obj_id in matched_ids or len(history_boxes) == 0:
                continue
            avg_box = average_box(history_boxes)
            iou = compute_iou(cur_box, avg_box)
            if iou > 0.3:
                new_object_ids[obj_id] = (cur_box, cls_id, label)
                matched_ids.add(obj_id)
                matched = True
                break
        if not matched:
            new_object_ids[next_object_id] = (cur_box, cls_id, label)
            # 新 ID 出現 → 計入對應小時桶
            if cls_id == 0:  # person
                if next_object_id not in seen_person_ids:
                    seen_person_ids.add(next_object_id)
                    hour_buckets[(date_str, hour_int)]['person'] += 1
            elif cls_id == 1:  # takebook
                if next_object_id not in seen_takebook_ids:
                    seen_takebook_ids.add(next_object_id)
                    hour_buckets[(date_str, hour_int)]['takebook'] += 1
            next_object_id += 1

    # 更新歷史紀錄並畫框
    for obj_id, (b, cls_id, label) in new_object_ids.items():
        box_history[obj_id].append(b)
        label_by_id[obj_id] = label
        class_by_id[obj_id] = cls_id
        if len(box_history[obj_id]) > 3:
            box_history[obj_id].pop(0)

        x1, y1, x2, y2 = map(int, b)
        color = (0, 255, 0) if cls_id == 0 else (0, 128, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{base_name}|ID:{obj_id} {label}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 疊 ROI 視覺化（半透明藍色 + 外框）
    if use_roi:
        x, y, w, h = ROI_XYWH
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 寫入輸出影片
    out.write(frame)

    # 顯示畫面（如果不想顯示，可以註解掉）
    cv2.imshow("Multi-Class Tracking (ROI + Hourly Counts)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

    frame_idx += 1

# === 正確釋放資源 ===
cap.release()
out.release()
cv2.destroyAllWindows()

# === 匯出「逐小時」統計到 Excel ===
rows = []
for (date_str, hour_int), cnt in sorted(hour_buckets.items(), key=lambda x: (x[0][0], x[0][1])):
    rows.append({
        "date": date_str,
        "hour": hour_int,
        "person": int(cnt.get("person", 0)),
        "takebook": int(cnt.get("takebook", 0)),
    })

df_hourly = pd.DataFrame(rows, columns=["date", "hour", "person", "takebook"])
output_file = "iou_tracking_hourly.xlsx"

if os.path.exists(output_file):
    try:
        existing = pd.read_excel(output_file)
        combined = pd.concat([existing, df_hourly], ignore_index=True)
        df_hourly = (combined
                     .groupby(["date", "hour"], as_index=False)[["person", "takebook"]]
                     .sum()
                     .sort_values(["date", "hour"]))
    except Exception as e:
        print("讀取舊 Excel 失敗，將覆蓋寫入。錯誤：", e)

df_hourly.to_excel(output_file, index=False)
print(f"已儲存逐小時統計：{output_file}")

# === 如果有 ffmpeg，自動轉成 MP4 (H.264) ===
ffmpeg = shutil.which("ffmpeg")
if ffmpeg:
    mp4_path = f"output_tracking_{base_name}.mp4"
    cmd = [
        ffmpeg, "-y", "-i", f"{avi_path}",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-movflags", "+faststart",
        mp4_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"已轉檔成 MP4：{mp4_path}")
    except subprocess.CalledProcessError as e:
        print("ffmpeg 轉檔失敗，保留 AVI。錯誤訊息：", e.stderr.decode("utf-8", errors="ignore"))
else:
    print("未偵測到 ffmpeg：保留 AVI。若要 MP4，請安裝 ffmpeg 或把它加到 PATH。")

