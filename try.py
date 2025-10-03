from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# IoU 函數
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# 計算 box 清單的平均框
def average_box(box_list):
    return np.mean(np.array(box_list), axis=0)

# 初始化
model = YOLO(r"C:\Users\user\Desktop\test\runs\detect\train4\weights\best.pt" )
cap = cv2.VideoCapture(r"D:\ch2.mp4")

# 每個 ID 對應的歷史框列表（最多保留 3 個）
box_history = defaultdict(list)
next_object_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 偵測（信心度過濾）
    results = model(frame, conf=0.6)
    current_boxes = []

    for box in results[0].boxes:
        if box.conf[0] < 0.2:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        current_boxes.append([x1, y1, x2, y2])

    # 比對：用歷史平均框來做比對
    new_object_ids = {}
    matched_ids = set()

    for cur_box in current_boxes:
        matched = False
        for obj_id, history_boxes in box_history.items():
            if obj_id in matched_ids:
                continue
            if len(history_boxes) == 0:
                continue
            avg_box = average_box(history_boxes)
            iou = compute_iou(cur_box, avg_box)
            if iou > 0.3:
                new_object_ids[obj_id] = cur_box
                matched_ids.add(obj_id)
                matched = True
                break
        if not matched:
            new_object_ids[next_object_id] = cur_box
            next_object_id += 1

    # 更新歷史框：最多保留 3 幀的框
    for obj_id, box in new_object_ids.items():
        box_history[obj_id].append(box)
        if len(box_history[obj_id]) > 3:
            box_history[obj_id].pop(0)

    # 畫圖
    for obj_id, box in new_object_ids.items():
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Stable Tracking by IoU + AvgBox", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
