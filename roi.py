import cv2, json

VIDEO_PATH = r"D:\ch1.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
ok, frame = cap.read()
if not ok:
    raise RuntimeError("讀不到第一幀")

# 顯示第一幀讓你拉框（Enter/Space 確認，Esc 取消）
roi = cv2.selectROI("框出 ROI（Enter 確認）", frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("框出 ROI（Enter 確認）")
x, y, w, h = map(int, roi)
print("ROI =", (x, y, w, h))

# 存成 JSON，之後你的主程式可讀取使用
base_name = "2"  # 例如對應 ch2.mp4
cfg = {base_name: [x, y, w, h]}
with open("roi_coords.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
cap.release()
