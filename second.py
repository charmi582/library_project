from ultralytics import YOLO
import ultralytics as ul
import cv2
import numpy as np
import pandas as pd
import os, sys, tempfile, traceback
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import torch
import math
import msvcrt
from tqdm import tqdm  # ✅ tqdm 進度條

# ========= 可開關設定 =========
ENABLE_VIDEO_OUTPUT = True   # ✅ 是否輸出影片
ENABLE_EXCEL_OUTPUT = True    # ✅ 是否輸出 Excel 統計結果

# ========= 加速與裝置設定 =========
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

DEVICE = "0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
HALF   = (DEVICE != "cpu")
IMGSZ  = 960
print(f"[Init] device={DEVICE}, half={HALF}, imgsz={IMGSZ}", flush=True)

def p(msg): print(msg, flush=True)

def clamp_roi_to_frame(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return int(x), int(y), int(w), int(h)

def calculate_iou(boxA, boxB):
    # box: (x1, y1, x2, y2)
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

# ========= 互動輸入 =========
p("enter year"); year  = int(input().strip())
p("enter month"); month = int(input().strip())
p("enter day");   day   = int(input().strip())
p("enter hour (0-23)"); hour = int(input().strip())
minute, second = 0, 0
INITIAL_TIME = datetime(year, month, day, hour, minute, second)
p(f"初始時間：{INITIAL_TIME:%Y-%m-%d %H:%M:%S}")

# ========= 路徑設定 =========
OUTPUT_DIR    = Path(r"D:\out").resolve()
VIDEO_OUT_DIR = OUTPUT_DIR / "videos"
MODEL_PATH    = r"C:\Users\user\Desktop\test\test\best.pt"
VIDEO_PATH    = r"D:\dav2mp4\2025-09-18\001\091812.mp4"

for d in [OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
if ENABLE_VIDEO_OUTPUT:
    VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    p(f"? 找不到模型：{MODEL_PATH}")
    sys.exit(1)
if not os.path.isfile(VIDEO_PATH):
    p(f"? 找不到影片：{VIDEO_PATH}")
    sys.exit(1)
p(f"輸出根目錄：{OUTPUT_DIR}")

# ========= 影片資訊 =========
p("開啟影片…")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    p("? 無法開啟影片，請確認檔案/編解碼器")
    sys.exit(1)

raw_fps = cap.get(cv2.CAP_PROP_FPS)
fps = raw_fps if (raw_fps and raw_fps > 1 and not math.isnan(raw_fps)) else 20.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames <= 0:
    total_frames = -1
p(f"影片：{VIDEO_PATH}\n尺寸={width}x{height}, fps?{fps:.2f}（來源:{raw_fps}）, 總幀={total_frames if total_frames>0 else '未知'}")

# ========= ROI =========
USE_ROI_FILTER = True
DRAW_ROI_BOX   = True
ROI_XYWH = (216, 276, 1805, 972)
rx, ry, rw, rh = clamp_roi_to_frame(*ROI_XYWH, width, height)
ROI_XYWH = (rx, ry, rw, rh)
p(f"關注 ROI：{ROI_XYWH}")

def roi_overlap_ok(box, roi_xywh, min_iou=0.05, min_inter_area_ratio=0.30):
    rx, ry, rw, rh = roi_xywh
    x1, y1, x2, y2 = box
    rx2, ry2 = rx + rw, ry + rh
    ix1, iy1 = max(x1, rx), max(y1, ry)
    ix2, iy2 = min(x2, rx2), min(y2, ry2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return False
    area_box = max(1.0, (x2 - x1) * (y2 - y1))
    area_roi = float(rw * rh)
    iou = inter / (area_box + area_roi - inter)
    inter_ratio = inter / area_box
    return (iou >= min_iou) or (inter_ratio >= min_inter_area_ratio)

roi_filter = (lambda b: True) if not USE_ROI_FILTER else (lambda b: roi_overlap_ok(b, ROI_XYWH))

# ========= 模型載入 =========
p("載入模型…")
model = YOLO(MODEL_PATH)
p("? 模型載入 OK")

def build_name_map(names):
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))

NAME_MAP = build_name_map(getattr(model, "names", {}))
p(f"Model classes: {NAME_MAP}")

def find_id_by_names(candidates):
    for k, v in NAME_MAP.items():
        n = str(v).lower()
        for c in candidates:
            if c in n:
                return k
    return None

PERSON_ID   = find_id_by_names(["person", "people", "human", "人"])
TAKEBOOK_ID = find_id_by_names(["takebook", "book", "holding book", "take_book", "拿書", "書"])
TRACK_CLASSES = [i for i in [PERSON_ID, TAKEBOOK_ID] if i is not None]
if not TRACK_CLASSES:
    raise RuntimeError("找不到 person/takebook 類別")

# ========= 參數與統計結構 =========
IOU_NMS         = 0.45
CONF_PERSON     = 0.60
CONF_TAKE       = 0.15
BASE_CONF       = min(CONF_PERSON, CONF_TAKE)
MIN_IOU_OVERLAP = 0.3

TRACKER_CFG = str(Path(ul.__file__).parent / "cfg" / "trackers" / "botsort.yaml")

person_tracks = {}  # {pid: {"has_taken": bool}}
hour_seen_ids = defaultdict(lambda: {"person": set(), "takebook": set()})

# ========= Excel 輸出相關 =========
XLSX_PATH = (OUTPUT_DIR / "excel" / "all_hourly.xlsx").resolve()

def write_excel_locked(xlsx_path: Path, df_new: pd.DataFrame):
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(xlsx_path) + ".lock"
    with open(lock_path, "w") as lk:
        try:
            msvcrt.locking(lk.fileno(), msvcrt.LK_LOCK, 1)

            if xlsx_path.exists() and xlsx_path.stat().st_size > 0:
                try:
                    df_old = pd.read_excel(xlsx_path)
                    keep_cols = ["date", "hour", "person", "takebook"]
                    df_old = df_old[[c for c in keep_cols if c in df_old.columns]]
                except Exception:
                    df_old = pd.DataFrame(columns=["date", "hour", "person", "takebook"])
            else:
                df_old = pd.DataFrame(columns=["date", "hour", "person", "takebook"])

            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all["date"]     = df_all["date"].astype(str)
            df_all["hour"]     = df_all["hour"].astype(int)
            df_all["person"]   = df_all["person"].astype(int)
            df_all["takebook"] = df_all["takebook"].astype(int)

            df_all = (
                df_all.sort_values(["date", "hour"])
                      .drop_duplicates(subset=["date", "hour"], keep="last")
            )

            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_all.to_excel(writer, sheet_name="hourly", index=False)
        finally:
            lk.flush()
            os.fsync(lk.fileno())
            msvcrt.locking(lk.fileno(), msvcrt.LK_UNLCK, 1)

# ========= 影片輸出設定（依開關） =========
writer = None
if ENABLE_VIDEO_OUTPUT:
    base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    avi_path  = str((VIDEO_OUT_DIR / f"output_tracking_{base_name}.avi").resolve())
    mp4_path  = str((VIDEO_OUT_DIR / f"output_tracking_{base_name}.mp4").resolve())

    def try_open_writer(out_path, fourcc_str, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(out_path, fourcc, float(fps), size)
        return vw if vw.isOpened() else None

    def init_video_writer(size):
        global writer
        for fourcc_str, path in (("XVID", avi_path), ("MJPG", avi_path)):
            vw = try_open_writer(path, fourcc_str, fps, size)
            if vw is not None:
                writer = vw
                p(f"? VideoWriter 使用 {fourcc_str} -> {path}")
                return
        vw = try_open_writer(mp4_path, "mp4v", fps, size)
        if vw is not None:
            writer = vw
            p(f"? VideoWriter 使用 mp4v -> {mp4_path}")
            return
        raise RuntimeError("VideoWriter 初始化失敗：請確認系統有可用編碼器/寫入權限。")

# ========= 主迴圈 =========
p("開始處理（使用 YOLO tracking + BoT-SORT ReID）…")
frame_idx = 0
det_total = 0
bad_frame_count = 0
MAX_BAD_FRAMES = 1000

progress_bar = tqdm(
    total=total_frames if total_frames > 0 else None,
    desc="處理中",
    unit="frame"
)

try:
    while True:
        ok, full_frame = cap.read()
        if not ok or full_frame is None:
            if total_frames > 0 and frame_idx >= total_frames:
                p("[資訊] 已處理完所有預期影格，影片正常結束。")
                break
            bad_frame_count += 1
            if bad_frame_count >= MAX_BAD_FRAMES:
                p(f"[錯誤] 連續壞影格已達 {MAX_BAD_FRAMES} 幀上限，程式終止。")
                break
            frame_idx += 1
            progress_bar.update(1)
            continue

        bad_frame_count = 0

        if ENABLE_VIDEO_OUTPUT and writer is None:
            init_video_writer((full_frame.shape[1], full_frame.shape[0]))

        timestamp = INITIAL_TIME + timedelta(seconds=frame_idx / float(fps))
        date_str = timestamp.date().isoformat()
        hour_int = timestamp.hour

        with torch.inference_mode():
            results_list = model.track(
                source=[full_frame],
                imgsz=IMGSZ,
                conf=BASE_CONF,
                iou=IOU_NMS,
                device=DEVICE,
                half=HALF,
                persist=True,
                verbose=False,
                tracker=TRACKER_CFG,
                stream=False
            )

        res = results_list[0] if isinstance(results_list, (list, tuple)) else results_list

        final_detections = []
        persons_in_frame = []
        takebooks_in_frame = []

        if hasattr(res, "boxes") and (res.boxes is not None):
            boxes = res.boxes
            xyxy  = boxes.xyxy.detach().cpu().numpy()
            clss  = boxes.cls.detach().cpu().numpy().astype(int)
            confs = boxes.conf.detach().cpu().numpy()
            ids   = (
                boxes.id.detach().cpu().numpy().astype(int)
                if getattr(boxes, "id", None) is not None
                else [None] * len(xyxy)
            )

            for (x1, y1, x2, y2), c, cf, pid in zip(xyxy, clss, confs, ids):
                if c not in TRACK_CLASSES:
                    continue
                if (c == PERSON_ID and cf < CONF_PERSON) or (c == TAKEBOOK_ID and cf < CONF_TAKE):
                    continue
                if not roi_filter((float(x1), float(y1), float(x2), float(y2))):
                    continue

                box_coords = (int(x1), int(y1), int(x2), int(y2))
                pid_int = int(pid) if pid is not None else None

                if c == PERSON_ID:
                    persons_in_frame.append((box_coords, float(cf), pid_int))
                elif c == TAKEBOOK_ID:
                    takebooks_in_frame.append((box_coords, float(cf), pid_int))

            # 紀錄 person
            for p_box, p_conf, p_pid in persons_in_frame:
                final_detections.append((p_box, PERSON_ID, p_conf, p_pid))
                if p_pid is not None:
                    hour_seen_ids[(date_str, hour_int)]["person"].add(p_pid)

            # 紀錄 takebook 並關聯到 person
            for t_box, t_conf, t_pid in takebooks_in_frame:
                associated_person_pid = None
                for p_box, _, p_pid in persons_in_frame:
                    if calculate_iou(t_box, p_box) >= MIN_IOU_OVERLAP:
                        associated_person_pid = p_pid
                        break

                if associated_person_pid is not None:
                    # 只記一次「這個人拿過書」
                    if not person_tracks.get(associated_person_pid, {}).get("has_taken", False):
                        final_detections.append((t_box, TAKEBOOK_ID, t_conf, t_pid))
                        if associated_person_pid is not None:
                            hour_seen_ids[(date_str, hour_int)]["takebook"].add(associated_person_pid)
                        person_tracks.setdefault(associated_person_pid, {})["has_taken"] = True
                        det_total += 1

        # ✅ 若啟用影片輸出才畫框 + 寫出
        if ENABLE_VIDEO_OUTPUT:
            out_img = full_frame.copy()
            if DRAW_ROI_BOX:
                cv2.rectangle(out_img, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
                cv2.putText(out_img, "ROI", (rx + 6, ry + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

            COLORS = {}
            if PERSON_ID is not None:
                COLORS[PERSON_ID] = (0, 200, 0)
            if TAKEBOOK_ID is not None:
                COLORS[TAKEBOOK_ID] = (0, 255, 255)

            for (x1, y1, x2, y2), cls_id, conf, pid in final_detections:
                color = COLORS.get(cls_id, (0, 0, 255))
                cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)

                cls_name = NAME_MAP.get(int(cls_id), str(cls_id))
                pid_txt  = f"ID {pid}" if pid is not None else "ID -"
                label    = f"{pid_txt} {cls_name} {conf:.2f}"

                (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tx1, ty1 = x1, max(0, y1 - th - 6)
                tx2, ty2 = x1 + tw + 6, y1
                cv2.rectangle(out_img, (tx1, ty1), (tx2, ty2), color, -1)
                cv2.putText(out_img, label, (x1 + 3, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

            writer.write(out_img)

        frame_idx += 1
        progress_bar.update(1)

except KeyboardInterrupt:
    p("\n🛑 偵測被使用者中止 (Ctrl+C)，將依目前進度輸出統計…")

finally:
    cap.release()
    if ENABLE_VIDEO_OUTPUT and writer is not None:
        writer.release()
    progress_bar.close()
    p(f"✅ 全部完成（包含中止情況），共偵測 {det_total} 次 takebook。")

    # ========= 無論正常結束或 Ctrl+C，都會輸出 Excel =========
    if ENABLE_EXCEL_OUTPUT:
        try:
            rows = []
            for (date_str, hour_int), seen_ids in sorted(hour_seen_ids.items()):
                rows.append({
                    "date":     str(date_str),
                    "hour":     int(hour_int),
                    "person":   len(seen_ids.get("person", set())),
                    "takebook": len(seen_ids.get("takebook", set())),
                })

            if not rows:
                rows = [{
                    "date": INITIAL_TIME.date().isoformat(),
                    "hour": INITIAL_TIME.hour,
                    "person": 0,
                    "takebook": 0
                }]

            df_hour = pd.DataFrame(rows, columns=["date", "hour", "person", "takebook"])
            write_excel_locked(XLSX_PATH, df_hour)
            p(f"📘 已更新 Excel：{XLSX_PATH}")
        except Exception:
            traceback.print_exc()
            p("? 寫入 Excel 失敗")
