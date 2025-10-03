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
import msvcrt  # ★ Windows 檔案鎖

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

# ========= 互動輸入（時間） =========
p("enter year"); year  = int(input().strip())
p("enter month"); month = int(input().strip())
p("enter day");   day   = int(input().strip())
p("enter hour (0-23)"); hour = int(input().strip())
minute, second = 0, 0
INITIAL_TIME = datetime(year, month, day, hour, minute, second)
p(f"初始時間：{INITIAL_TIME:%Y-%m-%d %H:%M:%S}")

# ========= 路徑設定（請改為你的 Windows 路徑） =========
OUTPUT_DIR   = Path(r"C:\Users\user\Desktop\test\test\out").resolve()
VIDEO_OUT_DIR= OUTPUT_DIR / "videos"
MODEL_PATH   = r"C:\Users\user\Desktop\test\test\best.pt"
VIDEO_PATH   = r"D:\ftp server\140.128.95.222\2025-09-17\001\91708.dav"  # ←確認實際檔名

for d in [OUTPUT_DIR, VIDEO_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
if not os.path.isfile(MODEL_PATH): p(f"? 找不到模型：{MODEL_PATH}"); sys.exit(1)
if not os.path.isfile(VIDEO_PATH): p(f"? 找不到影片：{VIDEO_PATH}"); sys.exit(1)
p(f"輸出根目錄：{OUTPUT_DIR}")

# ========= 影片資訊 =========
p("開啟影片…")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    p("? 無法開啟影片，請確認檔案/編解碼器"); sys.exit(1)

raw_fps = cap.get(cv2.CAP_PROP_FPS)
fps = raw_fps if (raw_fps and raw_fps > 1 and not math.isnan(raw_fps)) else 20.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames <= 0: total_frames = -1
p(f"影片：{VIDEO_PATH}\n尺寸={width}x{height}, fps?{fps:.2f}（來源:{raw_fps}）, 總幀={total_frames if total_frames>0 else '未知'}")

# ========= ROI（僅作為「關注區」與可視化，不裁切） =========
USE_ROI_FILTER = True
DRAW_ROI_BOX   = True
ROI_XYWH = (216, 276, 1805, 972)
rx, ry, rw, rh = clamp_roi_to_frame(*ROI_XYWH, width, height)
if (rx, ry, rw, rh) != ROI_XYWH:
    p(f"?ROI 超出範圍，已調整：{ROI_XYWH} -> {(rx,ry,rw,rh)}")
ROI_XYWH = (rx, ry, rw, rh)
p(f"關注 ROI：{ROI_XYWH}")

def roi_overlap_ok(box, roi_xywh, min_iou=0.05, min_inter_area_ratio=0.30):
    rx, ry, rw, rh = roi_xywh
    x1, y1, x2, y2 = box
    rx2, ry2 = rx + rw, ry + rh
    ix1, iy1 = max(x1, rx), max(y1, ry)
    ix2, iy2 = min(x2, rx2), min(y2, ry2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0: return False
    area_box = max(1.0, (x2 - x1) * (y2 - y1))
    area_roi = float(rw * rh)
    iou = inter / (area_box + area_roi - inter)
    inter_ratio = inter / area_box
    return (iou >= min_iou) or (inter_ratio >= min_inter_area_ratio)

roi_filter = (lambda b: True) if not USE_ROI_FILTER else (lambda b: roi_overlap_ok(b, ROI_XYWH))

# ========= 載入模型與類別對應 =========
p("載入模型…")
try:
    model = YOLO(MODEL_PATH)
    p("? 模型載入 OK")
except Exception:
    traceback.print_exc(); p("? 模型載入失敗"); sys.exit(1)

def build_name_map(names):
    if isinstance(names, dict): return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))
NAME_MAP = build_name_map(getattr(model, "names", {}))
p(f"Model classes: {NAME_MAP}")

def find_id_by_names(candidates):
    for k, v in NAME_MAP.items():
        n = str(v).lower()
        for c in candidates:
            if c in n: return k
    return None

PERSON_ID   = find_id_by_names(["person", "people", "human", "人"])
TAKEBOOK_ID = find_id_by_names(["takebook", "book", "holding book", "take_book", "拿書", "書"])
TRACK_CLASSES = [i for i in [PERSON_ID, TAKEBOOK_ID] if i is not None]
if not TRACK_CLASSES:
    raise RuntimeError("找不到 person/takebook 類別，請確認模型的 names。")
p(f"[Class IDs] PERSON_ID={PERSON_ID}, TAKEBOOK_ID={TAKEBOOK_ID}")

# ========= 推論/過濾參數（類別門檻） =========
IOU_NMS       = 0.45
CONF_PERSON   = 0.60
CONF_TAKE     = 0.15
BASE_CONF     = min(CONF_PERSON, CONF_TAKE)

# ========= 追蹤器設定（BoT-SORT，Ultralytics 內建，含 ReID） =========
TRACKER_CFG = str(Path(ul.__file__).parent / "cfg" / "trackers" / "botsort.yaml")

# ========= 統計用結構（用追蹤器給的 pid，不再自寫 IoU） =========
K_PERSON = 20
M_TAKE   = 10
DEBUG_PRINT = True
DEBUG_FIRST_N_FRAMES = 200

person_tracks = {}   # {pid: {"hits_person": int, "has_taken": bool, "take_hits": int}}
hour_buckets = defaultdict(lambda: Counter())
hour_seen_ids = defaultdict(lambda: {"person": set(), "takebook": set()})
hour_conf = defaultdict(lambda: {"person_sum": 0.0, "person_n": 0, "take_sum": 0.0, "take_n": 0})

# ========= 影片輸出：XVID/MJPG → mp4v 回退 =========
base_name   = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)
avi_path    = str((VIDEO_OUT_DIR / f"output_tracking_{base_name}.avi").resolve())
mp4_path    = str((VIDEO_OUT_DIR / f"output_tracking_{base_name}.mp4").resolve())
writer      = None
used_codec  = None
used_path   = None

def try_open_writer(out_path, fourcc_str, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    vw = cv2.VideoWriter(out_path, fourcc, float(fps), size)
    return vw if vw.isOpened() else None

def init_video_writer(size):
    global writer, used_codec, used_path
    for fourcc_str, path in (("XVID", avi_path), ("MJPG", avi_path)):
        vw = try_open_writer(path, fourcc_str, fps, size)
        if vw is not None:
            writer, used_codec, used_path = vw, fourcc_str, path
            p(f"? VideoWriter 使用 {fourcc_str} -> {path}")
            return
    vw = try_open_writer(mp4_path, "mp4v", fps, size)
    if vw is not None:
        writer, used_codec, used_path = vw, "mp4v", mp4_path
        p(f"? VideoWriter 使用 mp4v -> {mp4_path}")
        return
    raise RuntimeError("VideoWriter 初始化失敗：請確認系統有可用編碼器/寫入權限。")

# ========= 主迴圈（★ 使用 YOLO .track，含 ReID） =========
p("開始處理（使用 YOLO tracking + BoT-SORT ReID）…")
frame_idx = 0
last_processed_frame = None
det_total = 0
last_progress = -1

try:
    while True:
        ok, full_frame = cap.read()
        if not ok or full_frame is None:
            if frame_idx == 0:
                p("? 無法讀取任何影格（影片壞檔或編碼不支援）"); sys.exit(1)
            break

        if writer is None:
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
        det_boxes = []  # (box, cls_id, conf, pid)

        if res is not None and hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            b = res.boxes
            if hasattr(b, "id") and b.id is not None:
                ids = b.id.detach().cpu().numpy().astype(int).tolist()
            else:
                ids = [None] * len(b)

            num = len(b); det_total += num
            for i in range(num):
                pid = ids[i]
                if pid is None:
                    continue

                conf   = float(b.conf[i].item()) if hasattr(b.conf, "shape") else float(b.conf[i])
                cls_id = int(b.cls[i].item()) if hasattr(b.cls, "shape") else int(b.cls[i])

                if cls_id == PERSON_ID:
                    if conf < CONF_PERSON: continue
                elif TAKEBOOK_ID is not None and cls_id == TAKEBOOK_ID:
                    if conf < CONF_TAKE: continue
                else:
                    continue

                x1, y1, x2, y2 = b.xyxy[i].detach().cpu().numpy().tolist()
                box = [float(x1), float(y1), float(x2), float(y2)]
                if not roi_filter(box):
                    continue

                key_conf = (date_str, hour_int)
                if cls_id == PERSON_ID:
                    hour_conf[key_conf]["person_sum"] += conf
                    hour_conf[key_conf]["person_n"]   += 1
                elif TAKEBOOK_ID is not None and cls_id == TAKEBOOK_ID:
                    hour_conf[key_conf]["take_sum"] += conf
                    hour_conf[key_conf]["take_n"]   += 1

                det_boxes.append((box, cls_id, conf, pid))

        if DEBUG_PRINT and frame_idx < DEBUG_FIRST_N_FRAMES:
            for box, cls_id, conf, pid in det_boxes:
                x1, y1, x2, y2 = map(int, box)
                label = NAME_MAP.get(cls_id, cls_id)
                p(f"[DEBUG f{frame_idx}] det: {label} cls={cls_id} conf={conf:.2f} pid={pid} box=({x1},{y1},{x2},{y2})")

        dkey = (date_str, hour_int)
        for box, cls_id, conf, pid in det_boxes:
            if pid not in person_tracks:
                person_tracks[pid] = {"hits_person": 0, "has_taken": False, "take_hits": 0}

            if cls_id == PERSON_ID:
                person_tracks[pid]["hits_person"] = min(person_tracks[pid]["hits_person"] + 1, K_PERSON)
                if (person_tracks[pid]["hits_person"] >= K_PERSON) and (pid not in hour_seen_ids[dkey]["person"]):
                    hour_seen_ids[dkey]["person"].add(pid)
                    hour_buckets[dkey]["person"] += 1
                    if DEBUG_PRINT:
                        p(f"[COUNT] +person @ {date_str} {hour_int:02d}:00 pid={pid}")

            if TAKEBOOK_ID is not None and cls_id == TAKEBOOK_ID:
                if not person_tracks[pid]["has_taken"]:
                    person_tracks[pid]["take_hits"] = min(person_tracks[pid]["take_hits"] + 1, M_TAKE)
                    if person_tracks[pid]["take_hits"] >= M_TAKE:
                        if pid not in hour_seen_ids[dkey]["takebook"]:
                            hour_seen_ids[dkey]["takebook"].add(pid)
                            hour_buckets[dkey]["takebook"] += 1
                            if DEBUG_PRINT:
                                p(f"[COUNT] +takebook @ {date_str} {hour_int:02d}:00 pid={pid}")
                        person_tracks[pid]["has_taken"] = True

        out_img = full_frame.copy()
        for box, cls_id, conf, pid in det_boxes:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if cls_id == PERSON_ID else (0, 128, 255)
            cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
            tag = "P" if cls_id == PERSON_ID else "T"
            text = f"{tag}#{pid} {conf:.2f}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(out_img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out_img, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)

        if DRAW_ROI_BOX:
            cv2.rectangle(out_img, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
            cv2.putText(out_img, "ROI", (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        writer.write(out_img)
        last_processed_frame = out_img

        frame_idx += 1
        if total_frames > 0 and frame_idx % 200 == 0:
            pct = int(frame_idx * 100 / total_frames)
            if pct != last_progress:
                p(f"[進度] {frame_idx}/{total_frames} ({pct}%) | 偵測總數(含未過濾)：{det_total}")
                last_progress = pct

finally:
    cap.release()
    if writer is not None:
        writer.release()

p(f"? 已輸出影片：{used_path}（codec={used_codec}）")

# ========= 輸出最後一幀快照 =========
if last_processed_frame is not None:
    snapshot_path = str((OUTPUT_DIR / f"snapshot_{base_name}.jpg").resolve())
    try:
        cv2.imwrite(snapshot_path, last_processed_frame)
        p(f"已輸出快照：{snapshot_path}")
    except Exception:
        traceback.print_exc(); p("? 儲存快照失敗")
else:
    p("沒有可用的最後一幀，未輸出快照")

# ========= 匯出逐小時 Excel（Windows 檔案鎖） =========
XLSX_PATH = (OUTPUT_DIR / "excel" / "all_hourly.xlsx").resolve()

def write_excel_locked(xlsx_path: Path, df_new: pd.DataFrame):
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(xlsx_path) + ".lock"
    with open(lock_path, "w") as lk:
        try:
            msvcrt.locking(lk.fileno(), msvcrt.LK_LOCK, 1)  # ★ Windows 獨佔鎖
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

            df_all = (df_all.sort_values(["date", "hour"])
                             .drop_duplicates(subset=["date", "hour"], keep="last"))

            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_all.to_excel(writer, sheet_name="hourly", index=False)
        finally:
            lk.flush(); os.fsync(lk.fileno())
            msvcrt.locking(lk.fileno(), msvcrt.LK_UNLCK, 1)

try:
    rows = []
    for (date_str, hour_int), cnt in sorted(hour_buckets.items()):
        rows.append({
            "date":     str(date_str),
            "hour":     int(hour_int),
            "person":   int(cnt.get("person", 0)),
            "takebook": int(cnt.get("takebook", 0)),
        })
    if not rows:
        rows = [{"date": INITIAL_TIME.date().isoformat(),
                 "hour": INITIAL_TIME.hour, "person": 0, "takebook": 0}]
    df_hour = pd.DataFrame(rows, columns=["date", "hour", "person", "takebook"])
    write_excel_locked(XLSX_PATH, df_hour)
    p(f"已更新 Excel：{XLSX_PATH}")
except Exception:
    traceback.print_exc()
    p("? 寫入 Excel 失敗"); sys.exit(1)

p("全部完成 ✅")
