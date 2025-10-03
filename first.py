# -*- coding: utf-8 -*-

from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import os, sys, tempfile, traceback
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import torch
import uuid
import math

# ========= 加速與裝置設定 =========
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

DEVICE = "0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
HALF   = (DEVICE != "cpu")
IMGSZ  = 960
BATCH  = 128
print(f"[Init] device={DEVICE}, half={HALF}, imgsz={IMGSZ}, batch={BATCH}", flush=True)

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

# ========= 路徑設定 =========
OUTPUT_DIR = Path(r"C:\Users\user\Desktop\test\test\out").resolve()
VIDEO_OUT_DIR = OUTPUT_DIR / "videos"
MODEL_PATH = r"C:\Users\user\Desktop\test\test\best.pt"
VIDEO_PATH = r"D:\22.mp4"  # ←確認實際檔名

for d in [OUTPUT_DIR, VIDEO_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
if not os.path.isfile(MODEL_PATH): p(f"✗ 找不到模型：{MODEL_PATH}"); sys.exit(1)
if not os.path.isfile(VIDEO_PATH): p(f"✗ 找不到影片：{VIDEO_PATH}"); sys.exit(1)
p(f"輸出根目錄：{OUTPUT_DIR}")

# ========= 影片資訊 =========
p("開啟影片…")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    p("✗ 無法開啟影片，請確認 ffmpeg/gstreamer 已安裝且編解碼可用"); sys.exit(1)

raw_fps = cap.get(cv2.CAP_PROP_FPS)
fps = raw_fps if (raw_fps and raw_fps > 1 and not math.isnan(raw_fps)) else 20.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames <= 0:
    total_frames = -1
p(f"影片：{VIDEO_PATH}\n尺寸={width}x{height}, fps≈{fps:.2f}（來源:{raw_fps}）, 總幀={total_frames if total_frames>0 else '未知'}")

# ========= ROI（僅作為「關注區」與可視化，不裁切） =========
# 三個開關：依需求調整
USE_ROI_FILTER = True   # True：只統計/追蹤 ROI 內之偵測；False：全畫面
DRAW_ROI_BOX   = True   # 是否在畫面上畫出 ROI 框
NO_CROP        = True   # True：不裁切（整張影像推論）；False：會裁切（本版預設 True）

ROI_XYWH = (216, 276, 1805, 972)
rx, ry, rw, rh = clamp_roi_to_frame(*ROI_XYWH, width, height)
if (rx, ry, rw, rh) != ROI_XYWH:
    p(f"⚠ROI 超出範圍，已調整：{ROI_XYWH} -> {(rx,ry,rw,rh)}")
ROI_XYWH = (rx, ry, rw, rh)
p(f"關注 ROI：{ROI_XYWH}")

def roi_overlap_ok(box, roi_xywh, min_iou=0.05, min_inter_area_ratio=0.30):
    """以 IOU 與重疊比例判斷偵測框是否足夠位於 ROI 內"""
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

# 若不使用 ROI 過濾就直接通過
roi_filter = (lambda b: True) if not USE_ROI_FILTER else (lambda b: roi_overlap_ok(b, ROI_XYWH))

# ========= 載入模型與類別對應 =========
p("載入模型…")
try:
    model = YOLO(MODEL_PATH)
    p("✓ 模型載入 OK")
except Exception:
    traceback.print_exc(); p("✗ 模型載入失敗"); sys.exit(1)

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
            if c in n: return k
    return None

PERSON_ID   = find_id_by_names(["person", "people", "human", "人"])
TAKEBOOK_ID = find_id_by_names(["takebook", "book", "holding book", "take_book", "拿書", "書"])
TRACK_CLASSES = [i for i in [PERSON_ID, TAKEBOOK_ID] if i is not None]
if not TRACK_CLASSES:
    raise RuntimeError("找不到 person/takebook 類別，請確認模型的 names。")
p(f"[Class IDs] PERSON_ID={PERSON_ID}, TAKEBOOK_ID={TAKEBOOK_ID}")

# ========= 推論/過濾參數（只用類別專屬信心值） =========
IOU_NMS    = 0.4
MIN_AREA   = 40 * 40
MAX_AREA_RATIO = 0.5
frame_area = float(width * height)

# 類別專屬門檻（僅此）
CONF_PERSON = 0.60
CONF_TAKE   = 0.15

# 連續幀門檻（可調）
K_PERSON = 20
M_TAKE   = 10

# 偵錯輸出設定
DEBUG_PRINT = True
DEBUG_FIRST_N_FRAMES = 200  # 只印前 200 幀

# ========= 追蹤 & 計數 =========
def iou(a, c):
    xA=max(a[0],c[0]); yA=max(a[1],c[1])
    xB=min(a[2],c[2]); yB=min(a[3],c[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    if inter<=0: return 0.0
    areaA=max(1,a[2]-a[0])*max(1,a[3]-a[1])
    areaC=max(1,c[2]-c[0])*max(1,c[3]-c[1])
    return inter/(areaA+areaC-inter)

def track_avg_box(track_boxes):
    arr = np.array(track_boxes, dtype=float)
    return arr.mean(axis=0).tolist()

def match_to_tracks(box, tracks_dict, iou_thresh=0.25):
    best_id, best_iou = None, 0.0
    for pid, data in tracks_dict.items():
        if not data["boxes"]: continue
        v = iou(box, track_avg_box(data["boxes"]))
        if v > best_iou: best_iou, best_id = v, pid
    return best_id if best_iou >= iou_thresh else None

# person_tracks 結構擴充 hits_person / take_hits
person_tracks = {}   # {pid: {"boxes":[...], "hits_person": int, "has_taken": bool, "take_hits": int}}
next_person_id = 0
hour_buckets = defaultdict(lambda: Counter())
hour_seen_ids = defaultdict(lambda: {"person": set(), "takebook": set()})

# 逐小時平均信心值累積器（如需用得到）
hour_conf = defaultdict(lambda: {"person_sum": 0.0, "person_n": 0,
                                 "take_sum":   0.0, "take_n":   0})

# ========= 輸出 AVI（XVID -> MJPG 後援） =========
base_name = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
avi_path = str((VIDEO_OUT_DIR / f"output_tracking_{base_name}.avi").resolve())
writer = None  # 延後到拿到第一幀才建立

def init_avi_writer_or_fail(out_path, fps, size):
    """先嘗試 XVID，不行再用 MJPG。"""
    for fourcc_str in ("XVID", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(out_path, fourcc, float(fps), size)
        if vw.isOpened():
            p(f"✓ VideoWriter 使用 {fourcc_str} 成功 -> {out_path}")
            return vw, fourcc_str
        else:
            vw.release()
    raise RuntimeError("VideoWriter 初始化失敗：系統無法用 XVID 或 MJPG 輸出 AVI。請安裝對應編碼器。")

# ========= 主迴圈（批次推論 + 逐幀繪製/寫檔） =========
p("開始處理…（每 200 幀輸出一次進度）")
frame_idx = 0
det_total = 0
last_progress = -1
last_processed_frame = None

batch_frames = []   # ★ 直接放整張 frame（不裁切）
batch_full_frames = []
batch_meta = []

try:
    while True:
        ok, full_frame = cap.read()
        if not ok or full_frame is None:
            if frame_idx == 0:
                p("✗ 無法讀取任何影格（影片壞檔或編碼不支援）"); sys.exit(1)
            break

        # 第一次拿到幀時建立 writer（AVI）
        if writer is None:
            try:
                writer, used_codec = init_avi_writer_or_fail(avi_path, fps, (full_frame.shape[1], full_frame.shape[0]))
            except Exception:
                traceback.print_exc()
                p("✗ 建立 AVI 失敗，請確認系統有 XVID/MJPG 編碼器（opencv videoio）。")
                sys.exit(1)

        timestamp = INITIAL_TIME + timedelta(seconds=frame_idx / float(fps))
        date_str = timestamp.date().isoformat()
        hour_int = timestamp.hour

        # ★ 不裁切，直接整張推論
        batch_frames.append(full_frame)
        batch_full_frames.append(full_frame)
        batch_meta.append((date_str, hour_int))

        # 滿批或到尾端就送進模型
        if len(batch_frames) >= BATCH:
            with torch.inference_mode():
                results_list = model.predict(
                    source=batch_frames, conf=0.0, iou=IOU_NMS,
                    imgsz=IMGSZ, device=DEVICE, half=HALF, verbose=False, stream=False
                )

            for res, (dstr, hr), full_img in zip(results_list, batch_meta, batch_full_frames):
                det_boxes = []  # (box, cls_id, conf, label)

                if res and hasattr(res, "boxes") and res.boxes is not None:
                    b = res.boxes
                    num = len(b); det_total += num
                    for i in range(num):
                        conf = float(b.conf[i].item()) if hasattr(b.conf, "shape") else float(b.conf[i])
                        cls_id = int(b.cls[i].item()) if hasattr(b.cls, "shape") else int(b.cls[i])

                        # 只處理我們關心的類別（per-class 門檻）
                        if cls_id == PERSON_ID:
                            thr = CONF_PERSON
                        elif TAKEBOOK_ID is not None and cls_id == TAKEBOOK_ID:
                            thr = CONF_TAKE
                        else:
                            continue
                        if conf < thr:
                            continue

                        # ★ 直接取模型座標（全畫面），不做 rx/ry 偏移
                        x1, y1, x2, y2 = b.xyxy[i].detach().cpu().numpy().tolist()

                        w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
                        area = w * h
                        if area < MIN_AREA or area > frame_area * MAX_AREA_RATIO:
                            continue

                        box = [float(x1), float(y1), float(x2), float(y2)]
                        if not roi_filter(box):
                            continue

                        label = str(NAME_MAP.get(cls_id, cls_id))
                        det_boxes.append((box, cls_id, conf, label))

                        # 累積平均信心值（如需要）
                        if cls_id == PERSON_ID:
                            hour_conf[(dstr, hr)]["person_sum"] += conf
                            hour_conf[(dstr, hr)]["person_n"]   += 1
                        elif TAKEBOOK_ID is not None and cls_id == TAKEBOOK_ID:
                            hour_conf[(dstr, hr)]["take_sum"] += conf
                            hour_conf[(dstr, hr)]["take_n"]   += 1

                # 偵錯輸出
                if DEBUG_PRINT and frame_idx < DEBUG_FIRST_N_FRAMES:
                    for box, cls_id, conf, label in det_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        p(f"[DEBUG f{frame_idx}] det: {label} cls={cls_id} conf={conf:.2f} box=({x1},{y1},{x2},{y2})")

                # 追蹤＋計數，同時準備繪製項目（含 PID）
                draw_items = []   # [(box, cls_id, conf, pid)]

                # 人（PERSON）
                for box, cls_id, conf, label in det_boxes:
                    if cls_id != PERSON_ID:
                        continue
                    pid = match_to_tracks(box, person_tracks, iou_thresh=0.35)
                    if pid is None:
                        pid = next_person_id
                        person_tracks[pid] = {"boxes": [box], "hits_person": 1, "has_taken": False, "take_hits": 0}
                        next_person_id += 1
                    else:
                        person_tracks[pid]["boxes"].append(box)
                        if len(person_tracks[pid]["boxes"]) > 2:
                            person_tracks[pid]["boxes"].pop(0)
                        person_tracks[pid]["hits_person"] = min(person_tracks[pid].get("hits_person", 0) + 1, K_PERSON)

                    key = (dstr, hr)
                    if person_tracks[pid]["hits_person"] >= K_PERSON and pid not in hour_seen_ids[key]["person"]:
                        hour_seen_ids[key]["person"].add(pid)
                        hour_buckets[key]["person"] += 1
                        if DEBUG_PRINT:
                            p(f"[COUNT] +person @ {dstr} {hr:02d}:00 pid={pid}")

                    draw_items.append((box, cls_id, conf, pid))

                # 拿書（TAKEBOOK）
                for box, cls_id, conf, label in det_boxes:
                    if TAKEBOOK_ID is None or cls_id != TAKEBOOK_ID:
                        continue
                    pid = match_to_tracks(box, person_tracks, iou_thresh=0.25)
                    if pid is not None and not person_tracks[pid]["has_taken"]:
                        person_tracks[pid]["take_hits"] = min(person_tracks[pid].get("take_hits", 0) + 1, M_TAKE)
                        if person_tracks[pid]["take_hits"] >= M_TAKE:
                            key = (dstr, hr)
                            if pid not in hour_seen_ids[key]["takebook"]:
                                hour_seen_ids[key]["takebook"].add(pid)
                                hour_buckets[key]["takebook"] += 1
                            person_tracks[pid]["has_taken"] = True
                    if pid is not None:
                        draw_items.append((box, cls_id, conf, pid))

                # 繪製（含 PID）
                for box, cls_id, conf, pid in draw_items:
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 255, 0) if cls_id == PERSON_ID else (0, 128, 255)
                    cv2.rectangle(full_img, (x1, y1), (x2, y2), color, 2)
                    tag = "P" if cls_id == PERSON_ID else "T"
                    text = f"{tag}#{pid} {conf:.2f}"
                    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(full_img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(full_img, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)

                if DRAW_ROI_BOX:
                    cv2.rectangle(full_img, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
                    cv2.putText(full_img, "ROI", (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                writer.write(full_img)
                last_processed_frame = full_img

            batch_frames.clear(); batch_full_frames.clear(); batch_meta.clear()

        frame_idx += 1
        if total_frames > 0 and frame_idx % 200 == 0:
            pct = int(frame_idx * 100 / total_frames)
            if pct != last_progress:
                p(f"[進度] {frame_idx}/{total_frames} ({pct}%) | 偵測總數: {det_total}")
                last_progress = pct

    # ===== 尾批 =====
    if batch_frames:
        with torch.inference_mode():
            results_list = model.predict(
                source=batch_frames, conf=0.0, iou=IOU_NMS,
                imgsz=IMGSZ, device=DEVICE, half=HALF, verbose=False, stream=False
            )
        for res, (dstr, hr), full_img in zip(results_list, batch_meta, batch_full_frames):
            det_boxes = []
            if res and hasattr(res, "boxes") and res.boxes is not None:
                b = res.boxes
                num = len(b); det_total += num
                for i in range(num):
                    conf = float(b.conf[i].item()) if hasattr(b.conf, "shape") else float(b.conf[i])
                    cls_id = int(b.cls[i].item()) if hasattr(b.cls, "shape") else int(b.cls[i])

                    if cls_id == PERSON_ID:
                        thr = CONF_PERSON
                    elif TAKEBOOK_ID is not None and cls_id == TAKEBOOK_ID:
                        thr = CONF_TAKE
                    else:
                        continue
                    if conf < thr:
                        continue

                    x1, y1, x2, y2 = b.xyxy[i].detach().cpu().numpy().tolist()

                    w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
                    area = w * h
                    if area < MIN_AREA or area > frame_area * MAX_AREA_RATIO:
                        continue

                    box = [float(x1), float(y1), float(x2), float(y2)]
                    if not roi_filter(box):
                        continue
                    label = str(NAME_MAP.get(cls_id, cls_id))
                    det_boxes.append((box, cls_id, conf, label))

                    # 累積平均信心值（如需要）
                    if cls_id == PERSON_ID:
                        hour_conf[(dstr, hr)]["person_sum"] += conf
                        hour_conf[(dstr, hr)]["person_n"]   += 1
                    elif TAKEBOOK_ID is not None and cls_id == TAKEBOOK_ID:
                        hour_conf[(dstr, hr)]["take_sum"] += conf
                        hour_conf[(dstr, hr)]["take_n"]   += 1

            # 尾批 追蹤＋計數＋繪製
            draw_items = []

            for box, cls_id, conf, label in det_boxes:
                if cls_id == PERSON_ID:
                    pid = match_to_tracks(box, person_tracks, iou_thresh=0.35)
                    if pid is None:
                        pid = next_person_id
                        person_tracks[pid] = {"boxes": [box], "hits_person": 1, "has_taken": False, "take_hits": 0}
                        next_person_id += 1
                    else:
                        person_tracks[pid]["boxes"].append(box)
                        if len(person_tracks[pid]["boxes"]) > 10:
                            person_tracks[pid]["boxes"].pop(0)
                        person_tracks[pid]["hits_person"] = min(person_tracks[pid].get("hits_person", 0) + 1, K_PERSON)

                    key = (dstr, hr)
                    if person_tracks[pid]["hits_person"] >= K_PERSON and pid not in hour_seen_ids[key]["person"]:
                        hour_seen_ids[key]["person"].add(pid)
                        hour_buckets[key]["person"] += 1
                        if DEBUG_PRINT:
                            p(f"[COUNT] +person @ {dstr} {hr:02d}:00 pid={pid}")

                    draw_items.append((box, cls_id, conf, pid))

            for box, cls_id, conf, label in det_boxes:
                if TAKEBOOK_ID is not None and cls_id == TAKEBOOK_ID:
                    pid = match_to_tracks(box, person_tracks, iou_thresh=0.25)
                    if pid is not None and not person_tracks[pid]["has_taken"]:
                        person_tracks[pid]["take_hits"] = min(person_tracks[pid].get("take_hits", 0) + 1, M_TAKE)
                        if person_tracks[pid]["take_hits"] >= M_TAKE:
                            key = (dstr, hr)
                            if pid not in hour_seen_ids[key]["takebook"]:
                                hour_seen_ids[key]["takebook"].add(pid)
                                hour_buckets[key]["takebook"] += 1
                                if DEBUG_PRINT:
                                    p(f"[COUNT] +takebook @ {dstr} {hr:02d}:00 pid={pid}")
                            person_tracks[pid]["has_taken"] = True
                    if pid is not None:
                        draw_items.append((box, cls_id, conf, pid))

            for box, cls_id, conf, pid in draw_items:
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0) if cls_id == PERSON_ID else (0, 128, 255)
                cv2.rectangle(full_img, (x1, y1), (x2, y2), color, 2)
                tag = "P" if cls_id == PERSON_ID else "T"
                text = f"{tag}#{pid} {conf:.2f}"
                (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                cv2.rectangle(full_img, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), color, -1)
                cv2.putText(full_img, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 2)

            if DRAW_ROI_BOX:
                cv2.rectangle(full_img, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
                cv2.putText(full_img, "ROI", (rx + 5, ry + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

            writer.write(full_img)
            last_processed_frame = full_img

finally:
    cap.release()
    if writer is not None:
        writer.release()

p(f"✅ 已輸出影片(AVI)：{avi_path}")

# ========= 輸出最後一幀快照 =========
if last_processed_frame is not None:
    snapshot_path = str((OUTPUT_DIR / f"snapshot_{base_name}.jpg").resolve())
    try:
        cv2.imwrite(snapshot_path, last_processed_frame)
        p(f"已輸出快照：{snapshot_path}")
    except Exception:
        traceback.print_exc(); p("✗ 儲存快照失敗")
else:
    p("沒有可用的最後一幀，未輸出快照")

# ========= 匯出逐小時 Excel（單檔、單表；含鎖、去重併檔） =========
XLSX_PATH = (OUTPUT_DIR / "excel" / "all_hourly.xlsx").resolve()

def write_excel_locked(xlsx_path: Path, df_new: pd.DataFrame):
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(xlsx_path) + ".lock"
    with open(lock_path, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)

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
        df_all["date"] = df_all["date"].astype(str)
        df_all["hour"] = df_all["hour"].astype(int)
        df_all["person"] = df_all["person"].astype(int)
        df_all["takebook"] = df_all["takebook"].astype(int)

        df_all = (
            df_all.sort_values(["date", "hour"])
                  .drop_duplicates(subset=["date", "hour"], keep="last")
        )

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_all.to_excel(writer, sheet_name="hourly", index=False)

        lk.flush(); os.fsync(lk.fileno())
        fcntl.flock(lk, fcntl.LOCK_UN)

try:
    rows = []
    for (date_str, hour_int), cnt in sorted(hour_buckets.items()):
        rows.append({
            "date":     str(date_str),
            "hour":     int(hour_int),
            "person":   int(cnt.get("person", 0)),
            "takebook": int(cnt.get("takebook", 0)),
        })

    # ★ 若 rows 為空，補一筆 0 資料
    if not rows:
        rows = [{
            "date": INITIAL_TIME.date().isoformat(),
            "hour": INITIAL_TIME.hour,
            "person": 0,
            "takebook": 0,
        }]

    df_hour = pd.DataFrame(rows, columns=["date", "hour", "person", "takebook"])
    write_excel_locked(XLSX_PATH, df_hour)
    p(f"已更新 Excel：{XLSX_PATH}")

except Exception:
    traceback.print_exc()
    p("✗ 寫入 Excel 失敗"); sys.exit(1)

p("全部完成 ✅")
