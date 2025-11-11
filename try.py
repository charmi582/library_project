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
from tqdm import tqdm # ✅ tqdm 進度條

# ========= 可開關設定 =========
ENABLE_VIDEO_OUTPUT = True    # ✅ 是否輸出影片
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

# ✨ [新功能] 從檔名自動解析時間
def parse_time_from_filename(filename_stem: str, year: int):
    """
    從檔名 (不含副檔名) 解析時間。
    範例: '091709' (MMDDHH)
    """
    try:
        if len(filename_stem) != 6:
            raise ValueError(f"檔名 '{filename_stem}' 不是 6 位數 (MMDDHH)")
        
        month = int(filename_stem[0:2])
        day   = int(filename_stem[2:4])
        hour  = int(filename_stem[4:6])
        
        return datetime(year, month, day, hour, 0, 0)
    except Exception as e:
        p(f"!! 無法從檔名 '{filename_stem}' 解析時間: {e}")
        p("   將使用 1970/1/1 00:00:00 作為備用時間。")
        return datetime(1970, 1, 1, 0, 0, 0) # 回傳一個預設值

# ========= 互動輸入 (僅問一次年份) =========
p("=== 自動批次處理排程 ===")
p("請輸入影片的「年份」(例如 2025)，程式將會自動解析檔名中的 月/日/時"); 
BASE_YEAR = int(input().strip())
p(f"已設定年份為: {BASE_YEAR}")

# ========= 路徑設定 =========
OUTPUT_DIR     = Path(r"D:\out").resolve()
VIDEO_OUT_DIR  = OUTPUT_DIR / "videos"
MODEL_PATH     = r"C:\Users\user\Desktop\test\test\best.pt"

# ✨ [新設定] 設定影片的「根目錄」，程式會搜尋此資料夾 (包含子資料夾)
VIDEO_ROOT_DIR = Path(r"D:\dav2mp4").resolve() 

for d in [OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)
if ENABLE_VIDEO_OUTPUT:
    VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    p(f"? 找不到模型：{MODEL_PATH}"); sys.exit(1)
if not os.path.isdir(VIDEO_ROOT_DIR):
    p(f"? 找不到影片根目錄：{VIDEO_ROOT_DIR}"); sys.exit(1)
    
p(f"輸出根目錄：{OUTPUT_DIR}")
p(f"模型路徑：{MODEL_PATH}")
p(f"影片搜尋目錄：{VIDEO_ROOT_DIR}")

# ========= 載入模型 (只需載入一次) =========
p("載入模型 (只需一次)…")
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

# ========= ROI 設定 =========
USE_ROI_FILTER = True
DRAW_ROI_BOX   = True
# 範例 ROI (x, y, w, h)，請根據您的影片調整
ROI_XYWH = (216, 276, 1805, 972) 

def roi_overlap_ok(box, roi_xywh, min_iou=0.05, min_inter_area_ratio=0.30):
    """
    檢查一個 BBox (x1, y1, x2, y2) 是否與 ROI (x, y, w, h) 有足夠的重疊。
    """
    rx, ry, rw, rh = roi_xywh
    x1, y1, x2, y2 = box
    rx2, ry2 = rx + rw, ry + rh
    
    # 計算交集
    ix1, iy1 = max(x1, rx), max(y1, ry)
    ix2, iy2 = min(x2, rx2), min(y2, ry2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    
    if inter <= 0:
        return False
        
    area_box = max(1.0, (x2 - x1) * (y2 - y1))
    area_roi = float(rw * rh)
    
    # 計算 IoU
    iou = inter / (area_box + area_roi - inter)
    
    # 計算 BBox 有多少比例在 ROI 內
    inter_ratio = inter / area_box
    
    # 只要 IoU 夠大 *或* BBox 大部分在 ROI 內，就通過
    return (iou >= min_iou) or (inter_ratio >= min_inter_area_ratio)


# ========= 參數 =========
IOU_NMS         = 0.45
CONF_PERSON     = 0.60
CONF_TAKE       = 0.15
BASE_CONF       = min(CONF_PERSON, CONF_TAKE)
MIN_IOU_OVERLAP = 0.3
TRACKER_CFG = str(Path(ul.__file__).parent / "cfg" / "trackers" / "botsort.yaml")
MAX_BAD_FRAMES = 1000

# ========= Excel 輸出相關 (函式定義) =========
XLSX_PATH = (OUTPUT_DIR / "excel" / "all_hourly.xlsx").resolve()

def write_excel_locked(xlsx_path: Path, df_new: pd.DataFrame):
    """
    ✨ [新功能] 此函式已被修改為「累加」而不是「覆蓋」
    """
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(xlsx_path) + ".lock"
    with open(lock_path, "w") as lk:
        try:
            msvcrt.locking(lk.fileno(), msvcrt.LK_LOCK, 1)

            df_old = pd.DataFrame(columns=["date", "hour", "person", "takebook"])
            if xlsx_path.exists() and xlsx_path.stat().st_size > 0:
                try:
                    df_old = pd.read_excel(xlsx_path)
                    keep_cols = ["date", "hour", "person", "takebook"]
                    df_old = df_old[[c for c in keep_cols if c in df_old.columns]]
                except Exception:
                    p(f"警告: 無法讀取舊 Excel {xlsx_path}，將建立新檔案。")
                    df_old = pd.DataFrame(columns=["date", "hour", "person", "takebook"])
            
            # --- ✨ [新邏輯] 累加資料 ---
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            
            # 確保都是數值
            df_all["date"]     = df_all["date"].astype(str)
            df_all["hour"]     = pd.to_numeric(df_all["hour"])
            df_all["person"]   = pd.to_numeric(df_all["person"])
            df_all["takebook"] = pd.to_numeric(df_all["takebook"])

            # ✨ [新邏輯] 使用 groupby 和 sum 進行累加
            df_aggregated = df_all.groupby(["date", "hour"]).agg({
                "person": "sum",
                "takebook": "sum"
            }).reset_index()

            # 排序
            df_aggregated = df_aggregated.sort_values(["date", "hour"])
            # --- [新邏輯結束] ---

            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                # 寫入累加後的結果
                df_aggregated.to_excel(writer, sheet_name="hourly", index=False)
        
        except Exception as e:
            p(f"!!! 寫入 Excel 時發生嚴重錯誤: {e}")
            traceback.print_exc()

        finally:
            lk.flush()
            os.fsync(lk.fileno())
            msvcrt.locking(lk.fileno(), msvcrt.LK_UNLCK, 1)

# ========= 影片輸出設定 (函式定義) =========
def try_open_writer(out_path, fourcc_str, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    vw = cv2.VideoWriter(out_path, fourcc, float(fps), size)
    return vw if vw.isOpened() else None

def init_video_writer(video_path: Path, fps, size):
    """
    ✨ [新功能] 此函式已被修改為接收 video_path
    """
    base_name = video_path.stem # '091812'
    avi_path  = str((VIDEO_OUT_DIR / f"output_tracking_{base_name}.avi").resolve())
    mp4_path  = str((VIDEO_OUT_DIR / f"output_tracking_{base_name}.mp4").resolve())
    
    writer = None
    for fourcc_str, path in (("XVID", avi_path), ("MJPG", avi_path)):
        vw = try_open_writer(path, fourcc_str, fps, size)
        if vw is not None:
            writer = vw
            p(f"? VideoWriter 使用 {fourcc_str} -> {path}")
            return writer
    vw = try_open_writer(mp4_path, "mp4v", fps, size)
    if vw is not None:
        writer = vw
        p(f"? VideoWriter 使用 mp4v -> {mp4_path}")
        return writer
    
    # 如果都失敗，回傳 None (我們將在主迴圈中處理)
    return None

# ========= ✨ [新功能] 批次處理主迴圈 =========
p("正在搜尋影片…")
try:
    # rglob 會遞迴搜尋所有子資料夾
    video_files = sorted(list(VIDEO_ROOT_DIR.rglob("*.mp4")))
except Exception as e:
    p(f"搜尋影片時發生錯誤: {e}")
    sys.exit(1)

if not video_files:
    p(f"!! 在 {VIDEO_ROOT_DIR} 中找不到任何 .mp4 檔案。")
    sys.exit(0)

p(f"搜尋完畢。共找到 {len(video_files)} 部影片。")
p("---")

# 依序處理每部影片
for i, video_path in enumerate(video_files):
    
    video_path_str = str(video_path)
    p(f"\n[ 批次 {i+1}/{len(video_files)} ] 正在處理: {video_path_str}")
    
    # --- 1. 動態設定時間 ---
    filename_stem = video_path.stem # '091709'
    INITIAL_TIME = parse_time_from_filename(filename_stem, BASE_YEAR)
    p(f"影片初始時間 (已解析): {INITIAL_TIME:%Y-%m-%d %H:%M:%S}")

    # --- 2. 開啟影片 ---
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        p("? 無法開啟影片，跳過此檔案。")
        continue

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = raw_fps if (raw_fps and raw_fps > 1 and not math.isnan(raw_fps)) else 20.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = -1
    p(f"尺寸={width}x{height}, fps?{fps:.2f}, 總幀={total_frames if total_frames>0 else '未知'}")

    # --- 3. 設定 ROI ---
    rx, ry, rw, rh = clamp_roi_to_frame(*ROI_XYWH, width, height)
    CURRENT_ROI_XYWH = (rx, ry, rw, rh)
    roi_filter = (lambda b: True) if not USE_ROI_FILTER else (lambda b: roi_overlap_ok(b, CURRENT_ROI_XYWH))

    # --- 4. ✨ [新功能] 重置狀態變數 ---
    # 確保每部影片的計數都是獨立的
    p("重置統計狀態…")
    person_tracks = {}  # {pid: {"has_taken": bool}}
    hour_seen_ids = defaultdict(lambda: {"person": set(), "takebook": set()})
    
    # --- 5. 初始化影片輸出 (如果啟用) ---
    writer = None
    temp_enable_video = ENABLE_VIDEO_OUTPUT # 複製一份開關狀態
    if temp_enable_video:
        try:
            writer = init_video_writer(video_path, fps, (width, height))
            if writer is None:
                 raise RuntimeError("init_video_writer 回傳 None")
        except RuntimeError as e:
            p(f"!! 警告: {e}。將關閉「此部影片」的影像輸出功能。")
            temp_enable_video = False # 僅針對此影片關閉

    # --- 6. 處理單一影片的主迴圈 ---
    frame_idx = 0
    det_total = 0 # 此變數現在是「單部影片」的總數
    bad_frame_count = 0
    
    progress_bar = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc=f"處理中 {video_path.name[:20]}..",
        unit="frame",
        leave=False # 迴圈結束時關閉進度條
    )

    try:
        while True:
            ok, full_frame = cap.read()
            if not ok or full_frame is None:
                if total_frames > 0 and frame_idx >= total_frames:
                    # p("影片正常結束。") # 不用印，進度條會顯示
                    break
                bad_frame_count += 1
                if bad_frame_count >= MAX_BAD_FRAMES:
                    p(f"[錯誤] 連續壞影格已達 {MAX_BAD_FRAMES} 幀，中止此影片。")
                    break
                frame_idx += 1
                if progress_bar: progress_bar.update(1)
                continue

            bad_frame_count = 0
            
            timestamp = INITIAL_TIME + timedelta(seconds=frame_idx / float(fps))
            date_str = timestamp.date().isoformat()
            hour_int = timestamp.hour

            with torch.inference_mode():
                results_list = model.track(
                    source=[full_frame], imgsz=IMGSZ, conf=BASE_CONF, iou=IOU_NMS,
                    device=DEVICE, half=HALF, persist=True, verbose=False,
                    tracker=TRACKER_CFG, stream=False
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
                    # ✨ [BUG 修正] ROI 過濾器現在使用 CURRENT_ROI_XYWH
                    if not roi_filter((float(x1), float(y1), float(x2), float(y2)), CURRENT_ROI_XYWH):
                        continue

                    box_coords = (int(x1), int(y1), int(x2), int(y2))
                    pid_int = int(pid) if pid is not None else None

                    if c == PERSON_ID:
                        persons_in_frame.append((box_coords, float(cf), pid_int))
                    elif c == TAKEBOOK_ID:
                        takebooks_in_frame.append((box_coords, float(cf), pid_int))

                for p_box, p_conf, p_pid in persons_in_frame:
                    final_detections.append((p_box, PERSON_ID, p_conf, p_pid))
                    if p_pid is not None:
                        hour_seen_ids[(date_str, hour_int)]["person"].add(p_pid)

                for t_box, t_conf, t_pid in takebooks_in_frame:
                    associated_person_pid = None
                    for p_box, _, p_pid in persons_in_frame:
                        if calculate_iou(t_box, p_box) >= MIN_IOU_OVERLAP:
                            associated_person_pid = p_pid
                            break

                    if associated_person_pid is not None:
                        if not person_tracks.get(associated_person_pid, {}).get("has_taken", False):
                            final_detections.append((t_box, TAKEBOOK_ID, t_conf, t_pid))
                            if associated_person_pid is not None:
                                hour_seen_ids[(date_str, hour_int)]["takebook"].add(associated_person_pid)
                            person_tracks.setdefault(associated_person_pid, {})["has_taken"] = True
                            det_total += 1

            # ✅ (如果 writer 成功初始化) 才畫框 + 寫出
            if temp_enable_video and writer is not None:
                out_img = full_frame.copy()
                if DRAW_ROI_BOX:
                    # ✨ [BUG 修正] 繪圖時使用 CURRENT_ROI_XYWH
                    (drx, dry, drw, drh) = CURRENT_ROI_XYWH
                    cv2.rectangle(out_img, (drx, dry), (drx + drw, dry + drh), (255, 0, 0), 2)
                    cv2.putText(out_img, "ROI", (drx + 6, dry + 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                COLORS = {}
                if PERSON_ID is not None: COLORS[PERSON_ID] = (0, 200, 0)
                if TAKEBOOK_ID is not None: COLORS[TAKEBOOK_ID] = (0, 255, 255)

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
            if progress_bar: progress_bar.update(1)

    except KeyboardInterrupt:
        p("\n🛑 偵測被使用者中止 (Ctrl+C)。將儲存目前影片進度並結束整個批次。")
        if progress_bar: progress_bar.close()
        # --- 儲存進度 (複製 'finally' 區塊的 Excel 部分) ---
        if ENABLE_EXCEL_OUTPUT and any(hour_seen_ids):
            p("正在儲存中止前的最後進度到 Excel...")
            try:
                rows = []
                for (date_str, hour_int), seen_ids in sorted(hour_seen_ids.items()):
                    rows.append({
                        "date": str(date_str), "hour": int(hour_int),
                        "person": len(seen_ids.get("person", set())),
                        "takebook": len(seen_ids.get("takebook", set()))
                    })
                df_hour = pd.DataFrame(rows, columns=["date", "hour", "person", "takebook"])
                write_excel_locked(XLSX_PATH, df_hour)
                p(f"📘 已更新 Excel (中止)：{XLSX_PATH}")
            except Exception:
                p("? 寫入 Excel 失敗 (中止)")
        # --- 中止儲存結束 ---
        break # 強制跳出 'for video_path in ...' 迴圈

    finally:
        # --- 7. ✨ [新功能] 迴圈內部的清理 ---
        # 確保當前影片的資源被釋放
        cap.release()
        if writer is not None:
            writer.release()
        if progress_bar: progress_bar.close()
        p(f"✅ 影片 {video_path.name} 處理完畢。共偵測 {det_total} 次 takebook。")

        # --- 8. ✨ [新功能] 每處理完一部影片，就更新一次 Excel ---
        if ENABLE_EXCEL_OUTPUT:
            # 檢查是否有任何資料
            if not any(hour_seen_ids):
                p("📘 這部影片沒有偵測到任何資料，不更新 Excel。")
            else:
                try:
                    rows = []
                    for (date_str, hour_int), seen_ids in sorted(hour_seen_ids.items()):
                        rows.append({
                            "date":     str(date_str),
                            "hour":     int(hour_int),
                            "person":   len(seen_ids.get("person", set())),
                            "takebook": len(seen_ids.get("takebook", set())),
                        })

                    # (這段 "if not rows" 邏輯現在幾乎不會被觸發，
                    #  因為我們在 'if not any(hour_seen_ids)' 已經檢查過了，但保留也無妨)
                    if not rows:
                        rows = [{
                            "date": INITIAL_TIME.date().isoformat(),
                            "hour": INITIAL_TIME.hour,
                            "person": 0, "takebook": 0
                        }]

                    df_hour = pd.DataFrame(rows, columns=["date", "hour", "person", "takebook"])
                    write_excel_locked(XLSX_PATH, df_hour)
                    p(f"📘 已更新 Excel：{XLSX_PATH}")
                except Exception:
                    traceback.print_exc()
                    p("? 寫入 Excel 失敗")
        
        p("---") # 分隔下一部影片

p("\n🎉🎉🎉 所有批次處理任務皆已完成。 🎉🎉🎉")