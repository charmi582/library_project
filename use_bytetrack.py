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
# import msvcrt # (修改) 移至下方
from tqdm import tqdm # ✅ tqdm 進度條

# ✨ 匯入跨平台的檔案鎖模組
try:
    import msvcrt # 嘗試匯入 Windows 模組
except ImportError:
    import fcntl  # 如果失敗，則匯入 Linux/macOS 模組

# ✨ 匯入 logging
import logging

# (修改) 1. 匯入您的 ROI 設定檔 (不再需要 apply_roi_mask)
try:
    from roi import ROI_CONFIG
except ImportError:
    print("!!! 嚴重錯誤: 找不到 roi_config.py 檔案。 !!!")
    print("請確保 roi_config.py 與此腳本放在同一個資料夾中。")
    sys.exit(1)

# ========= LOGGING 設定 =========
def setup_logging(log_dir: Path):
    """設定日誌，同時輸出到檔案和控制台"""
    logger = logging.getLogger() # 取得根 logger
    logger.setLevel(logging.DEBUG) # 設定根 logger 的最低層級為 DEBUG

    # 移除所有已存在的 handlers，避免重複日誌
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 1. 檔案 Handler (FileHandler) - 記錄所有 DEBUG 以上的訊息
    log_file = log_dir / "tracking_log.log"
    file_handler = logging.FileHandler(str(log_file), mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) # 檔案記錄 DEBUG 層級
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 2. 控制台 Handler (StreamHandler) - 只顯示 INFO 以上的訊息
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # 控制台只顯示 INFO
    console_formatter = logging.Formatter('%(message)s') # 控制台使用更簡潔的格式
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

# ========= 可開關設定 =========
ENABLE_VIDEO_OUTPUT = False     # ✅ 是否輸出影片
ENABLE_EXCEL_OUTPUT = True      # ✅ 是否輸出 Excel 統計結果

# ========= 加速與裝置設定 =========
# ... (此區塊代碼不變) ...
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

DEVICE = "0" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu"
HALF   = (DEVICE != "cpu")
IMGSZ  = 960
# print(f"[Init] device={DEVICE}, half={HALF}, imgsz={IMGSZ}", flush=True) # (修改) logger 稍後設定

def p(msg): print(msg, flush=True) # (保留) p() 給 input() 提示使用

def clamp_roi_to_frame(x, y, w, h, W, H):
    # ... (此函式現在會被用到) ...
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return int(x), int(y), int(w), int(h)

def calculate_iou(boxA, boxB):
    # ... (此函式不變) ...
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

# ... (parse_time_from_filename, 互動輸入, 路徑設定, 載入模型, build_name_map, find_id_by_names 等不變) ...
# ✨ [新功能] 從檔名自動解析時間
def parse_time_from_filename(filename_stem: str, year: int):
    # ... (此函式不變) ...
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

# ========= (修改) 路徑設定 =========
# ✨ 1. 定義基礎輸出目錄
BASE_OUTPUT_DIR = Path(r"D:\out").resolve()

# ✨ 2. 建立唯一的執行 ID (時間戳 + Process ID)
run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
run_pid = os.getpid() # 取得目前腳本的 Process ID
run_id = f"{run_timestamp}_{run_pid}"

# ✨ 3. 建立此執行的專屬輸出資料夾
OUTPUT_DIR    = BASE_OUTPUT_DIR / run_id 
VIDEO_OUT_DIR = OUTPUT_DIR / "videos"

# ✨ 4. (重要) 將共享的 Excel 檔案路徑指向 "基礎" 目錄
XLSX_PATH = (BASE_OUTPUT_DIR / "excel" / "all_hourly.xlsx").resolve()

MODEL_PATH     = r"D:\test\test\best.pt"
VIDEO_ROOT_DIR = Path(r"D:\dav2mp4").resolve() 

# ✨ 5. 確保所有需要的資料夾 (包含共享的 excel 資料夾) 都被建立
for d in [OUTPUT_DIR, XLSX_PATH.parent]:
    d.mkdir(parents=True, exist_ok=True)
if ENABLE_VIDEO_OUTPUT:
    VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)

if not os.path.isfile(MODEL_PATH):
    p(f"? 找不到模型：{MODEL_PATH}"); sys.exit(1)
if not os.path.isdir(VIDEO_ROOT_DIR):
    p(f"? 找不到影片根目錄：{VIDEO_ROOT_DIR}"); sys.exit(1)

# ✨ 6. 在 OUTPUT_DIR 確定後，立刻設定 logging
setup_logging(OUTPUT_DIR) # 日誌會儲存在專屬的 OUTPUT_DIR 中
logger = logging.getLogger(__name__)

# ✨ 7. 現在可以用 logger 替換 print
logger.info(f"[Init] device={DEVICE}, half={HALF}, imgsz={IMGSZ}")
logger.info(f"基礎輸出目錄 (Base Dir): {BASE_OUTPUT_DIR}")
logger.info(f"專屬執行目錄 (Run Dir): {OUTPUT_DIR}")
logger.info(f"共享 Excel 路徑 (Excel Path): {XLSX_PATH}")
logger.info(f"模型路徑：{MODEL_PATH}")
logger.info(f"影片搜尋目錄：{VIDEO_ROOT_DIR}")

# ========= 載入模型 (只需載入一次) =========
logger.info("載入模型 (只需一次)…")
try:
    # ✨ [修改] 移除 ReID 預載入程式碼 (ByteTrack 不需要)
    # (原 L159-L167 的 ReID 載入 'try...except' 區塊已刪除)
        
    model = YOLO(MODEL_PATH)
    logger.info("? 模型載入 OK")
except Exception as e:
    logger.exception(f"? 模型載入失敗: {e}"); sys.exit(1)


def build_name_map(names):
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))

NAME_MAP = build_name_map(getattr(model, "names", {}))
logger.info(f"Model classes: {NAME_MAP}") # (修改)

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


# ========= (新) ROI 過濾設定 =========
# (新) 我們改用這個開關來控制是否在影片上「畫出」ROI 範圍
DRAW_ROI_BOX = True

# (新) 重新加入 'roi_overlap_ok' 函式
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
# ... (此區塊代碼不變) ...
IOU_NMS         = 0.45
CONF_PERSON     = 0.60
CONF_TAKE       = 0.15
BASE_CONF       = min(CONF_PERSON, CONF_TAKE)
MIN_IOU_OVERLAP = 0.3
# ✨ [修改] 改為使用 bytetrack.yaml
TRACKER_CFG = str(Path(ul.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml")
MAX_BAD_FRAMES = 1000

# ========= Excel 輸出相關 (函式定義) =========
# (修改) 移除 XLSX_PATH, 它已經在頂部被定義為共享路徑
# XLSX_PATH = (OUTPUT_DIR / "excel" / "all_hourly.xlsx").resolve() 
def write_excel_locked(xlsx_path: Path, df_new: pd.DataFrame):
    # ... (內部邏輯完全不變) ...
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(xlsx_path) + ".lock"
    with open(lock_path, "w") as lk:
        try:
            # --- ✨ 跨平台檔案鎖 START ---
            if os.name == 'nt': # Windows 系統
                msvcrt.locking(lk.fileno(), msvcrt.LK_LOCK, 1)
            else: # Linux/macOS 系統 (POSIX)
                fcntl.flock(lk.fileno(), fcntl.LOCK_EX) # 獨佔鎖
            # --- 跨平台檔案鎖 END ---

            df_old = pd.DataFrame(columns=["date", "hour", "person", "takebook"])
            if xlsx_path.exists() and xlsx_path.stat().st_size > 0:
                try:
                    df_old = pd.read_excel(xlsx_path)
                    keep_cols = ["date", "hour", "person", "takebook"]
                    df_old = df_old[[c for c in keep_cols if c in df_old.columns]]
                except Exception:
                    logger.warning(f"警告: 無法讀取舊 Excel {xlsx_path}，將建立新檔案。") # (修改)
                    df_old = pd.DataFrame(columns=["date", "hour", "person", "takebook"])
            
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            
            df_all["date"]     = df_all["date"].astype(str)
            df_all["hour"]     = pd.to_numeric(df_all["hour"])
            df_all["person"]   = pd.to_numeric(df_all["person"])
            df_all["takebook"] = pd.to_numeric(df_all["takebook"])

            # (保留) 您的累加邏輯
            df_aggregated = df_all.groupby(["date", "hour"]).agg({
                "person": "sum",
                "takebook": "sum"
            }).reset_index()

            df_aggregated = df_aggregated.sort_values(["date", "hour"])

            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                df_aggregated.to_excel(writer, sheet_name="hourly", index=False)
        
        except Exception as e:
            logger.exception(f"!!! 寫入 Excel 時發生嚴重錯誤: {e}") # (修改)
            # traceback.print_exc() # (修改) logger.exception 會自動處理

        finally:
            lk.flush()
            os.fsync(lk.fileno())
            # --- ✨ 跨平台解鎖 START ---
            if os.name == 'nt': # Windows
                msvcrt.locking(lk.fileno(), msvcrt.LK_UNLCK, 1)
            else: # Linux/macOS
                fcntl.flock(lk.fileno(), fcntl.LOCK_UN) # 解鎖
            # --- 跨平台解鎖 END ---


# ========= 影片輸出設定 (函式定義) =========
# ... (try_open_writer, init_video_writer 函式不變) ...
def try_open_writer(out_path, fourcc_str, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    vw = cv2.VideoWriter(out_path, fourcc, float(fps), size)
    return vw if vw.isOpened() else None

def init_video_writer(video_path: Path, fps, size):
    base_name = video_path.stem # '091812'
    # (修改) 影片會儲存到唯一的 VIDEO_OUT_DIR
    avi_path  = str((VIDEO_OUT_DIR / f"output_tracking_{base_name}.avi").resolve())
    mp4_path  = str((VIDEO_OUT_DIR / f"output_tracking_{base_name}.mp4").resolve())
    
    writer = None
    for fourcc_str, path in (("XVID", avi_path), ("MJPG", avi_path)):
        vw = try_open_writer(path, fourcc_str, fps, size)
        if vw is not None:
            writer = vw
            logger.info(f"? VideoWriter 使用 {fourcc_str} -> {path}") # (修改)
            return writer
    vw = try_open_writer(mp4_path, "mp4v", fps, size)
    if vw is not None:
        writer = vw
        logger.info(f"? VideoWriter 使用 mp4v -> {mp4_path}") # (修改)
        return writer
    
    return None

# ========= ✨ [新功能] 批次處理主迴圈 =========
logger.info("正在搜尋影片…") # (修改)
try:
    video_files = sorted(list(VIDEO_ROOT_DIR.rglob("*.mp4")))
    print(f"影片檔排列{video_files}, end='\n'")
    for i, file_path in enumerate(video_files):
        # 逐行印出清單中的每一個檔案路徑 (並加上編號)
        print(f"  [{i+1}] {file_path}")
except Exception as e:
    logger.error(f"搜尋影片時發生錯誤: {e}") # (修改)
    sys.exit(1)

if not video_files:
    logger.warning(f"!! 在 {VIDEO_ROOT_DIR} 中找不到任何 .mp4 檔案。") # (修改)
    sys.exit(0)

logger.info(f"搜尋完畢。共找到 {len(video_files)} 部影片。") # (修改)
logger.info("---") # (修改)

# 依序處理每部影片
for i, video_path in enumerate(video_files):
    
    video_path_str = str(video_path)
    logger.info(f"\n[ 批次 {i+1}/{len(video_files)} ] 正在處理: {video_path_str}") # (修改)
    
    # --- (新) 1a. 動態取得攝影機 ID ---
    camera_id = video_path.parent.name
    
    # --- (新) 1b. 從設定檔查找 ROI "資料" ---
    roi_data = ROI_CONFIG.get(camera_id)
    
    # --- (新) 1c. 檢查 ROI 是否存在 ---
    if roi_data is None:
        logger.warning(f"⚠️  警告: 在 roi_config.py 中找不到攝影機 {camera_id} 的 ROI 設定。") # (修改)
        logger.warning(f"將會跳過此影片: {video_path.name}") # (修改)
        logger.warning("---") # (修改)
        continue # 跳到下一部影片
    
    # (修改) 我們現在取得兩個值
    CURRENT_ROI_POLYGON = roi_data["polygon"] # 用於繪圖
    CURRENT_ROI_XYWH_RAW = roi_data["xywh"]   # 用於過濾
    
    logger.info(f"    (自動偵測) 攝影機 ID: {camera_id}，已載入 ROI。") # (修改)

    # --- 1d. 動態設定時間 (您原本的邏輯) ---
    filename_stem = video_path.stem # '091709'
    INITIAL_TIME = parse_time_from_filename(filename_stem, BASE_YEAR)
    logger.info(f"影片初始時間 (已解析): {INITIAL_TIME:%Y-%m-%d %H:%M:%S}") # (修改)

    # --- 2. 開啟影片 (您原本的邏輯) ---
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        logger.warning("? 無法開啟影片，跳過此檔案。") # (修改)
        continue

    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = raw_fps if (raw_fps and raw_fps > 1 and not math.isnan(raw_fps)) else 20.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = -1
    logger.info(f"尺寸={width}x{height}, fps?{fps:.2f}, 總幀={total_frames if total_frames>0 else '未知'}") # (修改)

    # --- (新) 3. 設定 "用於過濾的" ROI ---
    # 我們需要將 (x,y,w,h) 限制在影片範圍內
    rx, ry, rw, rh = clamp_roi_to_frame(*CURRENT_ROI_XYWH_RAW, width, height)
    CURRENT_ROI_XYWH_CLAMPED = (rx, ry, rw, rh)

    # --- 4. ✨ [新功能] 重置狀態變數 (您原本的邏輯) ---
    logger.info("重置統計狀態…") # (修改)
    person_tracks = {}  # {pid: {"has_taken": bool}}
    hour_seen_ids = defaultdict(lambda: {"person": set(), "takebook": set()})
    
    # --- 5. 初始化影片輸出 (您原本的邏輯) ---
    writer = None
    temp_enable_video = ENABLE_VIDEO_OUTPUT
    if temp_enable_video:
        try:
            writer = init_video_writer(video_path, fps, (width, height))
            if writer is None:
                    raise RuntimeError("init_video_writer 回傳 None")
        except RuntimeError as e:
            logger.warning(f"!! 警告: {e}。將關閉「此部影片」的影像輸出功能。") # (修改)
            temp_enable_video = False

    # --- 6. 處理單一影片的主迴圈 ---
    frame_idx = 0
    det_total = 0 
    bad_frame_count = 0
    
    progress_bar = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc=f"處理中 {video_path.name[:20]}..",
        unit="frame",
        leave=False 
    )

    try:
        while True:
            ok, full_frame = cap.read()
            if not ok or full_frame is None:
                if total_frames > 0 and frame_idx >= total_frames:
                    break
                bad_frame_count += 1
                if bad_frame_count >= MAX_BAD_FRAMES:
                    logger.error(f"[錯誤] 連續壞影格已達 {MAX_BAD_FRAMES} 幀，中止此影片。") # (修改)
                    break
                frame_idx += 1
                if progress_bar: progress_bar.update(1)
                continue

            bad_frame_count = 0
            
            # --- (刪除) 6a. 移除 ROI 遮罩 ---
            # (刪除) frame_for_model = apply_roi_mask(full_frame, CURRENT_ROI_POLYGON)
            
            with torch.inference_mode():
                results_list = model.track(
                    # (修改) 使用 "原始" 影像進行偵測
                    source=[full_frame], imgsz=IMGSZ, conf=BASE_CONF, iou=IOU_NMS,
                    device=DEVICE, half=HALF, persist=True, verbose=False,
                    tracker=TRACKER_CFG, stream=False
                    # ✨ [修改] ByteTrack 不需要 reid=True 參數
                )

            timestamp = INITIAL_TIME + timedelta(seconds=frame_idx / float(fps))
            date_str = timestamp.date().isoformat()
            hour_int = timestamp.hour

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
                        
                    # --- (新) 6b. 執行 "事後過濾" ---
                    # 使用我們在迴圈開始前設定的 CURRENT_ROI_XYWH_CLAMPED
                    if not roi_overlap_ok((float(x1), float(y1), float(x2), float(y2)), CURRENT_ROI_XYWH_CLAMPED):
                        continue # 如果 BBox 不在 ROI 中，則丟棄

                    box_coords = (int(x1), int(y1), int(x2), int(y2))
                    pid_int = int(pid) if pid is not None else None

                    if c == PERSON_ID:
                        persons_in_frame.append((box_coords, float(cf), pid_int))
                    elif c == TAKEBOOK_ID:
                        takebooks_in_frame.append((box_coords, float(cf), pid_int))

                # ... (此區塊的 "takebook" 關聯邏輯不變) ...
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
                # (修改) 我們在 "原始" 影像上畫框
                out_img = full_frame.copy() 
                
                if DRAW_ROI_BOX:
                    # (修改) 繪圖邏輯不變，我們仍然使用 "polygon" 來繪製
                    cv2.polylines(out_img, [CURRENT_ROI_POLYGON], 
                                    isClosed=True, color=(255, 0, 0), thickness=2)
                    
                    (drx, dry) = CURRENT_ROI_POLYGON[0][0] # 抓第一個頂點
                    cv2.putText(out_img, f"ROI (Cam: {camera_id})", (drx + 6, dry + 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                # ... (此區塊的 "COLORS" 和 "final_detections" 繪圖邏輯不變) ...
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
        logger.warning("\n🛑 偵測被使用者中止 (Ctrl+C)。將儲存目前影片進度並結束整個批次。") # (修改)
        if progress_bar: progress_bar.close()
        # ... (此區塊 "KeyboardInterrupt" 的 Excel 儲存邏輯不變) ...
        if ENABLE_EXCEL_OUTPUT and any(hour_seen_ids):
            logger.info("正在儲存中止前的最後進度到 Excel...") # (修改)
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
                logger.info(f"📘 已更新 Excel (中止)：{XLSX_PATH}") # (修改)
            except Exception:
                logger.error("? 寫入 Excel 失敗 (中止)") # (修改)
        break # 強制跳出 'for video_path in ...' 迴圈

    finally:
        # --- 7. ✨ [新功能] 迴圈內部的清理 ---
        # ... (此區塊 "finally" 的 cap.release() / writer.release() 邏輯不變) ...
        cap.release()
        if writer is not None:
            writer.release()
        if progress_bar: progress_bar.close()
        logger.info(f"✅ 影片 {video_path.name} 處理完畢。共偵測 {det_total} 次 takebook。") # (修改)

        # --- 8. ✨ [新功能] 每處理完一部影片，就更新一次 Excel ---
        # (修改) ❗❗ [修復] 確保即使沒有偵測到，也會寫入 0 筆紀錄 ❗❗
        if ENABLE_EXCEL_OUTPUT:
            try:
                rows = []
                # 1. 嘗試從 hour_seen_ids 填充
                for (date_str, hour_int), seen_ids in sorted(hour_seen_ids.items()):
                    rows.append({
                        "date":     str(date_str),
                        "hour":     int(hour_int),
                        "person":   len(seen_ids.get("person", set())),
                        "takebook": len(seen_ids.get("takebook", set())),
                    })

                # 2. 如果 rows 仍然是空的 (因為 hour_seen_ids 為空)
                if not rows:
                    logger.info(f"📘 這部影片 ({video_path.name}) 沒有偵測到資料，將寫入 0 筆紀錄。") # (修改)
                    rows = [{
                        "date": INITIAL_TIME.date().isoformat(),
                        "hour": INITIAL_TIME.hour,
                        "person": 0, "takebook": 0
                    }]

                # 3. 建立 DataFrame 並寫入
                df_hour = pd.DataFrame(rows, columns=["date", "hour", "person", "takebook"])
                write_excel_locked(XLSX_PATH, df_hour)
                logger.info(f"📘 已更新 Excel：{XLSX_PATH}") # (修改)
            
            except Exception:
                logger.exception("? 寫入 Excel 失敗") # (修改)
                # traceback.print_exc() # (修改)
        
        logger.info("---") # (修改) 分隔下一部影片

logger.info("\n🎉🎉🎉 所有批次處理任務皆已完成。 🎉🎉🎉")