import numpy as np
import cv2

# -----------------------------------------------------------------
# ✨ [您需要在此處設定] ✨
#
# 我們現在儲存兩種格式：
# 1. (x, y, w, h) 格式 -> 用於 "事後過濾"
# 2. Polygon 格式     -> 用於 "繪製"
# -----------------------------------------------------------------

# --- ch1 (001) ---
roi_1_xywh = (216, 276, 1805, 972)
roi_1_polygon = [ (216, 276), (2021, 276), (2021, 1248), (216, 1248) ]

# --- ch2 (002) ---
roi_2_xywh = (420, 0, 1612, 1292)
roi_2_polygon = [ (420, 0), (2032, 0), (2032, 1292), (420, 1292) ]

# --- ch3 (003) ---
roi_3_xywh = (223, 14, 1469, 1377)
roi_3_polygon = [ (223, 14), (1692, 14), (1692, 1391), (223, 1391) ]

# --- ch4 (004) ---
roi_4_xywh = (622, 275, 1617, 1159)
roi_4_polygon = [ (622, 275), (2239, 275), (2239, 1434), (622, 1434) ]

# ... 未來可以新增 roi_5_... = ...


# --- 1. 建立「查找字典」---
# 鍵(Key): 攝影機 ID (來自資料夾名稱，例如 "001")
# 值(Value): 一個包含 "polygon" 和 "xywh" 的字典
#
ROI_CONFIG = {
    "001": { # <--- 值(Value)是一個字典
        "polygon": np.array([roi_1_polygon], dtype=np.int32),
        "xywh": roi_1_xywh
    },
    "002": { # <--- 值(Value)是一個字典
        "polygon": np.array([roi_2_polygon], dtype=np.int32),
        "xywh": roi_2_xywh
    },
    "003": { # <--- 值(Value)是一個字典
        "polygon": np.array([roi_3_polygon], dtype=np.int32),
        "xywh": roi_3_xywh
    },
    "004": { # <--- 值(Value)是一個字典
        "polygon": np.array([roi_4_polygon], dtype=np.int32),
        "xywh": roi_4_xywh
    },
}


# --- (刪除) 輔助函式：應用遮罩 ---
# 我們不再需要 apply_roi_mask 函式，因為我們
# 不再將 ROI 以外的區域塗黑。
# def apply_roi_mask(frame, roi_polygon_numpy):
#     ...