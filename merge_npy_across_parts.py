import os
import glob
import numpy as np

# =========================
# CONFIG
# =========================
ROOT = "/home/didik/data_preprocess/feature_pcaps"

# 你圖一的資料夾名稱（請依你實際有的填；可保留 22-1-2015-0 或移除）
PART_DIRS = [
    "pcaps 17-2-2015-1",
    "pcaps 17-2-2015-2",
    "pcaps 17-2-2015-3",
    "pcaps 22-1-2015-0",
    "pcaps 22-1-2015-1",
    "pcaps 22-1-2015-2",
    "pcaps 22-1-2015-3",
    "pcaps 22-1-2015-4",
    "pcaps 22-1-2015-5",
]

# 每個 part 底下共同的相對路徑（到 ALL）
REL_ALL = "pcaps/IDS/NB15/120_5_flows(orign)/ALL"

# 輸出資料夾（合併後的 npy 放這）
OUT_DIR = os.path.join(ROOT, "_MERGED_ALL_NPY")

# 是否使用 memory-map 讀取（大型 npy 時有幫助）
USE_MMAP = True

# =========================
# HELPERS
# =========================
def list_all_npy_files(all_dir: str):
    return sorted(glob.glob(os.path.join(all_dir, "*.npy")))

def load_npy(path: str):
    if USE_MMAP:
        return np.load(path, mmap_mode="r")
    return np.load(path)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def human(n: int):
    # simple human readable
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"

# =========================
# MAIN
# =========================
ensure_dir(OUT_DIR)

# 1) 找到第一個存在的 ALL 目錄，拿它作為「類別檔名清單來源」
first_all = None
for d in PART_DIRS:
    p = os.path.join(ROOT, d, REL_ALL)
    if os.path.isdir(p):
        first_all = p
        break

if first_all is None:
    raise RuntimeError("No valid ALL directory found. Check ROOT / PART_DIRS / REL_ALL.")

# 2) 基準檔名集合（例如 benign_t.npy, DoS_t.npy...）
base_files = [os.path.basename(x) for x in list_all_npy_files(first_all)]
if not base_files:
    raise RuntimeError(f"No .npy found under: {first_all}")

print(f"[INFO] Output dir: {OUT_DIR}")
print(f"[INFO] Base ALL dir: {first_all}")
print(f"[INFO] Classes (npy): {len(base_files)}")

# 3) 逐個檔名合併
for fname in base_files:
    paths = []
    for d in PART_DIRS:
        p = os.path.join(ROOT, d, REL_ALL, fname)
        if os.path.isfile(p):
            paths.append(p)

    if not paths:
        print(f"[SKIP] {fname}: not found in any part.")
        continue

    # 讀取第一個做 shape 基準
    arrays = []
    ref = load_npy(paths[0])
    ref_shape = ref.shape
    arrays.append(ref)

    # 檢查後續 shape（除 axis=0 外必須一致）
    ok = True
    for p in paths[1:]:
        a = load_npy(p)
        if len(a.shape) != len(ref_shape) or a.shape[1:] != ref_shape[1:]:
            print(f"[ERROR] Shape mismatch for {fname}")
            print(f"        REF : {paths[0]} shape={ref_shape}")
            print(f"        THIS: {p} shape={a.shape}")
            ok = False
            break
        arrays.append(a)

    if not ok:
        print(f"[SKIP] {fname}: due to shape mismatch.")
        continue

    # 合併
    merged = np.concatenate(arrays, axis=0)

    out_path = os.path.join(OUT_DIR, fname)
    np.save(out_path, merged, allow_pickle=False)

    # 顯示簡要統計
    total_rows = merged.shape[0]
    print(f"[OK] {fname}: merged {len(paths)} parts -> shape={merged.shape} -> saved")

print("[DONE] All merging finished.")
