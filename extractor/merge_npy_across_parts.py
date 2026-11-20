import os
import glob
import numpy as np

# =========================
# CONFIG
# =========================
ROOT = "/home/didik/data_preprocess/feature_pcaps"

# List the available data partitions.
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

# Path to the ALL directory within each partition.
REL_ALL = "pcaps/IDS/NB15/120_5_flows(orign)/ALL"

# Output directory for merged arrays.
OUT_DIR = os.path.join(ROOT, "_MERGED_ALL_NPY")

# Memory-map large input arrays.
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

# Use the first available ALL directory to discover class files.
first_all = None
for d in PART_DIRS:
    p = os.path.join(ROOT, d, REL_ALL)
    if os.path.isdir(p):
        first_all = p
        break

if first_all is None:
    raise RuntimeError("No valid ALL directory found. Check ROOT / PART_DIRS / REL_ALL.")

# Collect the class filenames.
base_files = [os.path.basename(x) for x in list_all_npy_files(first_all)]
if not base_files:
    raise RuntimeError(f"No .npy found under: {first_all}")

print(f"[INFO] Output dir: {OUT_DIR}")
print(f"[INFO] Base ALL dir: {first_all}")
print(f"[INFO] Classes (npy): {len(base_files)}")

# Merge each class across partitions.
for fname in base_files:
    paths = []
    for d in PART_DIRS:
        p = os.path.join(ROOT, d, REL_ALL, fname)
        if os.path.isfile(p):
            paths.append(p)

    if not paths:
        print(f"[SKIP] {fname}: not found in any part.")
        continue

    # Use the first array as the shape reference.
    arrays = []
    ref = load_npy(paths[0])
    ref_shape = ref.shape
    arrays.append(ref)

    # Require matching shapes except on axis 0.
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

    # Concatenate the arrays.
    merged = np.concatenate(arrays, axis=0)

    out_path = os.path.join(OUT_DIR, fname)
    np.save(out_path, merged, allow_pickle=False)

    # Report the merged array statistics.
    total_rows = merged.shape[0]
    print(f"[OK] {fname}: merged {len(paths)} parts -> shape={merged.shape} -> saved")

print("[DONE] All merging finished.")
