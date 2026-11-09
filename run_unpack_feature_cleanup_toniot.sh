#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# CONFIG
# =========================================================

# All paths below are based on the folder containing this .sh file.
# Therefore the script can be moved together with the project folder
# without editing absolute /home/... paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 由 pack_flows.ps1 產生的 TON_IoT 7z 根目錄。
# find 會遞迴搜尋，因此可支援：
#   _packed_7z/normal_pcaps/*.7z
#   _packed_7z/normal_attack_pcaps/normal_DDoS/*.7z
#   _packed_7z/normal_attack_pcaps/normal_scanning/*.7z
#   ...
PACKED_ROOT="$SCRIPT_DIR/_packed_7z"

# 修改後的 TON_IoT getfeature
PY="$SCRIPT_DIR/get_featureTONIoT.py"

# TON_IoT Ground Truth CSV 所在資料夾：
#   GroundTruth_Network_1.csv
#   GroundTruth_Network_2.csv`
#   ...
GT_ROOT="$SCRIPT_DIR/SecurityEvents_Network_datasets"

# 解壓縮暫存根目錄。每個 archive 處理完會刪除。
EXTRACT_PARENT="$SCRIPT_DIR/_extract_toniot"

# feature 輸出根目錄
OUT_PARENT="$SCRIPT_DIR/feature_out_toniot"

# log 目錄
TEMP="$SCRIPT_DIR/temp_toniot"

# Python environment
PYTHON_BIN="$SCRIPT_DIR/venv/bin/python3"

# Ground Truth timestamp tolerance（秒）
# 目前 get_featureTONIoT.py 的預設也是 1.0。
# 建議先跑少量資料確認 timestamp 差值，再決定是否改為 0.1 / 0.01 等。
GT_TIME_TOLERANCE="1.0"

# Ground Truth CSV 第一次建立 cache 時，每次讀取的 row 數。
GT_CSV_CHUNK_ROWS="200000"

# 0 = 使用既有 Ground Truth cache（建議）
# 1 = 強制重新建立 SQLite / attack index cache
GT_REBUILD_CACHE="0"

# =========================================================
# PREPARE
# =========================================================

mkdir -p "$TEMP"
mkdir -p "$EXTRACT_PARENT"
mkdir -p "$OUT_PARENT"

if [[ ! -d "$PACKED_ROOT" ]]; then
    echo "[ERROR] PACKED_ROOT does not exist: $PACKED_ROOT" >&2
    exit 1
fi

if [[ ! -d "$GT_ROOT" ]]; then
    echo "[ERROR] GT_ROOT does not exist: $GT_ROOT" >&2
    exit 1
fi

if [[ ! -f "$PY" ]]; then
    echo "[ERROR] Python script does not exist: $PY" >&2
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Python executable does not exist or is not executable: $PYTHON_BIN" >&2
    exit 1
fi

# =========================================================
# Helper: map TON_IoT folder/archive name -> GT type
# =========================================================
infer_capture_attack_type() {
    local text="${1,,}"

    case "$text" in

        # Pure benign captures
        *normal_pcaps*)
            echo "benign"
            ;;

        # DDoS
        *normal_ddos*)
            echo "ddos"
            ;;

        # DoS
        *normal_dos*)
            echo "dos"
            ;;

        # Scanning
        *normal_scanning*)
            echo "scanning"
            ;;

        # XSS
        *normal_xss*)
            echo "xss"
            ;;

        # Backdoor
        *normal_backdoor*)
            echo "backdoor"
            ;;

        # IMPORTANT:
        # The original TON_IoT folder is actually named:
        # normal_runsomware
        *normal_runsomware*)
            echo "ransomware"
            ;;

        # Injection
        *injection_normal*)
            echo "injection"
            ;;

        # MITM
        *mitm_normal*)
            echo "mitm"
            ;;

        # Password
        *password_normal*)
            echo "password"
            ;;

        *)
            return 1
            ;;
    esac
}
# =========================================================
# Collect all 7z recursively
# =========================================================

mapfile -t ARCHIVES < <(
    find "$PACKED_ROOT" -type f -iname "*.7z" | sort
)

echo "Found ${#ARCHIVES[@]} .7z archives under: $PACKED_ROOT"

if [[ ${#ARCHIVES[@]} -eq 0 ]]; then
    echo "No .7z found. Exit."
    exit 0
fi

# =========================================================
# Process archives one by one
# =========================================================

i=0

for z in "${ARCHIVES[@]}"; do
    i=$((i + 1))

    base="$(basename "$z")"
    name="${base%.7z}"

    # 原始 category 通常就是 archive 的父資料夾，例如：
    #   normal_pcaps
    #   normal_DDoS
    #   normal_scanning
    parent_name="$(basename "$(dirname "$z")")"

    # 先使用父資料夾判斷；若不認得，再從 archive 名稱判斷。
    if capture_attack_type="$(infer_capture_attack_type "$parent_name")"; then
        :
    elif capture_attack_type="$(infer_capture_attack_type "$name")"; then
        :
    else
        echo "[ERROR] Cannot infer TON_IoT attack category." >&2
        echo "        Archive: $z" >&2
        echo "        Parent : $parent_name" >&2
        echo "        Name   : $name" >&2
        echo "        Add its mapping to infer_capture_attack_type()." >&2
        exit 1
    fi

    # 每個 archive 使用獨立解壓目錄。
    # 加入 category，避免不同 folder 中有同名 archive 時互相碰撞。
    safe_parent="${parent_name// /_}"
    outdir="$EXTRACT_PARENT/${safe_parent}__${name}"

    # Log 名稱也包含 category，避免同名 archive 覆蓋。
    log_prefix="${safe_parent}__${name}"

    echo "=========================================================================="
    echo "[$i/${#ARCHIVES[@]}]"
    echo "Archive              : $z"
    echo "Archive name         : $name"
    echo "Source folder        : $parent_name"
    echo "Candidate attack type: $capture_attack_type"
    echo "Extract to           : $outdir"
    echo "=========================================================================="

    # -----------------------------------------------------
    # Clean leftovers
    # -----------------------------------------------------

    rm -rf "$outdir"
    mkdir -p "$outdir"

    # -----------------------------------------------------
    # 1. Extract one 7z
    # -----------------------------------------------------

    echo "[1] Extracting..."

    if ! 7z x -y -o"$outdir" "$z" > "$TEMP/${log_prefix}_extract.log" 2>&1; then
        echo "[ERROR] Extraction failed: $z" >&2
        echo "        See: $TEMP/${log_prefix}_extract.log" >&2
        rm -rf "$outdir"
        exit 1
    fi

    # -----------------------------------------------------
    # 2. Run get_featureTONIoT.py
    # -----------------------------------------------------
    #
    # Important environment variables:
    #
    # PCAP_ROOT
    #   本次解壓後的 SplitCap flow PCAP root
    #
    # OUT_PREFIX
    #   archive 名稱，例如 normal_DDoS-1
    #
    # OUT_PARENT
    #   feature 輸出根目錄
    #
    # GT_ROOT
    #   所有 GroundTruth_Network_*.csv 所在目錄
    #
    # CAPTURE_ATTACK_TYPE
    #   folder-level candidate：ddos / scanning / benign / ...
    #   getfeature 只會查該 attack type 的 Ground Truth rows；
    #   flow 匹配不到則標為 benign。
    #
    # GT_TIME_TOLERANCE
    #   flow first-packet timestamp 與 GT ts 的容許差值
    #
    # GT_CSV_CHUNK_ROWS
    #   第一次建立 Ground Truth SQLite cache 時的 chunksize
    #
    # GT_REBUILD_CACHE
    #   是否強制重建 Ground Truth cache
    # -----------------------------------------------------

    echo "[2] Running get_featureTONIoT.py ..."

    PCAP_ROOT="$outdir" \
    OUT_PREFIX="$name" \
    OUT_PARENT="$OUT_PARENT" \
    GT_ROOT="$GT_ROOT" \
    CAPTURE_ATTACK_TYPE="$capture_attack_type" \
    GT_TIME_TOLERANCE="$GT_TIME_TOLERANCE" \
    GT_CSV_CHUNK_ROWS="$GT_CSV_CHUNK_ROWS" \
    GT_REBUILD_CACHE="$GT_REBUILD_CACHE" \
        "$PYTHON_BIN" "$PY" \
        2>&1 | tee "$TEMP/${log_prefix}_feature.log"

    # pipefail 已啟用：若 Python 失敗，即使 tee 成功，整條 pipeline 仍會失敗。

    # -----------------------------------------------------
    # 3. Delete extracted PCAPs
    # -----------------------------------------------------

    echo "[3] Deleting extracted folder: $outdir"
    rm -rf "$outdir"

    echo "[OK] Done: $base"
    echo

done

# =========================================================
# DONE
# =========================================================

echo "=========================================================================="
echo "[DONE] All TON_IoT archives processed."
echo "Feature output : $OUT_PARENT"
echo "Ground Truth   : $GT_ROOT"
echo "Logs           : $TEMP"
echo "=========================================================================="
