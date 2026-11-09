#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG =========
PACKED_ROOT="/home/didik/data_preprocess/_packed_7z/"
CSV="/home/didik/data_preprocess/NUSW-NB15_GT.csv"
PY="/home/didik/data_preprocess/get_featureNB15.py"

TEMP="/home/didik/data_preprocess/temp"                 # 暫存路徑
EXTRACT_PARENT="/home/didik/data_preprocess"            # 解壓縮後的路徑：/home/didik/data_preprocess/原檔名
OUT_PARENT="/home/didik/data_preprocess/feature_out_5"


PYTHON_BIN="/home/didik/data_preprocess/venv/bin/python3"   # 若你用 venv，可改成 /home/didik/data_preprocess/venv/bin/python
# =========================

mkdir -p "$TEMP"

# 收集所有 .7z
mapfile -t ARCHIVES < <(find "$PACKED_ROOT" -type f -iname "*.7z" | sort)

echo "Found ${#ARCHIVES[@]} .7z archives under: $PACKED_ROOT"
if [[ ${#ARCHIVES[@]} -eq 0 ]]; then
  echo "No .7z found. Exit."
  exit 0
fi

i=0
for z in "${ARCHIVES[@]}"; do
  i=$((i+1))
  base="$(basename "$z")"
  name="${base%.7z}"                     # 原檔名（去掉 .7z）
  outdir="$EXTRACT_PARENT/$name"         # 解壓後路徑：/home/didik/data_preprocess/原檔名

  echo "=================================================="
  echo "[$i/${#ARCHIVES[@]}] Archive: $z"
  echo "Extract to: $outdir"

  # 若上次殘留，先清掉
  rm -rf "$outdir"
  mkdir -p "$outdir"

  echo "[1] Extracting..."
  7z x -y -o"$outdir" "$z" > "$TEMP/${name}_extract.log"

  echo "[2] Running get_featureNB15.py ..."
  # 用 env 指定本次要處理的根目錄與 CSV
  PCAP_ROOT="$outdir" RULE_CSV_PATH="$CSV" OUT_PREFIX="$name" OUT_PARENT="$OUT_PARENT" \
      "$PYTHON_BIN" "$PY" 2>&1 | tee "$TEMP/${name}_feature.log"

  echo "[3] Deleting extracted folder: $outdir"
  rm -rf "$outdir"

  echo "[OK] Done: $base"
done

echo "=================================================="
echo "[DONE] All archives processed."
echo "Logs are in: $TEMP"
