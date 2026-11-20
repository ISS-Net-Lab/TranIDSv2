#!/usr/bin/env bash
set -euo pipefail

# ========= CONFIG =========
PACKED_ROOT="/home/didik/data_preprocess/_packed_7z/"
CSV="/home/didik/data_preprocess/NUSW-NB15_GT.csv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/get_featureNB15.py"

TEMP="/home/didik/data_preprocess/temp"  # Temporary log directory.
EXTRACT_PARENT="/home/didik/data_preprocess"  # Parent of the extracted archives.
OUT_PARENT="/home/didik/data_preprocess/feature_out_5"


PYTHON_BIN="/home/didik/data_preprocess/venv/bin/python3"  # Python executable for feature extraction.
# =========================

mkdir -p "$TEMP"

# Find all .7z archives.
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
  name="${base%.7z}"  # Strip the .7z extension.
  outdir="$EXTRACT_PARENT/$name"  # Choose the extraction directory.

  echo "=================================================="
  echo "[$i/${#ARCHIVES[@]}] Archive: $z"
  echo "Extract to: $outdir"

  # Remove leftovers from an earlier run.
  rm -rf "$outdir"
  mkdir -p "$outdir"

  echo "[1] Extracting..."
  7z x -y -o"$outdir" "$z" > "$TEMP/${name}_extract.log"

  echo "[2] Running get_featureNB15.py ..."
  # Pass the input paths to the extractor.
  PCAP_ROOT="$outdir" RULE_CSV_PATH="$CSV" OUT_PREFIX="$name" OUT_PARENT="$OUT_PARENT" \
      "$PYTHON_BIN" "$PY" 2>&1 | tee "$TEMP/${name}_feature.log"

  echo "[3] Deleting extracted folder: $outdir"
  rm -rf "$outdir"

  echo "[OK] Done: $base"
done

echo "=================================================="
echo "[DONE] All archives processed."
echo "Logs are in: $TEMP"
