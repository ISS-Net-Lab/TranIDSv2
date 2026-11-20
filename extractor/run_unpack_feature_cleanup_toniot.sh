#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# CONFIG
# =========================================================

# Resolve data paths from the project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Root of the archives produced by pack_flows_relative.ps1.
# Search nested archive directories.
#   _packed_7z/normal_pcaps/*.7z
#   _packed_7z/normal_attack_pcaps/normal_DDoS/*.7z
#   _packed_7z/normal_attack_pcaps/normal_scanning/*.7z
#   ...
PACKED_ROOT="$PROJECT_ROOT/_packed_7z"

# TON_IoT feature extractor.
PY="$SCRIPT_DIR/get_featureTONIoT.py"

# Directory containing GroundTruth_Network_*.csv files.
#   GroundTruth_Network_1.csv
#   GroundTruth_Network_2.csv`
#   ...
GT_ROOT="$PROJECT_ROOT/SecurityEvents_Network_datasets"

# Temporary extraction root.
EXTRACT_PARENT="$PROJECT_ROOT/_extract_toniot"

# Feature output directory.
OUT_PARENT="$PROJECT_ROOT/feature_out_toniot"

# Log directory.
TEMP="$PROJECT_ROOT/temp_toniot"

# Python environment
PYTHON_BIN="$PROJECT_ROOT/venv/bin/python3"

# Ground-truth timestamp tolerance in seconds.


GT_TIME_TOLERANCE="1.0"

# CSV chunk size for the initial cache build.
GT_CSV_CHUNK_ROWS="200000"

# Set to 1 to rebuild the SQLite and attack index caches.

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

    # Start with the archive parent as the capture category.
    #   normal_pcaps
    #   normal_DDoS
    #   normal_scanning
    parent_name="$(basename "$(dirname "$z")")"

    # Fall back to the archive name if the parent is unrecognized.
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

    # Include the category in extraction paths to avoid collisions.

    safe_parent="${parent_name// /_}"
    outdir="$EXTRACT_PARENT/${safe_parent}__${name}"

    # Include the category in log filenames.
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
    # Root of the extracted flow PCAPs.
    #
    # OUT_PREFIX
    # Archive name without the extension.
    #
    # OUT_PARENT
    # Feature output directory.
    #
    # GT_ROOT
    # Ground-truth CSV directory.
    #
    # CAPTURE_ATTACK_TYPE
    #   folder-level candidate:ddos / scanning / benign / ...
    # Restrict ground-truth matching to this attack type; label unmatched flows benign.

    #
    # GT_TIME_TOLERANCE
    # Tolerance between the first packet and ground-truth timestamps.
    #
    # GT_CSV_CHUNK_ROWS
    # CSV chunk size for the initial cache build.
    #
    # GT_REBUILD_CACHE
    # Whether to rebuild the ground-truth cache.
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

    # Use pipefail to propagate Python failures through tee.

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
