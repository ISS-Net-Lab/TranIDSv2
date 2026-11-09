import os
import re
import glob
import json
import pickle
import sqlite3
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd

from tqdm import tqdm

from scapy.all import (
    rdpcap,
    raw,
    IP,
    TCP,
    UDP,
)

from scapy.error import Scapy_Exception


# =========================================================
# SCRIPT DIRECTORY
# =========================================================

# All default paths are relative to the folder containing this Python file.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# CONFIG
# =========================================================

# ---------------------------------------------------------
# 解壓後其中一個 7z 的 PCAP root
#
# 正常情況由 .sh 傳入：
#
# PCAP_ROOT="$outdir"
# ---------------------------------------------------------

PCAP_ROOT = os.environ.get(
    "PCAP_ROOT",
    os.path.join(SCRIPT_DIR, "temp_extract")
).strip()


# ---------------------------------------------------------
# Archive 名稱，不包含 .7z
#
# 例如：
#
# normal_DDoS-1
# normal_DDoS-2
# normal_scanning-1
#
# 正常情況由 .sh 傳入
# ---------------------------------------------------------

OUT_PREFIX = os.environ.get(
    "OUT_PREFIX",
    ""
).strip()


# ---------------------------------------------------------
# Feature 輸出根目錄
# ---------------------------------------------------------

OUT_PARENT = os.environ.get(
    "OUT_PARENT",
    os.path.join(SCRIPT_DIR, "feature_out_toniot")
).strip()


# ---------------------------------------------------------
# Ground Truth CSV 所在資料夾
#
# 裡面應該有：
#
# GroundTruth_Network_1.csv
# GroundTruth_Network_2.csv
# ...
# GroundTruth_Network_17.csv
# ...
# ---------------------------------------------------------

GT_ROOT = os.environ.get(
    "GT_ROOT",
    os.path.join(SCRIPT_DIR, "SecurityEvents_GroundTruth_datasets")
).strip()


# ---------------------------------------------------------
# 可由 SH 明確傳入 capture attack type
#
# 可以填：
#
# normal_DDoS
# ddos
# normal_scanning
# scanning
# ...
#
# 如果不提供，程式會嘗試從：
#
# OUT_PREFIX
# PCAP_ROOT
#
# 自動判斷
# ---------------------------------------------------------

CAPTURE_ATTACK_TYPE_ENV = os.environ.get(
    "CAPTURE_ATTACK_TYPE",
    ""
).strip()


# ---------------------------------------------------------
# Ground Truth timestamp tolerance
#
# Ground Truth:
# ts = 156xxxxxxx.xxxxxx
#
# PCAP:
# pkt.time = 156xxxxxxx.xxxxxx
#
# 預設先使用 1 秒。
#
# 建議之後驗證 GT timestamp 與 PCAP timestamp 的差距，
# 再決定是否改成：
#
# 0.001
# 0.01
# 0.1
# 1.0
#
# SH 可以：
#
# GT_TIME_TOLERANCE=0.1
# ---------------------------------------------------------

GT_TIME_TOLERANCE = float(
    os.environ.get(
        "GT_TIME_TOLERANCE",
        "1.0"
    )
)


# ---------------------------------------------------------
# 一次讀多少 Ground Truth CSV rows
#
# 避免一次把約 1 GB CSV 全部塞進 RAM
# ---------------------------------------------------------

GT_CSV_CHUNK_ROWS = int(
    os.environ.get(
        "GT_CSV_CHUNK_ROWS",
        "200000"
    )
)


# ---------------------------------------------------------
# 強制重建 Ground Truth cache
#
# GT_REBUILD_CACHE=1
# ---------------------------------------------------------

GT_REBUILD_CACHE = (
    os.environ.get(
        "GT_REBUILD_CACHE",
        "0"
    ).strip().lower()
    in ("1", "true", "yes", "y")
)


# =========================================================
# TON_IoT folder -> Ground Truth type mapping
# =========================================================

#
# 左邊：
# 原始 PCAP folder / archive 名稱
#
# 右邊：
# Ground Truth CSV 的 type 欄位
#

FOLDER_TO_ATTACK = {

    # Pure benign
    "normal_pcaps": "benign",

    # Attack captures
    "normal_ddos": "ddos",
    "normal_dos": "dos",
    "normal_scanning": "scanning",
    "normal_xss": "xss",
    "normal_backdoor": "backdoor",
    "normal_ransomware": "ransomware",

    # TON_IoT 某些 folder 的 normal 在後面
    "injection_normal": "injection",
    "mitm_normal": "mitm",

    # 額外兼容
    "normal_injection": "injection",
    "normal_mitm": "mitm",
    "normal_password": "password",
    "password_normal": "password",
}


KNOWN_ATTACK_TYPES = {
    "ddos",
    "dos",
    "scanning",
    "xss",
    "backdoor",
    "ransomware",
    "injection",
    "mitm",
    "password",
}


# =========================================================
# Helper
# =========================================================

def find_flow_pcaps(pcap_root: str):
    """
    遞迴尋找解壓後所有 PCAP flow。

    支援：
        *.pcap
        *.pcapng
        *.cap
    """

    patterns = [
        "*.pcap",
        "*.pcapng",
        "*.cap",
    ]

    files = []

    for pattern in patterns:
        files.extend(
            glob.glob(
                os.path.join(
                    pcap_root,
                    "**",
                    pattern
                ),
                recursive=True
            )
        )

    files = sorted(set(files))

    return files


# =========================================================
# Attack category inference
# =========================================================

def _normalize_name(text: str):
    text = str(text).strip()

    text = os.path.basename(
        os.path.normpath(text)
    )

    text = text.lower()

    # Remove .7z if present
    if text.endswith(".7z"):
        text = text[:-3]

    return text


def infer_attack_type_from_text(text: str):
    """
    Examples:

    normal_DDoS
        -> ddos

    normal_DDoS-1
        -> ddos

    normal_scanning-4
        -> scanning

    Injection_normal-2
        -> injection

    normal_pcaps-3
        -> benign

    ddos
        -> ddos
    """

    if not text:
        return None

    name = _normalize_name(text)

    # -----------------------------------------
    # Case 1:
    # User directly supplies:
    #
    # ddos
    # scanning
    # xss
    # -----------------------------------------

    if name in KNOWN_ATTACK_TYPES:
        return name

    if name in ("benign", "normal"):
        return "benign"


    # -----------------------------------------
    # Case 2:
    # Match complete TON_IoT folder pattern
    # -----------------------------------------

    for folder_name in sorted(
        FOLDER_TO_ATTACK.keys(),
        key=len,
        reverse=True
    ):
        if folder_name in name:
            return FOLDER_TO_ATTACK[
                folder_name
            ]

    return None


def determine_capture_attack_type():
    """
    Priority:

    1. CAPTURE_ATTACK_TYPE env
    2. OUT_PREFIX
    3. PCAP_ROOT basename
    4. PCAP_ROOT path components
    """

    # -----------------------------------------
    # 1. Explicit env
    # -----------------------------------------

    if CAPTURE_ATTACK_TYPE_ENV:

        attack = infer_attack_type_from_text(
            CAPTURE_ATTACK_TYPE_ENV
        )

        if attack is None:
            raise ValueError(
                "Cannot understand "
                f"CAPTURE_ATTACK_TYPE="
                f"{CAPTURE_ATTACK_TYPE_ENV}"
            )

        return attack


    # -----------------------------------------
    # 2. OUT_PREFIX
    # -----------------------------------------

    if OUT_PREFIX:

        attack = infer_attack_type_from_text(
            OUT_PREFIX
        )

        if attack is not None:
            return attack


    # -----------------------------------------
    # 3. PCAP_ROOT basename
    # -----------------------------------------

    attack = infer_attack_type_from_text(
        PCAP_ROOT
    )

    if attack is not None:
        return attack


    # -----------------------------------------
    # 4. Search every path component
    # -----------------------------------------

    parts = os.path.normpath(
        PCAP_ROOT
    ).split(os.sep)

    for part in reversed(parts):

        attack = infer_attack_type_from_text(
            part
        )

        if attack is not None:
            return attack


    raise RuntimeError(
        "\nUnable to determine capture attack type.\n\n"
        "Please set environment variable, e.g.:\n\n"
        "CAPTURE_ATTACK_TYPE=normal_DDoS\n\n"
        "or:\n\n"
        "CAPTURE_ATTACK_TYPE=ddos\n"
    )


# =========================================================
# Ground Truth persistent store
# =========================================================

class GroundTruthStore:
    """
    Ground Truth architecture:

    GroundTruth_Network_1.csv
    GroundTruth_Network_2.csv
    ...
              |
              v
         SQLite cache
              |
              v
    query only one attack type
              |
              v
    5-tuple -> sorted timestamp array
              |
              v
         pickle cache


    This prevents every 7z from reparsing all CSV files.
    """

    GT_COLUMNS = [
        "ts",
        "src_ip",
        "src_port",
        "dst_ip",
        "dst_port",
        "proto",
        "type",
    ]

    CACHE_VERSION = "toniot_gt_v1"

    def __init__(self, gt_root):

        self.gt_root = gt_root

        if not os.path.isdir(
            self.gt_root
        ):
            raise RuntimeError(
                f"Ground Truth directory "
                f"does not exist:\n"
                f"{self.gt_root}"
            )

        # -----------------------------------------
        # Cache folder
        # -----------------------------------------

        self.cache_dir = os.path.join(
            self.gt_root,
            "_gt_cache"
        )

        os.makedirs(
            self.cache_dir,
            exist_ok=True
        )

        self.db_path = os.path.join(
            self.cache_dir,
            "toniot_groundtruth.sqlite"
        )

        self.index_cache_dir = os.path.join(
            self.cache_dir,
            "attack_indexes"
        )

        os.makedirs(
            self.index_cache_dir,
            exist_ok=True
        )

        # Process-level cache
        self.loaded_attack_indexes = {}

        # -----------------------------------------
        # Locate Ground Truth CSV
        # -----------------------------------------

        self.csv_files = sorted(
            glob.glob(
                os.path.join(
                    self.gt_root,
                    "GroundTruth_Network_*.csv"
                )
            )
        )

        if len(self.csv_files) == 0:
            raise RuntimeError(
                "No Ground Truth CSV files found:\n"
                f"{self.gt_root}/"
                "GroundTruth_Network_*.csv"
            )

        print(
            f"[GT] Found "
            f"{len(self.csv_files)} "
            f"Ground Truth CSV files."
        )

        # -----------------------------------------
        # Calculate source signature
        #
        # 如果 CSV 數量 / 大小 / 修改時間有改，
        # cache 自動重建。
        # -----------------------------------------

        self.source_signature = (
            self._calculate_source_signature()
        )

        # -----------------------------------------
        # Build / reuse DB
        # -----------------------------------------

        if (
            GT_REBUILD_CACHE
            or not self._cache_is_valid()
        ):

            self._build_sqlite_cache()

        else:

            print(
                "[GT] Existing SQLite cache "
                "is valid."
            )

        self.available_types = (
            self._get_available_types()
        )

        print(
            "[GT] Ground Truth attack types: "
            + ", ".join(
                self.available_types
            )
        )


    # =====================================================
    # Source signature
    # =====================================================

    def _calculate_source_signature(self):

        h = hashlib.sha256()

        for path in self.csv_files:

            stat = os.stat(path)

            item = (
                f"{os.path.basename(path)}|"
                f"{stat.st_size}|"
                f"{stat.st_mtime_ns}"
            )

            h.update(
                item.encode("utf-8")
            )

        return h.hexdigest()


    # =====================================================
    # Check cache
    # =====================================================

    def _cache_is_valid(self):

        if not os.path.isfile(
            self.db_path
        ):
            return False

        try:

            conn = sqlite3.connect(
                self.db_path
            )

            cur = conn.cursor()

            cur.execute(
                """
                SELECT value
                FROM metadata
                WHERE key = 'cache_version'
                """
            )

            row_version = cur.fetchone()

            cur.execute(
                """
                SELECT value
                FROM metadata
                WHERE key = 'source_signature'
                """
            )

            row_signature = cur.fetchone()

            cur.execute(
                """
                SELECT value
                FROM metadata
                WHERE key = 'build_complete'
                """
            )

            row_complete = cur.fetchone()

            conn.close()

        except Exception:
            return False

        if row_version is None:
            return False

        if row_signature is None:
            return False

        if row_complete is None:
            return False

        if row_version[0] != self.CACHE_VERSION:
            return False

        if row_signature[0] != self.source_signature:
            return False

        if row_complete[0] != "1":
            return False

        return True


    # =====================================================
    # Normalize Ground Truth chunk
    # =====================================================

    def _normalize_chunk(self, df):

        # -----------------------------------------
        # Numeric
        # -----------------------------------------

        df["ts"] = pd.to_numeric(
            df["ts"],
            errors="coerce"
        )

        df["src_port"] = pd.to_numeric(
            df["src_port"],
            errors="coerce"
        )

        df["dst_port"] = pd.to_numeric(
            df["dst_port"],
            errors="coerce"
        )


        # -----------------------------------------
        # String
        # -----------------------------------------

        df["src_ip"] = (
            df["src_ip"]
            .astype("string")
            .str.strip()
        )

        df["dst_ip"] = (
            df["dst_ip"]
            .astype("string")
            .str.strip()
        )

        df["proto"] = (
            df["proto"]
            .astype("string")
            .str.lower()
            .str.strip()
        )

        df["type"] = (
            df["type"]
            .astype("string")
            .str.lower()
            .str.strip()
        )


        # -----------------------------------------
        # Remove invalid rows
        # -----------------------------------------

        df = df.dropna(
            subset=self.GT_COLUMNS
        ).copy()


        # -----------------------------------------
        # Ports
        # -----------------------------------------

        valid_port = (
            (df["src_port"] >= 0)
            & (df["src_port"] <= 65535)
            & (df["dst_port"] >= 0)
            & (df["dst_port"] <= 65535)
        )

        df = df[
            valid_port
        ].copy()

        df["src_port"] = (
            df["src_port"]
            .astype(np.int64)
        )

        df["dst_port"] = (
            df["dst_port"]
            .astype(np.int64)
        )

        df["ts"] = (
            df["ts"]
            .astype(np.float64)
        )

        return df


    # =====================================================
    # Build merged SQLite Ground Truth
    # =====================================================

    def _build_sqlite_cache(self):

        print()
        print(
            "=" * 70
        )
        print(
            "[GT] Building merged SQLite "
            "Ground Truth cache..."
        )
        print(
            "[GT] This is required only "
            "the first time or when GT CSV changes."
        )
        print(
            "=" * 70
        )

        temp_db = (
            self.db_path
            + ".building"
        )

        if os.path.isfile(temp_db):
            os.remove(temp_db)

        conn = sqlite3.connect(
            temp_db
        )

        # Faster one-time bulk import
        conn.execute(
            "PRAGMA journal_mode=OFF"
        )

        conn.execute(
            "PRAGMA synchronous=OFF"
        )

        conn.execute(
            "PRAGMA temp_store=MEMORY"
        )


        # -----------------------------------------
        # Tables
        # -----------------------------------------

        conn.execute(
            """
            CREATE TABLE ground_truth (
                ts       REAL NOT NULL,
                src_ip   TEXT NOT NULL,
                src_port INTEGER NOT NULL,
                dst_ip   TEXT NOT NULL,
                dst_port INTEGER NOT NULL,
                proto    TEXT NOT NULL,
                type     TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE metadata (
                key   TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        total_rows = 0


        # -----------------------------------------
        # Read CSV files one-by-one
        # and chunks at a time
        # -----------------------------------------

        for file_no, csv_path in enumerate(
            self.csv_files,
            start=1
        ):

            print()
            print(
                f"[GT] "
                f"[{file_no}/"
                f"{len(self.csv_files)}] "
                f"Reading: "
                f"{os.path.basename(csv_path)}"
            )

            file_rows = 0

            try:

                reader = pd.read_csv(
                    csv_path,
                    usecols=self.GT_COLUMNS,
                    chunksize=GT_CSV_CHUNK_ROWS,
                    low_memory=False,
                )

            except Exception as e:

                conn.close()

                raise RuntimeError(
                    f"Unable to read Ground Truth CSV:\n"
                    f"{csv_path}\n"
                    f"{e}"
                )


            for chunk in reader:

                chunk = (
                    self._normalize_chunk(
                        chunk
                    )
                )

                if len(chunk) == 0:
                    continue

                # ---------------------------------
                # Append to SQLite
                # ---------------------------------

                chunk.to_sql(
                    "ground_truth",
                    conn,
                    if_exists="append",
                    index=False,
                    chunksize=5000,
                )

                file_rows += len(chunk)
                total_rows += len(chunk)

                print(
                    f"\r"
                    f"[GT] imported "
                    f"{file_rows:,} rows",
                    end="",
                    flush=True
                )

            print()


        # -----------------------------------------
        # Composite index
        #
        # type is first because feature extraction
        # first determines capture attack type.
        # -----------------------------------------

        print(
            "[GT] Creating SQLite index..."
        )

        conn.execute(
            """
            CREATE INDEX idx_gt_attack_flow_time
            ON ground_truth (
                type,
                proto,
                src_ip,
                src_port,
                dst_ip,
                dst_port,
                ts
            )
            """
        )


        # -----------------------------------------
        # Metadata
        # -----------------------------------------

        metadata = {
            "cache_version":
                self.CACHE_VERSION,

            "source_signature":
                self.source_signature,

            "source_file_count":
                str(
                    len(
                        self.csv_files
                    )
                ),

            "total_rows":
                str(total_rows),

            "build_complete":
                "1",
        }

        for key, value in metadata.items():

            conn.execute(
                """
                INSERT OR REPLACE
                INTO metadata(key, value)
                VALUES (?, ?)
                """,
                (
                    str(key),
                    str(value)
                )
            )

        conn.commit()

        conn.close()


        # -----------------------------------------
        # Atomic replacement
        # -----------------------------------------

        if os.path.isfile(
            self.db_path
        ):
            os.remove(
                self.db_path
            )

        os.replace(
            temp_db,
            self.db_path
        )


        # -----------------------------------------
        # Old attack indexes are no longer valid
        # -----------------------------------------

        for old_cache in glob.glob(
            os.path.join(
                self.index_cache_dir,
                "gt_index_*.pkl"
            )
        ):

            try:
                os.remove(old_cache)
            except OSError:
                pass


        print()
        print(
            f"[GT] SQLite cache created."
        )

        print(
            f"[GT] Total rows: "
            f"{total_rows:,}"
        )

        print(
            f"[GT] Cache: "
            f"{self.db_path}"
        )

        print(
            "=" * 70
        )
        print()


    # =====================================================
    # Attack types
    # =====================================================

    def _get_available_types(self):

        conn = sqlite3.connect(
            self.db_path
        )

        rows = conn.execute(
            """
            SELECT DISTINCT type
            FROM ground_truth
            ORDER BY type
            """
        ).fetchall()

        conn.close()

        return [
            str(row[0])
            for row in rows
        ]


    # =====================================================
    # Build/load attack-specific 5-tuple index
    # =====================================================

    def load_attack_index(
        self,
        attack_type
    ):
        """
        Result:

        {
            (
                proto,
                src_ip,
                src_port,
                dst_ip,
                dst_port
            ):
            np.array(
                sorted timestamps
            )
        }

        Only rows whose Ground Truth type equals
        attack_type are loaded.
        """

        attack_type = (
            str(attack_type)
            .lower()
            .strip()
        )

        # -----------------------------------------
        # Already loaded in current Python process
        # -----------------------------------------

        if (
            attack_type
            in self.loaded_attack_indexes
        ):

            return (
                self.loaded_attack_indexes[
                    attack_type
                ]
            )


        # -----------------------------------------
        # Validate attack type
        # -----------------------------------------

        if (
            attack_type
            not in self.available_types
        ):

            raise RuntimeError(
                f"\nGround Truth does not contain "
                f"type='{attack_type}'.\n"
                f"Available types:\n"
                f"{self.available_types}\n"
            )


        # -----------------------------------------
        # Persistent pickle cache
        # -----------------------------------------

        signature_short = (
            self.source_signature[:16]
        )

        cache_path = os.path.join(
            self.index_cache_dir,
            (
                f"gt_index_"
                f"{attack_type}_"
                f"{signature_short}.pkl"
            )
        )


        if os.path.isfile(
            cache_path
        ):

            print(
                f"[GT] Loading cached "
                f"'{attack_type}' index:"
            )

            print(
                f"     {cache_path}"
            )

            with open(
                cache_path,
                "rb"
            ) as f:

                attack_index = (
                    pickle.load(f)
                )

            self.loaded_attack_indexes[
                attack_type
            ] = attack_index

            print(
                f"[GT] Loaded "
                f"{len(attack_index):,} "
                f"unique 5-tuples."
            )

            return attack_index


        # -----------------------------------------
        # Build from SQLite
        # -----------------------------------------

        print()
        print(
            f"[GT] Building in-memory index "
            f"for attack type: "
            f"{attack_type}"
        )

        conn = sqlite3.connect(
            self.db_path
        )

        cursor = conn.execute(
            """
            SELECT
                proto,
                src_ip,
                src_port,
                dst_ip,
                dst_port,
                ts
            FROM ground_truth
            WHERE type = ?
            ORDER BY
                proto,
                src_ip,
                src_port,
                dst_ip,
                dst_port,
                ts
            """,
            (
                attack_type,
            )
        )


        attack_index = {}

        current_key = None
        timestamps = []

        rows_read = 0


        for row in cursor:

            key = (
                str(row[0]),
                str(row[1]),
                int(row[2]),
                str(row[3]),
                int(row[4]),
            )

            ts = float(
                row[5]
            )


            if current_key is None:

                current_key = key
                timestamps = [ts]


            elif key == current_key:

                timestamps.append(
                    ts
                )


            else:

                attack_index[
                    current_key
                ] = np.asarray(
                    timestamps,
                    dtype=np.float64
                )

                current_key = key
                timestamps = [ts]


            rows_read += 1


        # Last group
        if current_key is not None:

            attack_index[
                current_key
            ] = np.asarray(
                timestamps,
                dtype=np.float64
            )


        conn.close()


        print(
            f"[GT] Attack rows: "
            f"{rows_read:,}"
        )

        print(
            f"[GT] Unique 5-tuples: "
            f"{len(attack_index):,}"
        )


        # -----------------------------------------
        # Save attack-specific index
        # -----------------------------------------

        temp_cache = (
            cache_path
            + ".building"
        )

        with open(
            temp_cache,
            "wb"
        ) as f:

            pickle.dump(
                attack_index,
                f,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        os.replace(
            temp_cache,
            cache_path
        )


        print(
            f"[GT] Attack index saved:"
        )

        print(
            f"     {cache_path}"
        )


        self.loaded_attack_indexes[
            attack_type
        ] = attack_index

        return attack_index


# =========================================================
# Base TON_IoT feature extractor
# =========================================================

class Get_TONIoT:

    def __init__(
        self,
        paths,
        packet_shape,
        capture_attack_type,
        gt_store=None,
        output_mode="delall",
    ):

        self.paths = paths

        self.IMG_SHAPE = (
            packet_shape
        )

        self.capture_attack_type = (
            capture_attack_type
            .lower()
            .strip()
        )

        self.gt_store = (
            gt_store
        )

        self.TIME_TOLERANCE = (
            GT_TIME_TOLERANCE
        )


        # -----------------------------------------
        # Prefix
        # -----------------------------------------

        self.out_prefix = (
            OUT_PREFIX
            if OUT_PREFIX
            else os.path.basename(
                os.path.normpath(
                    PCAP_ROOT
                )
            )
        )


        # -----------------------------------------
        # Output directory
        #
        # Example:
        #
        # feature_out_toniot/
        #   normal_DDoS-1/
        #     120_5_flows(delall)/
        #       ALL/
        # -----------------------------------------

        self.directory = os.path.join(
            OUT_PARENT,
            self.out_prefix,
            (
                f"{self.IMG_SHAPE[0]}_"
                f"{self.IMG_SHAPE[1]}_"
                f"flows({output_mode})"
            ),
            "ALL",
        )


        # -----------------------------------------
        # Default fixed length
        # -----------------------------------------

        self.fixed_len = int(
            self.IMG_SHAPE[0]
            *
            self.IMG_SHAPE[1]
        )


        # -----------------------------------------
        # Load only candidate attack index
        #
        # normal_pcaps does not need Ground Truth.
        # -----------------------------------------

        self.attack_index = None

        if (
            self.capture_attack_type
            != "benign"
        ):

            if self.gt_store is None:

                raise RuntimeError(
                    "GroundTruthStore is required "
                    "for attack capture."
                )

            self.attack_index = (
                self.gt_store
                .load_attack_index(
                    self.capture_attack_type
                )
            )


    # =====================================================
    # Packet selection
    # =====================================================

    def get_used_pkt(
        self,
        pkts
    ):

        if len(pkts) == 0:
            return []


        pkt0 = pkts[0]


        # -----------------------------------------
        # TCP:
        # skip first 3 handshake packets
        #
        # packet 0,1,2 skipped
        # take next IMG_SHAPE[1]
        # -----------------------------------------

        if TCP in pkt0:

            return pkts[
                3:
                self.IMG_SHAPE[1] + 3
            ]


        # -----------------------------------------
        # UDP:
        # directly take first N packets
        # -----------------------------------------

        elif UDP in pkt0:

            return pkts[
                :
                self.IMG_SHAPE[1]
            ]


        return []


    # =====================================================
    # Ground Truth flow labeling
    # =====================================================

    def get_label(
        self,
        pkts
    ):
        """
        Two-stage logic:

        Stage 1:
            capture folder determines candidate attack

            normal_DDoS
                -> ddos

        Stage 2:
            flow's:
                proto
                src_ip
                src_port
                dst_ip
                dst_port
                timestamp

            is matched ONLY against ddos GT rows.

            Match:
                ddos

            No match:
                benign
        """


        # -----------------------------------------
        # Pure normal capture
        # -----------------------------------------

        if (
            self.capture_attack_type
            == "benign"
        ):

            return "benign"


        if len(pkts) == 0:
            return "benign"


        pkt0 = pkts[0]


        # -----------------------------------------
        # Only IPv4
        #
        # Ground Truth uses IPv4.
        # -----------------------------------------

        if IP not in pkt0:
            return "benign"


        # -----------------------------------------
        # Protocol / ports
        # -----------------------------------------

        if TCP in pkt0:

            proto = "tcp"

            sport = int(
                pkt0[TCP].sport
            )

            dport = int(
                pkt0[TCP].dport
            )


        elif UDP in pkt0:

            proto = "udp"

            sport = int(
                pkt0[UDP].sport
            )

            dport = int(
                pkt0[UDP].dport
            )


        else:
            return "benign"


        # -----------------------------------------
        # IP
        # -----------------------------------------

        src_ip = str(
            pkt0[IP].src
        ).strip()

        dst_ip = str(
            pkt0[IP].dst
        ).strip()


        # -----------------------------------------
        # Flow start timestamp
        # -----------------------------------------

        flow_ts = float(
            pkt0.time
        )


        # -----------------------------------------
        # 5-tuple
        # -----------------------------------------

        key = (
            proto,
            src_ip,
            sport,
            dst_ip,
            dport,
        )


        # -----------------------------------------
        # Search only candidate attack type
        # -----------------------------------------

        timestamps = (
            self.attack_index.get(
                key
            )
        )


        # -----------------------------------------
        # This 5-tuple never appears in the
        # candidate attack Ground Truth.
        #
        # Therefore:
        # benign
        # -----------------------------------------

        if timestamps is None:
            return "benign"


        if len(timestamps) == 0:
            return "benign"


        # -----------------------------------------
        # Binary search nearest GT timestamp
        #
        # timestamps are already sorted.
        # -----------------------------------------

        pos = int(
            np.searchsorted(
                timestamps,
                flow_ts,
                side="left"
            )
        )


        candidate_positions = []


        if pos < len(
            timestamps
        ):
            candidate_positions.append(
                pos
            )


        if pos > 0:
            candidate_positions.append(
                pos - 1
            )


        if len(
            candidate_positions
        ) == 0:

            return "benign"


        # -----------------------------------------
        # Find closest timestamp
        # -----------------------------------------

        best_pos = min(
            candidate_positions,
            key=lambda i: abs(
                float(
                    timestamps[i]
                )
                -
                flow_ts
            )
        )


        time_diff = abs(
            float(
                timestamps[
                    best_pos
                ]
            )
            -
            flow_ts
        )


        # -----------------------------------------
        # Match
        # -----------------------------------------

        if (
            time_diff
            <= self.TIME_TOLERANCE
        ):

            return (
                self.capture_attack_type
            )


        # -----------------------------------------
        # Candidate attack not matched
        # -> normal traffic
        # -----------------------------------------

        return "benign"


    # =====================================================
    # Override
    # =====================================================

    def preprocess_flow(
        self,
        pkts
    ):

        raise NotImplementedError


    # =====================================================
    # Flow -> fixed vector
    # =====================================================

    def _flow_to_fixed_vector(
        self,
        flow
    ):

        fixed_len = int(
            self.fixed_len
        )

        if flow is None:

            return np.zeros(
                fixed_len,
                dtype=np.uint8
            )


        arr = np.asarray(
            flow,
            dtype=np.uint8
        ).reshape(-1)


        # -----------------------------------------
        # Too long
        # -----------------------------------------

        if arr.size >= fixed_len:

            return arr[
                :fixed_len
            ]


        # -----------------------------------------
        # Too short -> zero pad
        # -----------------------------------------

        out = np.zeros(
            fixed_len,
            dtype=np.uint8
        )

        out[
            :arr.size
        ] = arr

        return out


    # =====================================================
    # All flows -> matrix
    # =====================================================

    def _flows_to_matrix(
        self,
        flows
    ):

        if (
            flows is None
            or len(flows) == 0
        ):

            return np.zeros(
                (
                    0,
                    self.fixed_len
                ),
                dtype=np.uint8
            )


        matrices = [

            self._flow_to_fixed_vector(
                flow
            )

            for flow in flows
        ]


        return np.stack(
            matrices,
            axis=0
        )


    # =====================================================
    # Save one class
    # =====================================================

    def _save_class_output(
        self,
        class_name,
        flows
    ):

        os.makedirs(
            self.directory,
            exist_ok=True
        )


        arr = (
            self._flows_to_matrix(
                flows
            )
        )


        # -----------------------------------------
        # Filename
        #
        # normal_DDoS-1_ddos_t.npy
        #
        # normal_DDoS-1_benign_t.npy
        # -----------------------------------------

        filename = (
            f"{self.out_prefix}_"
            f"{class_name}_t.npy"
        )


        output_path = os.path.join(
            self.directory,
            filename
        )


        np.save(
            output_path,
            arr,
            allow_pickle=False
        )


        print(
            f"[SAVE] "
            f"{class_name:<15} "
            f"flows={len(flows):,} "
            f"shape={arr.shape}"
        )

        print(
            f"       {output_path}"
        )


    # =====================================================
    # Main processing
    # =====================================================

    def run(self):

        benign_flows = []

        attack_flows = []


        # -----------------------------------------
        # Statistics
        # -----------------------------------------

        stat = {
            "total_files": 0,
            "read_error": 0,
            "empty": 0,
            "non_ipv4": 0,
            "non_tcp_udp": 0,
            "short_tcp": 0,
            "no_used_packet": 0,
            "preprocess_error": 0,
            "benign": 0,
            "attack": 0,
        }


        print()
        print(
            "=" * 75
        )

        print(
            "TON_IoT Raw Network "
            "Feature Extraction"
        )

        print(
            "=" * 75
        )

        print(
            f"PCAP_ROOT          : "
            f"{PCAP_ROOT}"
        )

        print(
            f"OUT_PREFIX         : "
            f"{self.out_prefix}"
        )

        print(
            f"Capture candidate  : "
            f"{self.capture_attack_type}"
        )

        print(
            f"Packet shape       : "
            f"{self.IMG_SHAPE}"
        )

        print(
            f"Output feature dim : "
            f"{self.fixed_len}"
        )

        print(
            f"Time tolerance     : "
            f"{self.TIME_TOLERANCE} sec"
        )

        print(
            f"PCAP flow files    : "
            f"{len(self.paths):,}"
        )

        print(
            f"Output directory   : "
            f"{self.directory}"
        )

        print(
            "=" * 75
        )

        print()


        # -----------------------------------------
        # Process every SplitCap flow
        # -----------------------------------------

        for f in tqdm(
            self.paths,
            desc=(
                self.out_prefix
            )
        ):

            stat[
                "total_files"
            ] += 1


            # =====================================
            # Read PCAP
            # =====================================

            try:

                # Need at most:
                #
                # TCP:
                # 3 handshake + 5 selected
                #
                # UDP:
                # 5 selected
                #
                pkts = rdpcap(
                    f,
                    count=(
                        self.IMG_SHAPE[1]
                        + 3
                    )
                )


            except (
                Scapy_Exception,
                EOFError,
                OSError,
            ):

                stat[
                    "read_error"
                ] += 1

                continue


            # =====================================
            # Empty
            # =====================================

            if len(pkts) == 0:

                stat[
                    "empty"
                ] += 1

                continue


            pkt0 = pkts[0]


            # =====================================
            # IPv4 only
            # =====================================

            if IP not in pkt0:

                stat[
                    "non_ipv4"
                ] += 1

                continue


            # =====================================
            # TCP / UDP only
            # =====================================

            if (
                TCP not in pkt0
                and UDP not in pkt0
            ):

                stat[
                    "non_tcp_udp"
                ] += 1

                continue


            # =====================================
            # TCP:
            # need at least one packet
            # after first 3
            # =====================================

            if (
                TCP in pkt0
                and len(pkts) < 4
            ):

                stat[
                    "short_tcp"
                ] += 1

                continue


            # =====================================
            # IMPORTANT:
            #
            # Label BEFORE deleting/skipping
            # packet information.
            # =====================================

            label = self.get_label(
                pkts
            )


            # =====================================
            # Select actual packets for feature
            # =====================================

            pkts_used = (
                self.get_used_pkt(
                    pkts
                )
            )


            if len(
                pkts_used
            ) == 0:

                stat[
                    "no_used_packet"
                ] += 1

                continue


            # =====================================
            # Packet-byte feature
            # =====================================

            try:

                flow = (
                    self.preprocess_flow(
                        pkts_used
                    )
                )

            except Exception as e:

                stat[
                    "preprocess_error"
                ] += 1

                print(
                    f"\n[WARN] "
                    f"preprocess failed:\n"
                    f"{f}\n"
                    f"{e}"
                )

                continue


            # =====================================
            # Save in corresponding class
            # =====================================

            if label == "benign":

                benign_flows.append(
                    flow
                )

                stat[
                    "benign"
                ] += 1


            else:

                attack_flows.append(
                    flow
                )

                stat[
                    "attack"
                ] += 1


            del pkts


        # =================================================
        # Save NPY
        # =================================================

        print()
        print(
            "=" * 75
        )

        print(
            "Saving feature files..."
        )

        print(
            "=" * 75
        )


        # Always save benign
        self._save_class_output(
            "benign",
            benign_flows
        )


        # For attack capture:
        # always save attack file,
        # even if it contains zero rows.
        if (
            self.capture_attack_type
            != "benign"
        ):

            self._save_class_output(
                self.capture_attack_type,
                attack_flows
            )


        # =================================================
        # Summary
        # =================================================

        print()
        print(
            "=" * 75
        )

        print(
            "Processing Summary"
        )

        print(
            "=" * 75
        )

        for key, value in stat.items():

            print(
                f"{key:<20}: "
                f"{value:,}"
            )


        usable = (
            stat["benign"]
            +
            stat["attack"]
        )

        print(
            "-" * 75
        )

        print(
            f"{'usable flows':<20}: "
            f"{usable:,}"
        )

        print(
            f"{'benign ratio':<20}: "
            f"{(
                stat['benign'] / usable
                if usable > 0
                else 0
            ):.6f}"
        )

        print(
            f"{'attack ratio':<20}: "
            f"{(
                stat['attack'] / usable
                if usable > 0
                else 0
            ):.6f}"
        )

        print(
            "=" * 75
        )


# =========================================================
# TON_IoT delall
#
# Same packet representation as your NB15 version
# =========================================================

class Get_TONIoT_del(
    Get_TONIoT
):

    def __init__(
        self,
        paths,
        packet_shape,
        capture_attack_type,
        gt_store=None,
    ):

        super().__init__(
            paths=paths,
            packet_shape=packet_shape,
            capture_attack_type=(
                capture_attack_type
            ),
            gt_store=gt_store,
            output_mode="delall",
        )


        # -----------------------------------------
        #
        # packet_shape = (120, 5)
        #
        # Remove:
        #
        # MAC:
        # 0 ~ 11
        #
        # IP + ports:
        # 26 ~ 37
        #
        # Total:
        # 12 + 12 = 24 bytes
        #
        # Per packet:
        # 120 - 24 = 96
        #
        # 5 packets:
        # 96 * 5 = 480
        #
        # -----------------------------------------

        self.fixed_len = int(
            (
                self.IMG_SHAPE[0]
                - 24
            )
            *
            self.IMG_SHAPE[1]
        )


    # =====================================================
    # Packet-byte preprocessing
    # =====================================================

    def preprocess_flow(
        self,
        pkts
    ):

        max_size = (
            self.IMG_SHAPE[0]
            - 24
        )

        flow = []


        for pkt in pkts:

            # -------------------------------------
            # Packet -> raw bytes
            # -------------------------------------

            pkt_head = [
                byte
                for byte
                in raw(pkt)
            ]


            # -------------------------------------
            # Add padding before deleting fields
            # -------------------------------------

            pkt_head.extend(
                [0]
                *
                self.IMG_SHAPE[0]
            )


            # -------------------------------------
            # Delete:
            #
            # 0-11:
            # Ethernet dst/src MAC
            #
            # 26-37:
            # IPv4 src/dst + TCP/UDP ports
            #
            # Same logic as original NB15 code.
            # -------------------------------------

            for start, end in [
                (0, 11),
                (26, 37),
            ]:

                pkt_head[
                    start:
                    end + 1
                ] = (
                    [None]
                    *
                    (
                        end
                        - start
                        + 1
                    )
                )


            # -------------------------------------
            # Remove deleted bytes
            # -------------------------------------

            pkt_head = [
                x
                for x in pkt_head
                if x is not None
            ]


            # -------------------------------------
            # 96 bytes per packet
            # -------------------------------------

            flow.extend(
                pkt_head[
                    :max_size
                ]
            )


        # -----------------------------------------
        # Fixed flow length:
        #
        # 96 × 5 = 480
        # -----------------------------------------

        size = (
            max_size
            *
            self.IMG_SHAPE[1]
        )


        if len(flow) < size:

            flow.extend(
                [0]
                *
                (
                    size
                    - len(flow)
                )
            )


        return flow[
            :size
        ]


# =========================================================
# Main runner
# =========================================================

def run_all():

    # -----------------------------------------------------
    # 1. Determine capture category
    # -----------------------------------------------------

    capture_attack_type = (
        determine_capture_attack_type()
    )


    print()
    print(
        f"[INFO] Capture attack type: "
        f"{capture_attack_type}"
    )


    # -----------------------------------------------------
    # 2. Find SplitCap flow PCAPs
    # -----------------------------------------------------

    paths = find_flow_pcaps(
        PCAP_ROOT
    )


    if len(paths) == 0:

        raise RuntimeError(
            "\nNo PCAP flow files found under:\n"
            f"{PCAP_ROOT}\n"
        )


    print(
        f"[INFO] Found "
        f"{len(paths):,} "
        f"flow PCAP files."
    )


    # -----------------------------------------------------
    # 3. Ground Truth
    #
    # Pure normal capture does not need GT.
    # -----------------------------------------------------

    gt_store = None


    if capture_attack_type != "benign":

        gt_store = GroundTruthStore(
            GT_ROOT
        )


    # -----------------------------------------------------
    # 4. Feature extraction
    #
    # Same setting as original:
    #
    # packet bytes = 120
    # packets      = 5
    # output       = 480
    # -----------------------------------------------------

    extractor = Get_TONIoT_del(
        paths=paths,
        packet_shape=(120, 5),
        capture_attack_type=(
            capture_attack_type
        ),
        gt_store=gt_store,
    )


    extractor.run()


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    run_all()