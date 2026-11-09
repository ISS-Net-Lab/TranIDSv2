import numpy as np
from scapy.all import *
import glob
import os
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd


# =========================================================
# Single input root (ONLY ONE input path to set)
# =========================================================

PCAP_ROOT = os.environ.get("PCAP_ROOT", "/home/didik/data_preprocess/pcaps 22-1-2015-0/flows")

# ===== [MOD] NEW: output naming / output root from env =====
# OUT_PREFIX: 由 .sh 傳入（通常是 archive 的檔名去掉 .7z），用於避免輸出覆蓋
OUT_PREFIX = os.environ.get("OUT_PREFIX", "").strip()
# OUT_PARENT: 若你希望集中輸出到某個根目錄；不設就用 save_to / 預設 directory
OUT_PARENT = os.environ.get("OUT_PARENT", "").strip()

def _glob_5tuple_pcaps_all(pcap_root: str):
    """
    Single source of truth for where to find pcap files.
    Expected structure under pcap_root:
        <anything>/5_tuple_flows/*   (TCP/UDP mixed)
    """ 
    pattern = os.path.join(pcap_root, "*")
    return glob.glob(pattern)


# =========================================================
# CSV Rule Loader (ONLY uses required columns)
# =========================================================
def _load_attack_rules_csv(rule_csv_path: str) -> pd.DataFrame:
    """
    Only uses these columns:
      Start time, Last time, Protocol, Source IP, Source Port,
      Destination IP, Destination Port, Attack category

    Assumptions:
      - Start time / Last time are UNIX epoch seconds (int)
      - Protocol is 'tcp' or 'udp' (case-insensitive)
    """
    use_cols = [
        "Start time",
        "Last time",
        "Protocol",
        "Source IP",
        "Source Port",
        "Destination IP",
        "Destination Port",
        "Attack category",
    ]
    df = pd.read_csv(rule_csv_path, usecols=use_cols)

    # Normalize dtypes for stable matching
    df["Start time"] = df["Start time"].astype("int64")
    df["Last time"] = df["Last time"].astype("int64")
    df["Protocol"] = df["Protocol"].astype(str).str.lower().str.strip()

    df["Source IP"] = df["Source IP"].astype(str).str.strip()
    df["Destination IP"] = df["Destination IP"].astype(str).str.strip()

    df["Source Port"] = df["Source Port"].astype("int64")
    df["Destination Port"] = df["Destination Port"].astype("int64")

    df["Attack category"] = df["Attack category"].astype(str).str.strip()

    # Keep only required columns (explicit)
    df = df[use_cols].copy()
    return df


# =========================================================
# Base Class: CSV-based labeling + Dual output (NPY + CSV)
# =========================================================
class Get_NB15:  # Interface
    def __init__(
        self,
        paths,
        packet_shape,
        save_to=None,
        rule_csv_path=None,
        traffic_type="ALL",  # kept for output folder naming only
        save_csv=True,       # NEW: whether to also export CSV
        save_npy=True,       # NEW: whether to also export NPY
    ):
        self.paths = paths
        self.save_to = save_to
        self.TRAFFIC_TYPE = traffic_type
        self.IMG_SHAPE = packet_shape
        self.fixed_len = int(self.IMG_SHAPE[0] * self.IMG_SHAPE[1])

        self.SAVE_CSV = bool(save_csv)
        self.SAVE_NPY = bool(save_npy)

        if rule_csv_path is None:
            raise ValueError("rule_csv_path must be provided for CSV-based labeling.")
        self.rule_csv_path = rule_csv_path
        self.attack_rules = _load_attack_rules_csv(self.rule_csv_path)

        # ===== [MOD] NEW: store prefix for file naming =====
        self.out_prefix = OUT_PREFIX  # may be ""

        # ===== [MOD] NEW: optionally override output root using OUT_PARENT =====
        # If OUT_PARENT is set, output will go to:
        #   OUT_PARENT/<run_name>/<packetshape>_flows(orign)/ALL
        # run_name: prefer OUT_PREFIX; fallback to basename(PCAP_ROOT)
        if OUT_PARENT:
            run_name = self.out_prefix or os.path.basename(os.path.normpath(PCAP_ROOT)) or "run"
            self.directory = os.path.join(
                OUT_PARENT,
                run_name,
                f"{self.IMG_SHAPE[0]}_{self.IMG_SHAPE[1]}_flows(delall)",
                self.TRAFFIC_TYPE,
            )
        else:
            # original behavior
            if save_to is None:
                self.directory = (
                    f"D:/pcaps/IDS/NB15/"
                    f"{str(self.IMG_SHAPE[0])}_{str(self.IMG_SHAPE[1])}_flows/{self.TRAFFIC_TYPE}"
                )
            else:
                self.directory = save_to

    # -------------------------
    # CSV-based attack classes
    # -------------------------
    def get_attack_classes(self):
        """
        Returns attack_class list from CSV's 'Attack category' (excluding 'benign').
        """
        classes = sorted(self.attack_rules["Attack category"].dropna().unique().tolist())
        classes = [c for c in classes if c.lower() != "benign"]
        return classes

    # -------------------------
    # CSV-based label matching
    # -------------------------
    def get_label(self, pkts):
        """
        Flow-level label determined by the first packet only.

        Matching rule (exact):
          Protocol + SrcIP + SrcPort + DstIP + DstPort + Start/Last time window

        Returns:
          - "benign" if no match
          - Attack category string if matched
        """
        if len(pkts) == 0:
            return "benign"

        pkt0 = pkts[0]

        # Determine protocol and ports from packet itself (NO external TCP/UDP selection)
        if TCP in pkt0:
            proto = "tcp"
            sport = int(pkt0[TCP].sport)
            dport = int(pkt0[TCP].dport)
        elif UDP in pkt0:
            proto = "udp"
            sport = int(pkt0[UDP].sport)
            dport = int(pkt0[UDP].dport)
        else:
            return "benign"

        # Must be IPv4 for matching IP fields
        if IP not in pkt0:
            return "benign"

        src_ip = str(pkt0[IP].src).strip()
        dst_ip = str(pkt0[IP].dst).strip()

        # UNIX epoch seconds
        t_sec = int(pkt0.time)

        df = self.attack_rules
        hit = df[
            (df["Protocol"] == proto)
            & (df["Source IP"] == src_ip)
            & (df["Source Port"] == sport)
            & (df["Destination IP"] == dst_ip)
            & (df["Destination Port"] == dport)
            & (df["Start time"] <= t_sec)
            & (df["Last time"] >= t_sec)
        ]

        if len(hit) == 0:
            return "benign"

        # If multiple rules match (rare), take the first
        return str(hit.iloc[0]["Attack category"])

    # -------------------------
    # Packet selection (protocol-aware per-flow)
    # -------------------------
    def get_used_pkt(self, pkts):
        """
        Previously this depended on a fixed self.TRAFFIC_TYPE.
        Now we infer from the first packet per flow:
          - TCP: skip 3 packets (handshake) then take IMG_SHAPE[1]
          - UDP: take first IMG_SHAPE[1]
        """
        if len(pkts) == 0:
            return pkts

        pkt0 = pkts[0]
        if TCP in pkt0:
            return pkts[3 : self.IMG_SHAPE[1] + 3]
        elif UDP in pkt0:
            return pkts[: self.IMG_SHAPE[1]]
        else:
            return []

    # -------------------------
    # Flow preprocessing (to be overridden)
    # -------------------------
    def preprocess_flow(self, pkts):
        raise NotImplementedError

    # =========================================================
    # NEW: Unified "flow -> fixed vector" adapter for NPY export
    # =========================================================
    def _flow_to_fixed_vector(self, flow):
        """
        Convert a flow (whatever preprocess_flow returns) into a fixed-length 1D uint8 vector.

        Supports:
        - 1D list/np.array of numbers (already flat)
        - 2D nested list (e.g., [[pkt_bytes...], [pkt_bytes...], ...]) -> pad/trunc per packet
        """
        fixed_len = int(self.fixed_len)


        # Case A: already 1D
        if isinstance(flow, np.ndarray) and flow.ndim == 1:
            arr = flow
            if arr.size < fixed_len:
                out = np.zeros((fixed_len,), dtype=np.uint8)
                out[: arr.size] = arr.astype(np.uint8, copy=False)
                return out
            return arr[:fixed_len].astype(np.uint8, copy=False)

        if isinstance(flow, (list, tuple)) and len(flow) > 0 and not isinstance(flow[0], (list, tuple, np.ndarray)):
            arr = np.asarray(flow, dtype=np.uint8)
            if arr.size < fixed_len:
                out = np.zeros((fixed_len,), dtype=np.uint8)
                out[: arr.size] = arr
                return out
            return arr[:fixed_len]

        # Case B: nested packets (2D)
        flat = []
        # If empty or unrecognized, return zeros
        if flow is None:
            return np.zeros((fixed_len,), dtype=np.uint8)

        for pkt_bytes in flow:
            pkt_bytes = list(pkt_bytes)
            if len(pkt_bytes) < self.IMG_SHAPE[0]:
                pkt_bytes = pkt_bytes + [0] * (self.IMG_SHAPE[0] - len(pkt_bytes))
            else:
                pkt_bytes = pkt_bytes[: self.IMG_SHAPE[0]]
            flat.extend(pkt_bytes)

        if len(flat) < fixed_len:
            flat = flat + [0] * (fixed_len - len(flat))
        else:
            flat = flat[:fixed_len]

        return np.asarray(flat, dtype=np.uint8)

    def _flows_to_matrix(self, flows_list):
        """
        flows_list: list of flow objects
        return: (N, fixed_len) uint8
        """
        fixed_len = int(self.fixed_len)

        if flows_list is None or len(flows_list) == 0:
            return np.zeros((0, fixed_len), dtype=np.uint8)
        mats = [self._flow_to_fixed_vector(f) for f in flows_list]
        return np.stack(mats, axis=0)

    def _save_class_outputs(self, directory, class_name, flows):
        """
        Save both CSV and NPY (if enabled) for one class.
        """
        # ===== [MOD] NEW: prefix for filenames to avoid overwrite =====
        prefix = (self.out_prefix + "_") if self.out_prefix else ""

        if self.SAVE_CSV:
            pd.DataFrame(flows).to_csv(f"{directory}/{prefix}{class_name}_t.csv", index=False)

        if self.SAVE_NPY:
            arr = self._flows_to_matrix(flows)
            np.save(f"{directory}/{prefix}{class_name}_t.npy", arr, allow_pickle=False)
            return arr.shape
        return None

    # -------------------------
    # Save outputs (npy + csv)
    # -------------------------
    def save_np_files(self, b_flows, m_flows, attack_class):
        directory = self.directory
        os.makedirs(directory, exist_ok=True)

        print("Number of each class: ")
        print("-" * 50)

        for i, a in enumerate(attack_class):
            shape = self._save_class_outputs(directory, a, m_flows[i])
            if shape is None:
                print(f"{a} is {len(m_flows[i])}")
            else:
                print(f"{a} is {len(m_flows[i])} | npy shape={shape}")

        shape_b = self._save_class_outputs(directory, "benign", b_flows)
        if shape_b is None:
            print(f"benign is {len(b_flows)}")
        else:
            print(f"benign is {len(b_flows)} | npy shape={shape_b}")

    def record_error_pcap(self, path, file_name):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/errorpcap.txt", "a") as file:
            file.write(f"{file_name}\n")

    # -------------------------
    # Main run
    # -------------------------
    def run(self):
        attack_class = self.get_attack_classes()
        class_to_idx = {c: i for i, c in enumerate(attack_class)}

        dirs = self.paths
        b_flows = []
        m_flows = [[] for _ in range(len(attack_class))]

        print(f"packet_shape: {self.IMG_SHAPE}, traffic_type: {self.TRAFFIC_TYPE}")
        for d in tqdm(dirs):
            files = glob.glob(d)

            for f in files:
                try:
                    # read enough for TCP handshake + payload selection
                    pkts = rdpcap(f, count=(self.IMG_SHAPE[1] + 3))
                except (Scapy_Exception, EOFError):
                    continue

                if len(pkts) == 0:
                    continue

                # If TCP flow but too short for handshake skipping, skip
                if (TCP in pkts[0]) and (len(pkts) < 4):
                    continue

                label = self.get_label(pkts)

                pkts_used = self.get_used_pkt(pkts)
                if len(pkts_used) == 0:
                    continue

                flow = self.preprocess_flow(pkts_used)

                if label == "benign" or label.lower() == "benign":
                    b_flows.append(flow)
                else:
                    if label not in class_to_idx:
                        b_flows.append(flow)
                    else:
                        m_flows[class_to_idx[label]].append(flow)

                del pkts

            del files

        self.save_np_files(b_flows, m_flows, attack_class)


# =========================================================
# Variants (ALL now inherit dual-output behavior)
# =========================================================
class Get_NB15_rand(Get_NB15):  # Randoming MAC IP Port
    def __init__(self, paths, packet_shape, save_to=None, rule_csv_path=None, save_csv=True, save_npy=True):
        directory = save_to or (
            f"D:/pcaps/IDS/NB15/"
            f"{str(packet_shape[0])}_{str(packet_shape[1])}_flows(rand)/ALL"
        )
        super().__init__(
            paths,
            packet_shape,
            save_to=directory,
            rule_csv_path=rule_csv_path,
            traffic_type="ALL",
            save_csv=save_csv,
            save_npy=save_npy,
        )

    def preprocess_flow(self, pkts):
        flow = []

        for pkt in pkts:
            pkt.src = str(RandMAC())
            pkt.dst = str(RandMAC())

            try:
                pkt.payload.src = str(RandIP())
            except (Scapy_Exception, OSError):
                pkt.payload.src = str(RandIP6())

            try:
                pkt.payload.dst = str(RandIP())
            except (Scapy_Exception, OSError):
                pkt.payload.dst = str(RandIP6())

            pkt.payload.payload.sport = int(RandShort())
            pkt.payload.payload.dport = int(RandShort())

            pkt_head = [byte for byte in raw(pkt)]
            pkt_head.extend([0] * self.IMG_SHAPE[0])  # padding
            flow.extend(pkt_head[: self.IMG_SHAPE[0]])

        size = self.IMG_SHAPE[0] * self.IMG_SHAPE[1]
        if len(flow) < size:
            flow.extend([0] * size)
            flow = flow[:size]

        return flow


class Get_NB15_del(Get_NB15):  # delete IP, Mac, Port
    def __init__(self, paths, packet_shape, save_to=None, rule_csv_path=None, save_csv=True, save_npy=True):
        directory = save_to or (
            f"D:/pcaps/IDS/NB15/"
            f"{str(packet_shape[0])}_{str(packet_shape[1])}_flows(delall)/ALL"
        )
        super().__init__(
            paths,
            packet_shape,
            save_to=directory,
            rule_csv_path=rule_csv_path,
            traffic_type="ALL",
            save_csv=save_csv,
            save_npy=save_npy,
        )
        self.fixed_len = int((self.IMG_SHAPE[0] - 24) * self.IMG_SHAPE[1]) 

    def preprocess_flow(self, pkts):
        max_size = self.IMG_SHAPE[0] - 24
        flow = []
        for pkt in pkts:
            pkt_head = [byte for byte in raw(pkt)]
            pkt_head.extend([0] * max_size)  # padding

            # delete Destination and Source MAC, IP, Port
            for start, end in [(0, 11), (26, 37)]:
                pkt_head[start : end + 1] = [None] * (end - start + 1)
            pkt_head = [x for x in pkt_head if x is not None]

            flow.extend(pkt_head[:max_size])

        size = max_size * self.IMG_SHAPE[1]
        if len(flow) < size:
            flow.extend([0] * size)
            flow = flow[:size]
        return flow


class Get_NB15_port(Get_NB15):  # delete MAC IP Port, randing port
    def __init__(self, paths, packet_shape, save_to=None, rule_csv_path=None, save_csv=True, save_npy=True):
        directory = save_to or (
            f"D:/pcaps/IDS/NB15/"
            f"{str(packet_shape[0])}_{str(packet_shape[1])}_flows(port)/ALL"
        )
        super().__init__(
            paths,
            packet_shape,
            save_to=directory,
            rule_csv_path=rule_csv_path,
            traffic_type="ALL",
            save_csv=save_csv,
            save_npy=save_npy,
        )

    def preprocess_flow(self, pkts):
        max_size = self.IMG_SHAPE[0] - 20
        flow = []
        for pkt in pkts:
            pkt.payload.payload.sport = int(RandShort())
            pkt.payload.payload.dport = int(RandShort())
            pkt_head = [byte for byte in raw(pkt)]
            pkt_head.extend([0] * max_size)

            for start, end in [(0, 11), (26, 33)]:
                pkt_head[start : end + 1] = [None] * (end - start + 1)
            pkt_head = [x for x in pkt_head if x is not None]

            flow.extend(pkt_head[:max_size])

        size = max_size * self.IMG_SHAPE[1]
        if len(flow) < size:
            flow.extend([0] * size)
            flow = flow[:size]
        return flow


class Get_NB15_packet_cicfeature(Get_NB15_del):  # concat packets and cicflowmeter feature
    def __init__(self, paths, packet_shape, save_to=None, rule_csv_path=None, save_csv=True, save_npy=True):
        directory = save_to or (
            f"D:/pcaps/IDS/NB15/"
            f"{str(packet_shape[0])}_{str(packet_shape[1])}_flows(fp)/ALL"
        )
        super().__init__(paths, packet_shape, save_to=directory, rule_csv_path=rule_csv_path, save_csv=save_csv, save_npy=save_npy)

    def process_cicFeature(self, feature: dict):
        keys_to_remove = ["src_ip", "src_port", "dst_ip", "dst_port", "timestamp"]
        for k in keys_to_remove:
            if k in feature:
                del feature[k]
        return list(feature.values())

    def get_cicflowmester_feature(self, pkts):
        flow = Flow(pkts[0], direction=PacketDirection.FORWARD)
        for pkt in pkts:
            flow.add_packet(pkt, PacketDirection.FORWARD)
        return self.process_cicFeature(flow.get_data())

    def run(self):
        attack_class = self.get_attack_classes()
        class_to_idx = {c: i for i, c in enumerate(attack_class)}

        dirs = self.paths
        b_flows = []
        m_flows = [[] for _ in range(len(attack_class))]

        print(f"packet_shape: {self.IMG_SHAPE}, traffic_type: {self.TRAFFIC_TYPE}")
        error_log_path = (
            f"D:/pcaps/IDS/error_log/dataset/NB15/"
            f"{str(self.IMG_SHAPE[0])}_{str(self.IMG_SHAPE[1])}_flows(fp)"
        )

        for d in tqdm(dirs):
            files = glob.glob(d)

            for f in files:
                try:
                    pkts_full = rdpcap(f)
                    if len(pkts_full) == 0:
                        continue
                    if pkts_full[0].haslayer(IPv6):
                        continue
                except (Scapy_Exception, EOFError):
                    continue

                try:
                    cicfeature = self.get_cicflowmester_feature(pkts_full)
                except TypeError:
                    self.record_error_pcap(error_log_path, f)
                    continue

                try:
                    pkts = pkts_full[: (self.IMG_SHAPE[1] + 3)]
                except Exception:
                    continue

                if len(pkts) == 0:
                    continue
                if (TCP in pkts[0]) and (len(pkts) < 4):
                    continue

                label = self.get_label(pkts)

                pkts_used = self.get_used_pkt(pkts)
                if len(pkts_used) == 0:
                    continue

                flow_vec = self.preprocess_flow(pkts_used)  # from Get_NB15_del (flat list)
                flow_vec.extend(cicfeature)  # now length may exceed IMG_SHAPE[0]*IMG_SHAPE[1]

                if label == "benign" or label.lower() == "benign":
                    b_flows.append(flow_vec)
                else:
                    if label not in class_to_idx:
                        b_flows.append(flow_vec)
                    else:
                        m_flows[class_to_idx[label]].append(flow_vec)

                del pkts_full, pkts, cicfeature

            del files

        # Save via base dual-output saver (CSV+NPY)
        self.save_np_files(b_flows, m_flows, attack_class)


class Get_NB15_orignal(Get_NB15):  # save raw packets as CSV (+ also save fixed NPY)
    def __init__(self, paths, packet_shape, save_to=None, rule_csv_path=None, save_csv=True, save_npy=True):
        directory = save_to or (
            f"D:/pcaps/IDS/NB15/"
            f"{str(packet_shape[0])}_{str(packet_shape[1])}_flows(orign)/ALL"
        )
        super().__init__(
            paths,
            packet_shape,
            save_to=directory,
            rule_csv_path=rule_csv_path,
            traffic_type="ALL",
            save_csv=save_csv,
            save_npy=save_npy,
        )

    def preprocess_flow(self, pkts):
        flow = []
        for pkt in pkts:
            flow.append([byte for byte in raw(pkt)])
        return flow


# =========================================================
# Single runner (NO UDP/TCP split)
# =========================================================
def run_all(rule_csv_path):
    paths = _glob_5tuple_pcaps_all(PCAP_ROOT)

    # Choose which variant to run here:
    # Get_NB15_orignal(paths, packet_shape=(120, 10), rule_csv_path=rule_csv_path, save_csv=True, save_npy=True).run()

    # Alternatives:
    Get_NB15_del(paths, packet_shape=(120, 5), rule_csv_path=rule_csv_path, save_csv=False, save_npy=True).run()
    # Get_NB15_rand(paths, packet_shape=(120, 5), rule_csv_path=rule_csv_path, save_csv=True, save_npy=True).run()
    # Get_NB15_port(paths, packet_shape=(60, 3), rule_csv_path=rule_csv_path, save_csv=True, save_npy=True).run()
    # Get_NB15_packet_cicfeature(paths, packet_shape=(120, 5), rule_csv_path=rule_csv_path, save_csv=True, save_npy=True).run()


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    RULE_CSV_PATH = os.environ.get("RULE_CSV_PATH", "/home/didik/data_preprocess/NUSW-NB15_GT.csv")
    run_all(RULE_CSV_PATH)
