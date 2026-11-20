import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("is_available:", torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found, using CPU")

import os
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from dataclasses import dataclass
from typing import Optional
from typing import Dict, Optional
import math
from torch.utils.data import DataLoader, Subset
import optuna
from optuna.samplers import TPESampler
from util.data_preparing import (
    evaluate_test_data,
    score_reconstruction,
    select_validation_threshold,
)
# ============================================================
# 1D data processing - remove the 2D transformation and keep only 1D loading
# ============================================================

def vec_to_tensor(
    x: np.ndarray,
    input_mode: str = "flat_1d",   # "flat_1d" or "packet_2d"
    packet_shape: tuple[int, int] = (5, 96),
) -> torch.Tensor:
    """Reshape 480 features into five packets of 96 bytes each."""
    t = torch.from_numpy(np.array(x, copy=False)).float()

    expected_dim = packet_shape[0] * packet_shape[1]
    if t.numel() != expected_dim:
        raise ValueError(
            f"Expected {expected_dim}-dim vector, got {t.numel()} (shape={x.shape})"
        )

    if input_mode == "flat_1d":
        return t.view(1, expected_dim)   # (1, 480)

    elif input_mode == "packet_2d":
        return t.view(packet_shape[0], packet_shape[1])   # (5, 96)

    else:
        raise ValueError(f"Unknown input_mode: {input_mode}")

class SplitFolderNpyDataset(Dataset):
    def __init__(
        self,
        split_dir: str | Path,
        use_label_file: bool = False,
        mmap: bool = True,
        input_mode: str = "flat_1d",     # Added
        packet_shape: tuple[int, int] = (5, 96),   # Added
    ):
        self.split_dir = Path(split_dir)
        self.use_label_file = use_label_file
        self.mmap = mmap
        self.input_mode = input_mode
        self.packet_shape = packet_shape

        if not self.split_dir.exists():
            raise FileNotFoundError(f"split_dir not found: {self.split_dir.resolve()}")

        # Find all .npy files (excluding label files)
        x_files = sorted(
            [p for p in self.split_dir.glob("*.npy") if not p.name.lower().endswith("_label.npy")]
        )
        if not x_files:
            raise FileNotFoundError(f"No X .npy files found under: {self.split_dir.resolve()}")

        self.class_names = [p.stem for p in x_files]
        uniq = sorted(set(self.class_names), key=self.class_names.index)
        self.class_to_idx = {c: i for i, c in enumerate(uniq)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        self.arrs = []
        self.file_class_idx = []
        self.label_arrs = []
        self.lens = []

        for p in x_files:
            arr = np.load(str(p), mmap_mode="r" if self.mmap else None)
            n = int(arr.shape[0])
            cls_idx = int(self.class_to_idx[p.stem])

            # Process label files
            lp = p.with_name(p.stem + "_label.npy")
            lbl = None
            if lp.exists():
                lbl = np.load(str(lp), mmap_mode="r" if self.mmap else None)
                if lbl.ndim == 2 and lbl.shape[1] == 1:
                    lbl = lbl.reshape(-1)

            self.arrs.append(arr)
            self.file_class_idx.append(cls_idx)
            self.label_arrs.append(lbl)
            self.lens.append(n)

        self.cum = np.cumsum([0] + self.lens)
        print(f"[Dataset] Loaded from {self.split_dir.resolve()}, total samples: {self.cum[-1]}")

    def __len__(self):
        return int(self.cum[-1])

    def _locate(self, idx: int):
        file_i = int(np.searchsorted(self.cum, idx, side="right") - 1)
        local_i = int(idx - self.cum[file_i])
        return file_i, local_i

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        t = vec_to_tensor(
            x,
            input_mode=self.input_mode,
            packet_shape=self.packet_shape,
        )
        if t.max() > 1.5:
            t = t / 255.0
        return t

    def __getitem__(self, idx: int):
        file_i, local_i = self._locate(idx)
        x = self.arrs[file_i][local_i]
        x = self._to_tensor(x)

        if self.use_label_file and (self.label_arrs[file_i] is not None):
            y = int(self.label_arrs[file_i][local_i])
        else:
            y = int(self.file_class_idx[file_i])
        return x, y


class TabularParquetDataset(Dataset):
    """
    Read tabular parquet data such as NF-UNSW-NB15-v3.parquet.

    Output shape follows the same AE-ViT input contract used by SplitFolderNpyDataset:
        input_mode="flat_1d":   x shape = (1, packet_shape[0] * packet_shape[1])
        input_mode="packet_2d": x shape = packet_shape

    For NF-UNSW-NB15-v3.parquet:
        - Label is used as the binary label column.
        - Attack is dropped because it is an attack-category text column.
        - Label and Attack are NOT used as features.
        - label 0 = benign, non-zero = attack.
    """
    def __init__(
        self,
        parquet_path: str | Path,
        label_col: str = "Label",
        attack_col: str = "Attack",
        drop_cols: Optional[list[str]] = None,
        feature_cols: Optional[list[str]] = None,
        input_mode: str = "flat_1d",
        packet_shape: tuple[int, int] = (1, 48),
        scale: bool = True,
        allow_truncate: bool = False,
        max_rows: Optional[int] = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.label_col = label_col
        self.attack_col = attack_col
        self.input_mode = input_mode
        self.packet_shape = packet_shape
        self.scale = scale
        self.allow_truncate = allow_truncate
        self.target_dim = int(packet_shape[0] * packet_shape[1])

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"parquet_path not found: {self.parquet_path.resolve()}")

        print(f"[ParquetDataset] Loading: {self.parquet_path.resolve()}")
        df = pd.read_parquet(self.parquet_path)

        if max_rows is not None:
            df = df.iloc[:int(max_rows)].copy()
            print(f"[ParquetDataset] Using first {len(df)} rows because max_rows={max_rows}")

        if label_col not in df.columns:
            raise ValueError(
                f"label_col='{label_col}' not found. Available columns: {list(df.columns)}"
            )

        # ------------------------------------------------------------
        # 1. Labels
        # ------------------------------------------------------------
        labels = df[label_col].astype(np.int64).to_numpy()
        labels = np.where(labels == 0, 0, 1).astype(np.int64)

        # ------------------------------------------------------------
        # 2. Features
        # ------------------------------------------------------------
        if drop_cols is None:
            drop_cols = []

        auto_drop = [label_col]
        if attack_col in df.columns:
            auto_drop.append(attack_col)

        final_drop_cols = list(dict.fromkeys(auto_drop + drop_cols))

        if feature_cols is None:
            feature_df = df.drop(columns=[c for c in final_drop_cols if c in df.columns])
            feature_df = feature_df.select_dtypes(include=[np.number])
            feature_cols = feature_df.columns.tolist()
        else:
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"feature_cols contains missing columns: {missing}")
            feature_df = df[feature_cols].select_dtypes(include=[np.number])
            feature_cols = feature_df.columns.tolist()

        if len(feature_cols) == 0:
            raise ValueError("No numeric feature columns found after dropping label/attack columns.")

        X = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32).to_numpy()
        self.raw_feature_dim = int(X.shape[1])
        self.feature_cols = feature_cols

        # ------------------------------------------------------------
        # 3. Scale
        # ------------------------------------------------------------
        if scale:
            self.scaler = MinMaxScaler()
            X = self.scaler.fit_transform(X).astype(np.float32)
        else:
            self.scaler = None

        # ------------------------------------------------------------
        # 4. Pad or truncate to the selected AE-ViT input dimension
        # ------------------------------------------------------------
        if X.shape[1] < self.target_dim:
            pad_size = self.target_dim - X.shape[1]
            X = np.pad(
                X,
                pad_width=((0, 0), (0, pad_size)),
                mode="constant",
                constant_values=0.0,
            )
            self.adjustment = f"padded {self.raw_feature_dim} -> {self.target_dim}"
        elif X.shape[1] > self.target_dim:
            if not allow_truncate:
                raise ValueError(
                    f"Feature dim {X.shape[1]} is larger than target_dim={self.target_dim}. "
                    f"Increase PACKET_SHAPE or set allow_truncate=True."
                )
            X = X[:, :self.target_dim]
            self.adjustment = f"truncated {self.raw_feature_dim} -> {self.target_dim}"
        else:
            self.adjustment = "no padding/truncation"

        self.X = X.astype(np.float32)
        self.labels = labels
        self.targets = labels

        self.classes = ["benign", "attack"]
        self.class_to_idx = {"benign": 0, "attack": 1}
        self.idx_to_class = {0: "benign", 1: "attack"}

        print("[ParquetDataset] Loaded successfully")
        print("  Original parquet shape:", df.shape)
        print("  Numeric feature dim:", self.raw_feature_dim)
        print("  Target AE-ViT dim:", self.target_dim)
        print("  Adjustment:", self.adjustment)
        print("  input_mode:", self.input_mode)
        print("  packet_shape:", self.packet_shape)
        print("  Label distribution:")
        print("    benign:", int(np.sum(self.labels == 0)))
        print("    attack:", int(np.sum(self.labels == 1)))

    def __len__(self):
        return int(len(self.X))

    def __getitem__(self, idx: int):
        x = self.X[int(idx)]
        x = vec_to_tensor(
            x,
            input_mode=self.input_mode,
            packet_shape=self.packet_shape,
        )
        y = int(self.labels[int(idx)])
        return x, y


def get_label_fast(ds: Dataset, idx: int) -> int:
    """
    Fast label reader for large datasets. It avoids loading X when labels are available.
    Supports TabularParquetDataset, SplitFolderNpyDataset, Subset, and ConcatDataset.
    """
    idx = int(idx)

    if isinstance(ds, Subset):
        return get_label_fast(ds.dataset, int(ds.indices[idx]))

    if isinstance(ds, ConcatDataset):
        dataset_idx = int(np.searchsorted(ds.cumulative_sizes, idx, side="right"))
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - ds.cumulative_sizes[dataset_idx - 1]
        return get_label_fast(ds.datasets[dataset_idx], sample_idx)

    if hasattr(ds, "labels"):
        return int(ds.labels[idx])

    if hasattr(ds, "targets"):
        return int(ds.targets[idx])

    if hasattr(ds, "cum") and hasattr(ds, "file_class_idx"):
        file_i = int(np.searchsorted(ds.cum, idx, side="right") - 1)
        return int(ds.file_class_idx[file_i])

    _, y = ds[idx]
    return int(y)


class BinaryBenignAttackWrapper(Dataset):
    """Map benign labels to 1 and attack labels to -1."""
    def __init__(self, base_ds: Dataset, benign_idx: int):
        self.base_ds = base_ds
        self.benign_idx = int(benign_idx)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        y_binary = 1 if int(y) == self.benign_idx else -1
        return x, y_binary

# ============================================================
# Dataset construction logic - ensure train_subset and test_subset are returned
# ============================================================
import numpy as np
from torch.utils.data import ConcatDataset, Subset

def build_split_and_balance_dataset(
    full_ds: Optional[Dataset] = None,
    train_ds: Optional[Dataset] = None,
    test_ds: Optional[Dataset] = None,
    benign_name: str = "benign",
    split_ratio: float = 0.8,
    train_use_ratio: float = 1.0,
    test_use_ratio: float = 1.0,
    train_target_size: Optional[int] = None,
    max_test_size_per_class: Optional[int] = 10000,
    seed: int = 42
):
    """Split the data, train on benign samples, and balance the test classes."""

    rng = np.random.default_rng(seed)

    # -------------------------------------------------------
    # 0. Check the input source
    # -------------------------------------------------------
    if full_ds is not None:
        base_ds = full_ds
    elif train_ds is not None and test_ds is not None:
        base_ds = ConcatDataset([train_ds, test_ds])
    else:
        raise ValueError(
            "You must provide either full_ds, or both train_ds and test_ds."
        )

    # -------------------------------------------------------
    # 1. Retrieve class_to_idx
    # -------------------------------------------------------
    if hasattr(base_ds, "class_to_idx"):
        class_to_idx = base_ds.class_to_idx
    elif train_ds is not None and hasattr(train_ds, "class_to_idx"):
        class_to_idx = train_ds.class_to_idx
    elif test_ds is not None and hasattr(test_ds, "class_to_idx"):
        class_to_idx = test_ds.class_to_idx
    else:
        raise AttributeError(
            "Cannot find 'class_to_idx'. Please ensure dataset has class_to_idx attribute."
        )

    if benign_name not in class_to_idx:
        raise ValueError(
            f"benign_name='{benign_name}' not found in class_to_idx: {list(class_to_idx.keys())}"
        )

    benign_idx_val = class_to_idx[benign_name]

    # -------------------------------------------------------
    # 2. Validate the ratio parameters
    # -------------------------------------------------------
    if not (0 < split_ratio < 1.0):
        raise ValueError(f"split_ratio must be in (0, 1), got {split_ratio}")

    if not (0 < train_use_ratio <= 1.0):
        raise ValueError(f"train_use_ratio must be in (0, 1], got {train_use_ratio}")

    if not (0 < test_use_ratio <= 1.0):
        raise ValueError(f"test_use_ratio must be in (0, 1], got {test_use_ratio}")

    # -------------------------------------------------------
    # 3. Shuffle and split
    # -------------------------------------------------------
    all_indices = np.arange(len(base_ds))
    rng.shuffle(all_indices)

    split_point = int(len(all_indices) * split_ratio)
    idx_80 = all_indices[:split_point]
    idx_20 = all_indices[split_point:]

    # -------------------------------------------------------
    # 4. Extract benign samples from the 80% partition as training candidates
    # -------------------------------------------------------
    train_benign_idx = []
    for i in idx_80:
        y = get_label_fast(base_ds, int(i))
        if int(y) == benign_idx_val:
            train_benign_idx.append(i)

    full_train_benign_count = len(train_benign_idx)

    if full_train_benign_count == 0:
        raise RuntimeError("No benign samples found in the training region.")

    # -------------------------------------------------------
    # Added: control the training-set size
    # -------------------------------------------------------
    if train_target_size is not None:
        if train_target_size > full_train_benign_count:
            print(f"[Warn] Requested {train_target_size} training samples; using all {full_train_benign_count} benign samples")
            num_train_used = full_train_benign_count
        else:
            num_train_used = train_target_size

    else:
        num_train_used = max(1, int(full_train_benign_count * train_use_ratio))

    rng.shuffle(train_benign_idx)
    train_benign_idx = train_benign_idx[:num_train_used]

    # -------------------------------------------------------
    # 5. Build balanced test candidates from the 20% partition
    # -------------------------------------------------------
    test_benign_pool = []
    test_attack_pool = []

    for i in idx_20:
        y = get_label_fast(base_ds, int(i))
        if int(y) == benign_idx_val:
            test_benign_pool.append(i)
        else:
            test_attack_pool.append(i)

    if len(test_benign_pool) == 0 or len(test_attack_pool) == 0:
        raise RuntimeError(
            f"Cannot build balanced test set. benign={len(test_benign_pool)}, attack={len(test_attack_pool)}"
        )

    K = min(len(test_benign_pool), len(test_attack_pool))

    if K >  max_test_size_per_class:
        print(f"[Info] Test samples per class capped at {max_test_size_per_class}")
        K =  max_test_size_per_class

    if len(test_benign_pool) != len(test_attack_pool):
        print(f"[Warn] Test classes are imbalanced; using K={K}")

    rng.shuffle(test_benign_pool)
    rng.shuffle(test_attack_pool)

    test_benign_sel = test_benign_pool[:K]
    test_attack_sel = test_attack_pool[:K]

    full_test_benign_count = len(test_benign_sel)
    full_test_attack_count = len(test_attack_sel)

    # -------------------------------------------------------
    # 6. Control the proportion of test data actually used
    # -------------------------------------------------------
    num_test_benign_used = max(1, int(full_test_benign_count * test_use_ratio))
    num_test_attack_used = max(1, int(full_test_attack_count * test_use_ratio))

    rng.shuffle(test_benign_sel)
    rng.shuffle(test_attack_sel)

    test_benign_sel = test_benign_sel[:num_test_benign_used]
    test_attack_sel = test_attack_sel[:num_test_attack_used]

    test_balanced_idx = np.concatenate([test_benign_sel, test_attack_sel])
    rng.shuffle(test_balanced_idx)

    # -------------------------------------------------------
    # 7. Wrap the data as a binary-classification dataset
    # -------------------------------------------------------
    full_bin = BinaryBenignAttackWrapper(base_ds, benign_idx_val)

    train_subset = Subset(full_bin, train_benign_idx)
    test_subset = Subset(full_bin, test_balanced_idx.tolist())

    # -------------------------------------------------------
    # 8. Print the report
    # -------------------------------------------------------
    print(f"--- Dataset split ---")
    print(f"Total samples: {len(base_ds)}")
    print(f"Training partition size: {len(idx_80)}")
    print(f"Benign samples in the training partition: {full_train_benign_count}")

    if train_target_size is not None:
            print(f"Requested training size: {train_target_size}")
    else:
        print(f"Training fraction used: {train_use_ratio:.2f}")

    print(f"Final training size: {len(train_subset)}")
    print(f"Test partition size: {len(idx_20)}")
    print(f"Benign test candidates: {full_test_benign_count}")
    print(f"Attack test candidates: {full_test_attack_count}")
    print(f"Test fraction used: {test_use_ratio:.2f}")
    print(f"Final test size: {len(test_subset)}")
    print(f"benign_idx_val: {benign_idx_val}")

    return train_subset, test_subset, benign_idx_val

# ============================================================
# Dataset selection
# ============================================================

# Use "parquet" for NF-UNSW-NB15-v3.parquet tabular data.
# Use "npy_split" to keep the original train/test .npy folder behavior.
DATA_FORMAT = "parquet"       # "parquet" or "npy_split"

# -------------------------
# Parquet tabular settings
# -------------------------
# PARQUET_PATH = "../data/New Data/NF-UNSW-NB15-v3.parquet"
PARQUET_PATH = "../data/New Data/NF-CICIDS2018-v3.parquet"
LABEL_COL = "Label"
ATTACK_COL = "Attack"

# -------------------------
# Original npy split settings
# -------------------------
DATA_ROOT = "../data/New Data/CICIDS2018/data_pcaps_5"
# DATA_ROOT = "../data/New Data/NB15/feature_out_5/_MERGED_ALL_NPY"

# ============================================================
# Adjustable AE-ViT input shape
# ============================================================
# For 47 tabular features, recommended:
#   flat_1d:   PACKET_SHAPE=(1, 48), PATCH_SIZE=4
#   packet_2d: PACKET_SHAPE=(6, 8),  PATCH_SIZE=2
#
# This does not change the AE-ViT architecture class.
# It only changes how tabular features are padded/reshaped before patch embedding.
INPUT_MODE = "flat_1d"        # "flat_1d" or "packet_2d"
PACKET_SHAPE = (1, 48)        # 47 features + 1 zero padding = 48
PATCH_SIZE = 4                # 48 / 4 = 12 patches in flat_1d mode

TRAIN_TARGET_SIZE = 700000
MAX_TEST_SIZE_PER_CLASS = 10000


@dataclass
class ViTAutoencoderConfig:
    input_mode: str = "flat_1d"          # Added
    packet_shape: tuple[int, int] = (5, 96)   # Added

    seq_len: int = 480
    patch_size: int = 12
    in_chans: int = 1

    embed_dim: int = 192
    latent_dim: int = 64
    depth: int = 4
    decoder_depth: int = 2
    num_heads: int = 6
    mlp_ratio: float = 3.0
    dropout: float = 0.1
    drop_path_rate: float = 0.1
    use_cls_token: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.input_mode == "flat_1d":
            self.in_chans = 1
            self.seq_len = self.packet_shape[0] * self.packet_shape[1]   # 480

        elif self.input_mode == "packet_2d":
            self.in_chans = self.packet_shape[0]   # 5
            self.seq_len = self.packet_shape[1]    # 96

        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}")

        if self.seq_len % self.patch_size != 0:
            raise ValueError(
                f"seq_len={self.seq_len} must be divisible by patch_size={self.patch_size}"
            )

# 2. 1D Patch Embedding (process a 480-byte sequence)
class PatchEmbed1D(nn.Module):
    def __init__(self, seq_len=480, patch_size=12, in_chans=1, embed_dim=192):
        super().__init__()
        self.num_patches = seq_len // patch_size
        patch_dim = in_chans * patch_size
        self.to_patch = Rearrange('b c (n p) -> b n (p c)', p=patch_size)
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.to_patch(x)
        x = self.proj(x)
        x = self.norm(x)
        return x

# Core components (DropPath, MLP, Block)
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training: return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x / keep * mask

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, dropout, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop_path1(attn_out)
        h = self.norm2(x)
        x = x + self.drop_path2(self.mlp(h))
        return x

# 3. Integrated model: one ViT Encoder and one ViT Decoder
class ViTTrafficAutoencoder(nn.Module):
    def __init__(self, cfg: ViTAutoencoderConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed1D(cfg.seq_len, cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        n = self.patch_embed.num_patches

        # Positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n + 1, cfg.embed_dim) * 0.02)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        # --- A. Integrated ViT Encoder ---
        dpr = torch.linspace(0, cfg.drop_path_rate, steps=cfg.depth + cfg.decoder_depth).tolist()
        self.encoder = nn.ModuleList([
            EncoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, drop_path=dpr[i])
            for i in range(cfg.depth)
        ])
        self.encoder_norm = nn.LayerNorm(cfg.embed_dim)

        # Latent-feature bottleneck layer (integrates the original ae_encoder logic)
        self.bottleneck = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.embed_dim)
        )

        # --- B. ViT Decoder (using Transformer Blocks) ---
        self.decoder = nn.ModuleList([
            EncoderBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout, drop_path=dpr[cfg.depth + i])
            for i in range(cfg.decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(cfg.embed_dim)

        # Reconstruction head: restore the original patch values
        self.patch_head = nn.Linear(cfg.embed_dim, cfg.patch_size * cfg.in_chans)
        self.unpatch = Rearrange('b n (p c) -> b c (n p)', p=cfg.patch_size, c=cfg.in_chans)

    def forward(self, x: torch.Tensor):
        # 1. Encoder stage
        x_patches = self.patch_embed(x)                  # (B, 40, 192)
        cls_tokens = self.cls_token.expand(x_patches.size(0), -1, -1)
        x_patches = torch.cat((cls_tokens, x_patches), dim=1)  # (B, 41, 192)
        x_patches = x_patches + self.pos_embed
        x_patches = self.emb_dropout(x_patches)

        for blk in self.encoder:
            x_patches = blk(x_patches)
        vit_out = self.encoder_norm(x_patches)

        # Extract the core feature (CLS token)
        latent_in = vit_out[:, 0]                        # (B, 192)

        # 2. Bottleneck transformation (compression and restoration)
        latent_out = self.bottleneck(latent_in)          # (B, 192)

        # 3. Decoder stage (Transformer-based)
        # Insert the bottleneck-processed feature back into the sequence
        dec_input = vit_out.clone()
        dec_input[:, 0] = latent_out

        for blk in self.decoder:
            dec_input = blk(dec_input)
        dec_out = self.decoder_norm(dec_input)

        # 4. Reconstruct the output (restore the original 480-byte signal)
        recon_patches = self.patch_head(dec_out[:, 1:])  # (B, 40, 12)
        reconstructed_x = self.unpatch(recon_patches)    # (B, 1, 480)

        return reconstructed_x, latent_in

    def get_loss(self, reconstructed, original):
        # Compute the MSE loss between the reconstructed data and the original input
        return F.mse_loss(reconstructed, original)


# --- Training and validation functions ---

def pretrain_one_epoch(model: ViTTrafficAutoencoder,
                       loader: torch.utils.data.DataLoader,
                       optimizer: torch.optim.Optimizer,
                       cfg: ViTAutoencoderConfig,
                       scheduler=None) -> Dict[str, float]:
    model.train()
    mse_losses = []
    for x, _ in loader:
        x = x.to(cfg.device, non_blocking=True)
        # Reconstruct the original input x
        reconstructed, _ = model(x)
        loss = model.get_loss(reconstructed, x)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None: scheduler.step()
        mse_losses.append(loss.item())

    avg_loss = sum(mse_losses) / len(mse_losses)
    return {"loss": float(avg_loss), "mse": float(avg_loss)}

@torch.no_grad()
def validate(model: ViTTrafficAutoencoder,
             loader: torch.utils.data.DataLoader,
             cfg: ViTAutoencoderConfig) -> Dict[str, float]:
    model.eval()
    mse_losses = []
    for x, _ in loader:
        x = x.to(cfg.device, non_blocking=True)
        reconstructed, _ = model(x)
        loss = model.get_loss(reconstructed, x)
        mse_losses.append(loss.item())

    avg_val_loss = sum(mse_losses) / len(mse_losses)
    return {"val_loss": float(avg_val_loss), "val_mse": float(avg_val_loss)}

def build_model(cfg: Optional[ViTAutoencoderConfig] = None) -> ViTTrafficAutoencoder:
    cfg = cfg or ViTAutoencoderConfig()
    return ViTTrafficAutoencoder(cfg).to(cfg.device)




# ============================================================
# 1. Initial split (same as before)
# ============================================================
all_idx = np.arange(len(train_subset))
train_idx, val_idx = train_test_split(
    all_idx, test_size=0.2, random_state=42, shuffle=True
)

# ============================================================
# 2. Optuna objective function
# ============================================================
def objective(trial):
    # Suggest the batch size
    bs = trial.suggest_categorical("batch_size", [16,32, 64, 128, 256, 512])#
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    best_model_state = None

    # cfg = ViTAutoencoderConfig(
    #     seq_len=480,            # Length of the 1D traffic sequence (40 patches * 12 bytes)
    #     patch_size=12,          # Length of each patch
    #     in_chans=1,             # Single-channel input
    #     embed_dim=300,          # Internal Transformer feature dimension (d_model)
    #     latent_dim=64,          # Dimension after bottleneck compression
    #     depth=10,               # Number of ViT Encoder layers
    #     decoder_depth=4,        # Number of ViT Decoder layers (recommended: 2-4)
    #     num_heads=6,            # Number of attention heads (300 must be divisible by 6)
    #     mlp_ratio=5.0,          # Expansion ratio of the feed-forward network (300 * 5 = 1500)
    #     dropout=0.1,            # Dropout rate
    #     drop_path_rate=0.1,     # Stochastic depth rate
    #     use_cls_token=True,     # Use the CLS token as the global feature representation
    #     device="cuda" if torch.cuda.is_available() else "cpu"
    # )


    # Apply the values to the configuration
    # cfg = ViTAutoencoderConfig(
    #     input_mode="packet_2d",
    #     packet_shape=(5, 96),
    #     patch_size=12,      # 96 / 12 = 8 patches
    #     embed_dim=embed_dim,
    #     latent_dim=latent_dim,
    #     depth=depth ,
    #     decoder_depth=2,
    #     num_heads=6,
    #     mlp_ratio=3.0,
    #     dropout=0.1,
    #     drop_path_rate=0.1,
    # )
    # Enable Optuna search
    embed_dim = trial.suggest_categorical("embed_dim", [384])
    head_dim = trial.suggest_categorical("head_dim", [32])

    # Ensure that the MultiheadAttention configuration is valid
    if embed_dim % head_dim != 0:
        raise optuna.TrialPruned()

    num_heads = embed_dim // head_dim

    latent_dim = trial.suggest_categorical("latent_dim", [32])
    depth = trial.suggest_categorical("depth", [6])
    decoder_depth = trial.suggest_categorical("decoder_depth", [2])

    cfg = ViTAutoencoderConfig(
        input_mode=INPUT_MODE,
        packet_shape=PACKET_SHAPE,
        patch_size=PATCH_SIZE,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        depth=depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        mlp_ratio=3.0,
        dropout=0.1,
        drop_path_rate=0.1,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        Subset(train_subset, train_idx),
        batch_size=bs, shuffle=True, num_workers=2, pin_memory=True
    )

    val_loader = DataLoader(
        Subset(train_subset, val_idx),
        batch_size=bs, shuffle=False, num_workers=2, pin_memory=True
    )

    # Create the model and optimizer
    xb, yb = next(iter(train_loader))
    print("Batch input shape:", xb.shape)
    print("cfg.input_mode:", cfg.input_mode)
    print("cfg.in_chans:", cfg.in_chans)
    print("cfg.seq_len:", cfg.seq_len)
    print("cfg.patch_size:", cfg.patch_size)


    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # Configure the scheduler (cosine decay)
    total_steps = epochs * len(train_loader)
    warmup = int(0.05 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda step: min(1., step / max(1, warmup)) *
            0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
    )

    # --- Note: the Autoencoder does not require init_center_c ---

    best = float("inf")
    patience = 10
    bad = 0

    # Training loop
    for epoch in range(epochs):
        # Use the pretraining function adapted for the AE
        meters = pretrain_one_epoch(model, train_loader, opt, cfg, scheduler=scheduler)
        # Use the validation function adapted for the AE
        val = validate(model, val_loader, cfg)

        print(f"[{epoch:03d}] MSE_Loss={meters['loss']:.6f}  Val_MSE={val['val_loss']:.6f}")

        if val["val_loss"] < best - 1e-6:
            best = val["val_loss"]
            bad = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    validation_scores, _ = score_reconstruction(model, val_loader, cfg.device)
    threshold = select_validation_threshold(validation_scores)
    trial.set_user_attr("validation_threshold", threshold)
    trial.set_user_attr("epochs", epoch + 1)

    # Save the model
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": cfg.__dict__,
        "validation_threshold": threshold,
    }, f'{save_dir}/models/AE_trial_{trial.number}.pt')

    del model
    torch.cuda.empty_cache()

    return best

if __name__ == "__main__":
    if DATA_FORMAT == "parquet":
        full_ds = TabularParquetDataset(
            parquet_path=PARQUET_PATH,
            label_col=LABEL_COL,
            attack_col=ATTACK_COL,
            input_mode=INPUT_MODE,
            packet_shape=PACKET_SHAPE,
            scale=True,
            allow_truncate=False,
            max_rows=None,        # set a small value for quick debugging, e.g. 10000
        )

        train_subset, test_subset, benign_idx_train = build_split_and_balance_dataset(
            full_ds=full_ds,
            train_ds=None,
            test_ds=None,
            benign_name="benign",
            split_ratio=0.7,
            train_use_ratio=1.0,
            test_use_ratio=1.0,
            train_target_size=TRAIN_TARGET_SIZE,
            max_test_size_per_class=MAX_TEST_SIZE_PER_CLASS,
            seed=42
        )

    elif DATA_FORMAT == "npy_split":
        train_ds = SplitFolderNpyDataset(
            Path(DATA_ROOT) / "train",
            use_label_file=False,
            input_mode=INPUT_MODE,
            packet_shape=PACKET_SHAPE,
        )

        test_ds = SplitFolderNpyDataset(
            Path(DATA_ROOT) / "test",
            use_label_file=False,
            input_mode=INPUT_MODE,
            packet_shape=PACKET_SHAPE,
        )

        train_subset, test_subset, benign_idx_train = build_split_and_balance_dataset(
            full_ds=None,
            train_ds=train_ds,
            test_ds=test_ds,
            benign_name="benign",
            split_ratio=0.7,
            train_use_ratio=1.0,
            test_use_ratio=1.0,
            train_target_size=TRAIN_TARGET_SIZE,
            max_test_size_per_class=MAX_TEST_SIZE_PER_CLASS,
            seed=42
        )

    else:
        raise ValueError(f"Unknown DATA_FORMAT: {DATA_FORMAT}")

    epochs = 50

    shape_tag = f"{INPUT_MODE}_{PACKET_SHAPE[0]}x{PACKET_SHAPE[1]}_patch{PATCH_SIZE}"
    if DATA_FORMAT == "parquet":
        experiment_root = f"./results_AEViT_tabular_parquet_{shape_tag}"
    else:
        experiment_root = f"./results_AEViT_npy_{shape_tag}"

    save_dir = f"{experiment_root}/AEViT/optuna_validation_mse/"
    study_storage = f"{experiment_root}/AEViT.db"
    study_name = f"AEViT-{DATA_FORMAT}-{shape_tag}-validation-mse"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        sampler=TPESampler(n_startup_trials=10, n_ei_candidates=24, multivariate=True),
        storage=f'sqlite:///{study_storage}',
        load_if_exists=True
    )

    study.optimize(objective,n_trials=1)

    best_trial = study.best_trial
    checkpoint_path = os.path.join(save_dir, "models", f"AE_trial_{best_trial.number}.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = ViTAutoencoderConfig(**checkpoint["cfg"])
    model = build_model(cfg).to(cfg.device)
    model.load_state_dict(checkpoint["state_dict"])

    test_loader = DataLoader(
        test_subset,
        batch_size=best_trial.params["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_scores, test_labels = score_reconstruction(model, test_loader, cfg.device)
    test_metrics = evaluate_test_data(
        test_labels, test_scores, checkpoint["validation_threshold"]
    )
    pd.DataFrame([test_metrics]).to_csv(
        os.path.join(save_dir, "best_trial_test_metrics.csv"), index=False
    )

    results = study.trials_dataframe()
    results = results.loc[results["state"] == "COMPLETE"].sort_values("value")
    results.to_csv(f'{save_dir}/result_AEViT.csv', index=False)

    print("Saved results to:", f'{save_dir}/result_AEViT.csv')
