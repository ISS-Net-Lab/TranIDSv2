# aevit_finetune_attack_ratio_auto_common_attack_pool_fixed_trial.py

import os
import gc
import time
import random
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset

from transformer_AEViT_1D import (
    ViTAutoencoderConfig,
    build_model,
    SplitFolderNpyDataset,
    build_split_and_balance_dataset
)

from util.data_preparing import (
    select_validation_threshold,
    evaluate_test_data,
    plot_confusion_matrix
)

from util.clean_ram_npy import (
    drop_large_npy_file_cache,
    show_mem
)


# ============================================================
# Basic settings
# ============================================================

SPLIT_SEED = 42

# For initial testing, use a single seed
# SEEDS = [7]
SEEDS = [7, 17, 27, 37, 47]

# ATTACK_RATIOS = [1, 2, 5, 7, 10, 15, 20, 25, 30]
ATTACK_RATIOS = [2, 5, 7, 10, 15, 20, 25, 30]
# ATTACK_RATIOS = [30]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================================================
# Robust fine-tune loss setting
# ============================================================

# "naive"   = all samples contribute to the reconstruction loss, as in the original version
# "trimmed" = in each batch, update using only samples with lower reconstruction errors
TRAIN_LOSS_MODE = "naive"

# keep_ratio = proportion of low-reconstruction-error samples to retain
# For example, 0.9 retains the 90% of samples with the lowest errors and discards the highest 10%
TRIM_KEEP_RATIO = 0.97

if torch.cuda.is_available():
    print("CUDA available.")
    print("GPU count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
else:
    print("No GPU found, using CPU")


# ============================================================
# AEViT input setting
# ============================================================

INPUT_MODE = "packet_2d"
PACKET_SHAPE = (5, 96)

# If class_to_idx is {"Benign": ...}, it will be detected automatically.
# This is only the preferred name.
BENIGN_NAME = "benign"

# ============================================================
# Fixed pretrain trial settings
# ============================================================
# When source_name = CICIDS2018, always use CICIDS2018 pretraining trial 6
# When source_name = NB15, always use NB15 pretraining trial 2

PRETRAIN_FIXED_TRIALS = {
    "CICIDS2018": 6,
    "NB15": 2,
}



# ============================================================
# Fine-tune layer configs
# ============================================================

FT_LAYER_CONFIGS = {
    # 1: {
    #     "base_modules": ["patch_head", "decoder_norm"],
    #     "num_decoder_blocks": 0,
    #     "num_encoder_blocks": 0,
    # },
    # 2: {
    #     "base_modules": ["patch_head", "decoder_norm"],
    #     "num_decoder_blocks": 1,
    #     "num_encoder_blocks": 0,
    # },
        1: {
        "base_modules": [],
        "num_decoder_blocks": 0,
        "num_encoder_blocks": 1,
    },
    # 3: {
    #     "base_modules": ["patch_head", "decoder_norm"],
    #     "num_decoder_blocks": 2,
    #     "num_encoder_blocks": 0,
    # },
    # 5: {
    #     "base_modules": ["patch_head", "decoder_norm", "bottleneck"],
    #     "num_decoder_blocks": 2,
    #     "num_encoder_blocks": 1,
    # },
}


# ============================================================
# Paths
# ============================================================

RESULTS_ROOT = Path("./results_attack_ratio_AEViT")



AZURE_BASE_PREFIX = "pcap_experiments/c6"


# ============================================================
# Pretrain model result folders
# AEViT has no K, so use only the dataset name as the key
# ============================================================

PRETRAIN_RUNS = {
    "CICIDS2018": Path("./results_AEViT_image_2018_96_5_1000000_1"),
    "NB15": Path("./results_AEViT_image_NB15_96_5_1000000_1"),
}


# ============================================================
# Target fine-tune data folders
# AEViT has no K, so use only the dataset name as the key
# ============================================================

TARGET_DATA = {
    # CICIDS2018:
    # target_data_dir/
    #   train/
    #   test/
    "CICIDS2018": Path("../data/New Data/CICIDS2018/data_pcaps_5"),

    # NB15:
    # target_data_dir/
    #   full dataset files
    "NB15": Path("../data/New Data/NB15/feature_out_5/_MERGED_ALL_NPY"),
}


# ============================================================
# Output root
# ============================================================

FT_OUTPUT_ROOT = RESULTS_ROOT / "finetune_attack_ratio_1layer_opposite"
FT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================

def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def set_global_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True, warn_only=True)


def make_worker(seed_base: int):
    def _seed_worker(worker_id):
        s = seed_base + worker_id
        np.random.seed(s)
        random.seed(s)

    return _seed_worker


def fmt_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def calc_val_size_from_train_size(train_size: int, val_ratio: float) -> int:
    """Compute the validation size while keeping the benign training count fixed."""
    if train_size <= 0:
        raise ValueError(f"train_size must be positive, got {train_size}")

    if not (0 <= val_ratio < 1):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    if val_ratio == 0:
        return 0

    return int(round(train_size * val_ratio / (1.0 - val_ratio)))


def release_memory(objects_to_delete=None, cache_roots=None):
    if objects_to_delete is not None:
        for obj in objects_to_delete:
            try:
                del obj
            except Exception:
                pass

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if cache_roots is not None:
        try:
            drop_large_npy_file_cache(
                roots=cache_roots,
                min_size_gb=0.1,
                pattern="*.npy",
                show_before=True,
                show_after=True,
                sync_before=True,
                sync_after=True,
                verbose=True
            )
        except Exception as e:
            print("[Warn] drop_large_npy_file_cache failed:", e)

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def normalize_state_dict_keys(state_dict):
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            k.replace("module.", "", 1): v
            for k, v in state_dict.items()
        }

    return state_dict


def get_state_dict_from_checkpoint(ckpt):
    """Extract model weights from supported checkpoint formats."""
    if not isinstance(ckpt, dict):
        return ckpt

    for key in ["state_dict", "model_state_dict", "model"]:
        if key in ckpt:
            return ckpt[key]

    tensor_like = all(torch.is_tensor(v) for v in ckpt.values())

    if tensor_like:
        return ckpt

    raise RuntimeError(
        "Could not extract state_dict from the checkpoint."
        f"checkpoint keys: {list(ckpt.keys())}"
    )


# ============================================================
# AEViT config / checkpoint
# ============================================================

def build_aevit_config_from_trial_row(trial_row):
    """Build the model configuration from an Optuna trial."""

    embed_dim = int(trial_row["params_embed_dim"])
    head_dim = int(trial_row["params_head_dim"])
    latent_dim = int(trial_row["params_latent_dim"])
    depth = int(trial_row["params_depth"])
    decoder_depth = int(trial_row["params_decoder_depth"])

    if embed_dim % head_dim != 0:
        raise ValueError(
            f"embed_dim must be divisible by head_dim;"
            f" got embed_dim={embed_dim}, head_dim={head_dim}"
        )

    num_heads = embed_dim // head_dim

    cfg = ViTAutoencoderConfig(
        input_mode=INPUT_MODE,
        packet_shape=PACKET_SHAPE,
        patch_size=12,
        embed_dim=embed_dim,
        latent_dim=latent_dim,
        depth=depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        mlp_ratio=3.0,
        dropout=0.1,
        drop_path_rate=0.1,
    )

    return cfg


def find_top_aevit_pretrain_trials(
    pretrain_run_dir: Path,
    top_k: int = 1,
    fixed_trial_number: Optional[int] = None,
):
    """Select a fixed trial or take the top trials by score when available."""

    candidate_csvs = [
        pretrain_run_dir / "AEViT_image" / "optuna" / "result_imgAEViT.csv",
        pretrain_run_dir / "AEViT_image" / "optuna" / "result_AEViT.csv",
        pretrain_run_dir / "AEViT" / "optuna" / "result_imgAEViT.csv",
        pretrain_run_dir / "AEViT" / "optuna" / "result_AEViT.csv",
        pretrain_run_dir / "optuna" / "result_imgAEViT.csv",
        pretrain_run_dir / "optuna" / "result_AEViT.csv",

        # Maintain compatibility with legacy folder names that may still be in use
        pretrain_run_dir / "SVDD_image" / "optuna" / "result_imgSVDD.csv",
    ]

    result_csv = None

    for p in candidate_csvs:
        if p.exists():
            result_csv = p
            break

    if result_csv is None:
        raise FileNotFoundError(
            "Cannot find AEViT result csv. Tried:\n"
            + "\n".join(str(p) for p in candidate_csvs)
        )

    optuna_dir = result_csv.parent
    model_dir = optuna_dir / "models"

    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model dir: {model_dir}")

    df = pd.read_csv(result_csv)

    required_cols = [
        "number",
        "params_batch_size",
        "params_embed_dim",
        "params_head_dim",
        "params_latent_dim",
        "params_depth",
        "params_decoder_depth",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise RuntimeError(
            f"Result CSV is missing required columns: {missing_cols}\n"
            f"CSV path: {result_csv}\n"
            f"Existing columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Fixed trial mode
    # --------------------------------------------------------
    if fixed_trial_number is not None:
        fixed_trial_number = int(fixed_trial_number)

        df["_number_int"] = df["number"].astype(int)
        top_trials = df[df["_number_int"] == fixed_trial_number].copy()

        if len(top_trials) == 0:
            available_trials = sorted(df["_number_int"].unique().tolist())
            raise ValueError(
                f"fixed_trial_number={fixed_trial_number} not found in {result_csv}\n"
                f"Available trials: {available_trials}"
            )

        top_trials = top_trials.drop(columns=["_number_int"])

        print("\n[Fixed Trial Selection]")
        print("Pretrain run dir:", pretrain_run_dir)
        print("Result CSV:", result_csv)
        print("Fixed trial number:", fixed_trial_number)
        print(top_trials)

        return {
            "result_csv": result_csv,
            "model_dir": model_dir,
            "top_trials": top_trials,
        }

    # --------------------------------------------------------
    # Auto top-k mode
    # --------------------------------------------------------
    if "value" in df.columns:
        df = df.dropna(subset=["value"]).copy()
        lower_is_better = "user_attrs_validation_threshold" in df.columns
        df = df.sort_values(by="value", ascending=lower_is_better).reset_index(drop=True)

    top_trials = df.head(top_k).copy()

    print("\n[Auto Top-K Trial Selection]")
    print("Pretrain run dir:", pretrain_run_dir)
    print("Result CSV:", result_csv)
    print("top_k:", top_k)
    print(top_trials)

    return {
        "result_csv": result_csv,
        "model_dir": model_dir,
        "top_trials": top_trials,
    }


def find_aevit_checkpoint_by_trial(
    model_dir: Path,
    trial_number: int
) -> Path:
    """Find the checkpoint for a trial number."""

    candidate_ckpts = [
        model_dir / f"AEViT_trial_{trial_number}.pt",
        model_dir / f"AEViT_trial_{trial_number}.pth",
        model_dir / f"autoencoder_trial_{trial_number}.pt",
        model_dir / f"autoencoder_trial_{trial_number}.pth",
        model_dir / f"model_trial_{trial_number}.pt",
        model_dir / f"model_trial_{trial_number}.pth",
        model_dir / f"AE_trial_{trial_number}.pt",
        # If the AEViT checkpoint still uses the old naming convention
        model_dir / f"SVDD_trial_{trial_number}.pt",
    ]

    for p in candidate_ckpts:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Cannot find checkpoint for trial {trial_number}. Tried:\n"
        + "\n".join(str(p) for p in candidate_ckpts)
    )


# ============================================================
# Dataset helper functions
# ============================================================

class BinaryBenignAttackWrapperLocal(Dataset):
    """Map raw dataset labels to 1 for benign and -1 for attacks."""
    def __init__(self, base_ds: Dataset, benign_idx: int):
        self.base_ds = base_ds
        self.benign_idx = int(benign_idx)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[int(idx)]
        y_binary = 1 if int(y) == self.benign_idx else -1
        return x, y_binary


def get_label_by_getitem(ds, idx: int) -> int:
    """Read the label from a dataset item."""
    _, y = ds[int(idx)]
    return int(y)


def get_subset_labels_by_getitem(subset: Subset) -> np.ndarray:
    """Read labels from subset items, including binary labels from wrappers."""
    labels = []

    for i in range(len(subset)):
        _, y = subset[i]
        labels.append(int(y))

    return np.asarray(labels, dtype=np.int64)


def get_class_to_idx(ds):
    if hasattr(ds, "class_to_idx"):
        return ds.class_to_idx

    if hasattr(ds, "classes"):
        classes = list(ds.classes)
        return {name: i for i, name in enumerate(classes)}

    raise AttributeError(
        "Cannot identify the benign class without class_to_idx or classes."
    )


def get_idx_to_class(ds):
    if hasattr(ds, "idx_to_class"):
        return ds.idx_to_class

    class_to_idx = get_class_to_idx(ds)
    return {v: k for k, v in class_to_idx.items()}


def resolve_benign_name(class_to_idx, preferred="benign"):
    """Find the benign class name without regard to case."""
    if preferred in class_to_idx:
        return preferred

    for name in ["benign", "Benign", "BENIGN"]:
        if name in class_to_idx:
            return name

    raise ValueError(f"Cannot find benign class in class_to_idx: {class_to_idx}")


def get_base_dataset_from_binary_wrapper(binary_dataset):
    """Recover the underlying dataset and its benign class index."""
    if hasattr(binary_dataset, "base_ds"):
        return binary_dataset.base_ds

    return binary_dataset


def get_raw_label_fast(ds, idx: int) -> int:
    """Use label metadata where available and fall back to loading the item."""

    idx = int(idx)

    # Case 1: Subset
    if isinstance(ds, Subset):
        real_idx = int(ds.indices[idx])
        return get_raw_label_fast(ds.dataset, real_idx)

    # Case 2: ConcatDataset
    if isinstance(ds, ConcatDataset):
        cumulative_sizes = ds.cumulative_sizes
        dataset_idx = int(np.searchsorted(cumulative_sizes, idx, side="right"))

        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - cumulative_sizes[dataset_idx - 1]

        return get_raw_label_fast(ds.datasets[dataset_idx], sample_idx)

    # Case 3: SplitFolderNpyDataset style
    if hasattr(ds, "cum") and hasattr(ds, "file_class_idx"):
        file_i = int(np.searchsorted(ds.cum, idx, side="right") - 1)
        return int(ds.file_class_idx[file_i])

    # Case 4: labels-like attributes
    for attr in ["y", "labels", "targets"]:
        if hasattr(ds, attr):
            arr = getattr(ds, attr)
            return int(arr[idx])

    # Fallback
    _, y = ds[idx]
    return int(y)


def get_binary_label_fast(binary_dataset, idx: int) -> int:
    """Get the binary label without loading features when possible."""

    idx = int(idx)

    if hasattr(binary_dataset, "base_ds") and hasattr(binary_dataset, "benign_idx"):
        base_ds = binary_dataset.base_ds
        benign_idx = int(binary_dataset.benign_idx)

        raw_label = get_raw_label_fast(base_ds, idx)

        return 1 if int(raw_label) == benign_idx else -1

    _, y = binary_dataset[idx]
    return int(y)


def get_subset_labels_fast(subset: Subset) -> np.ndarray:
    """Read subset labels without loading features when possible."""

    labels = []
    base_ds = subset.dataset

    for idx in subset.indices:
        if hasattr(base_ds, "base_ds") and hasattr(base_ds, "benign_idx"):
            labels.append(get_binary_label_fast(base_ds, int(idx)))
        else:
            labels.append(get_raw_label_fast(base_ds, int(idx)))

    return np.asarray(labels, dtype=np.int64)


def check_subset_label_distribution_fast(subset, name):
    """Check the subset label distribution using label metadata where possible."""

    labels = get_subset_labels_fast(subset)

    unique, counts = np.unique(labels, return_counts=True)

    print(f"\n{name} label distribution:")

    for u, c in zip(unique, counts):
        if int(u) == 1:
            label_name = "+1 / benign"
        elif int(u) == -1:
            label_name = "-1 / attack"
        else:
            label_name = str(int(u))

        print(f"  {label_name}: {int(c)}")

    return dict(zip(unique.tolist(), counts.tolist()))


def load_aevit_base_dataset(
    dataset_name: str,
    data_dir: Path,
    input_mode: str = "packet_2d",
    packet_shape: Tuple[int, int] = (5, 96),
    benign_name: str = "benign",
):
    """Load a full dataset or concatenate its train and test folders."""

    dataset_name_upper = dataset_name.upper()
    data_dir = Path(data_dir)

    if dataset_name_upper == "NB15":
        base_ds = SplitFolderNpyDataset(
            data_dir,
            use_label_file=False,
            input_mode=input_mode,
            packet_shape=packet_shape,
        )

        detected_benign_name = resolve_benign_name(
            get_class_to_idx(base_ds),
            preferred=benign_name
        )

        benign_idx_val = int(get_class_to_idx(base_ds)[detected_benign_name])

    elif dataset_name_upper == "CICIDS2018":
        train_ds = SplitFolderNpyDataset(
            data_dir / "train",
            use_label_file=False,
            input_mode=input_mode,
            packet_shape=packet_shape,
        )

        test_ds = SplitFolderNpyDataset(
            data_dir / "test",
            use_label_file=False,
            input_mode=input_mode,
            packet_shape=packet_shape,
        )

        detected_benign_name = resolve_benign_name(
            get_class_to_idx(train_ds),
            preferred=benign_name
        )

        benign_idx_val = int(get_class_to_idx(train_ds)[detected_benign_name])
        base_ds = ConcatDataset([train_ds, test_ds])

    else:
        raise ValueError(
            f"Unsupported dataset_name={dataset_name}. "
            f"Only support 'NB15' and 'CICIDS2018'."
        )

    binary_ds = BinaryBenignAttackWrapperLocal(
        base_ds=base_ds,
        benign_idx=benign_idx_val
    )

    return {
        "dataset_name": dataset_name,
        "base_ds": base_ds,
        "binary_ds": binary_ds,
        "benign_name": detected_benign_name,
        "benign_idx_val": benign_idx_val,
    }


def collect_attack_indices_from_binary_dataset(
    binary_dataset,
    exclude_indices=None,
    max_count: Optional[int] = None,
):
    """Collect attack indices without loading features when possible."""

    if exclude_indices is None:
        exclude_set = set()
    else:
        exclude_set = set(int(i) for i in exclude_indices)

    attack_indices = []

    for idx in range(len(binary_dataset)):
        if idx in exclude_set:
            continue

        y = get_binary_label_fast(binary_dataset, idx)

        if int(y) == -1:
            attack_indices.append(idx)

            if max_count is not None and len(attack_indices) >= max_count:
                break

    return np.asarray(attack_indices, dtype=np.int64)


def prepare_aevit_target_data(
    target_name: str,
    target_data_dir: Path,
    input_mode: str = "packet_2d",
    packet_shape: Tuple[int, int] = (5, 96),
    benign_name: str = "benign",
    split_ratio: float = 0.7,
    train_use_ratio: float = 1.0,
    test_use_ratio: float = 1.0,
    train_target_size: Optional[int] = 150000,
    max_test_size_per_class: Optional[int] = 10000,
    seed: int = 42,
):
    """Build target splits and return the benign class label with the subsets."""

    target_name_upper = target_name.upper()
    target_data_dir = Path(target_data_dir)

    print("\n" + "=" * 80)
    print("Prepare AEViT target data")
    print("=" * 80)
    print("target_name:", target_name)
    print("target_data_dir:", target_data_dir)
    print("input_mode:", input_mode)
    print("packet_shape:", packet_shape)
    print("preferred benign_name:", benign_name)
    print("train_target_size:", train_target_size)
    print("max_test_size_per_class:", max_test_size_per_class)

    if target_name_upper == "NB15":

        full_ds = SplitFolderNpyDataset(
            target_data_dir,
            use_label_file=False,
            input_mode=input_mode,
            packet_shape=packet_shape,
        )

        detected_benign_name = resolve_benign_name(
            get_class_to_idx(full_ds),
            preferred=benign_name
        )

        train_ds = None
        test_ds = None

        train_subset, test_subset, benign_idx_val = build_split_and_balance_dataset(
            full_ds=full_ds,
            train_ds=None,
            test_ds=None,
            benign_name=detected_benign_name,
            split_ratio=split_ratio,
            train_use_ratio=train_use_ratio,
            test_use_ratio=test_use_ratio,
            train_target_size=train_target_size,
            max_test_size_per_class=max_test_size_per_class,
            seed=seed
        )

        base_train_ds = full_ds
        base_test_ds = full_ds

    elif target_name_upper == "CICIDS2018":

        train_ds = SplitFolderNpyDataset(
            target_data_dir / "train",
            use_label_file=False,
            input_mode=input_mode,
            packet_shape=packet_shape,
        )

        test_ds = SplitFolderNpyDataset(
            target_data_dir / "test",
            use_label_file=False,
            input_mode=input_mode,
            packet_shape=packet_shape,
        )

        detected_benign_name = resolve_benign_name(
            get_class_to_idx(train_ds),
            preferred=benign_name
        )

        full_ds = None

        train_subset, test_subset, benign_idx_val = build_split_and_balance_dataset(
            full_ds=None,
            train_ds=train_ds,
            test_ds=test_ds,
            benign_name=detected_benign_name,
            split_ratio=split_ratio,
            train_use_ratio=train_use_ratio,
            test_use_ratio=test_use_ratio,
            train_target_size=train_target_size,
            max_test_size_per_class=max_test_size_per_class,
            seed=seed
        )

        base_train_ds = train_ds
        base_test_ds = test_ds

    else:
        raise ValueError(
            f"Unsupported target_name={target_name}. "
            f"Only support 'NB15' and 'CICIDS2018'."
        )

    y_test_tgt = get_subset_labels_fast(test_subset)

    print("\nAEViT target data prepared.")
    print("detected benign_name:", detected_benign_name)
    print("Train subset:", len(train_subset))
    print("Test subset:", len(test_subset))
    print("Benign class index value:", benign_idx_val)
    print("y_test_tgt distribution:")
    print("  Benign(+1):", int(np.sum(y_test_tgt == 1)))
    print("  Attack(-1):", int(np.sum(y_test_tgt == -1)))

    check_subset_label_distribution_fast(
        subset=train_subset,
        name="target_train_subset"
    )

    check_subset_label_distribution_fast(
        subset=test_subset,
        name="target_test_subset"
    )

    return {
        "full_ds": full_ds,
        "train_ds": train_ds,
        "test_ds": test_ds,
        "base_train_ds": base_train_ds,
        "base_test_ds": base_test_ds,
        "train_subset": train_subset,
        "test_subset": test_subset,
        "benign_idx_val": benign_idx_val,
        "y_test_tgt": y_test_tgt,
        "benign_name": detected_benign_name,
    }


def split_train_subset_into_train_val_indices(
    train_subset: Subset,
    train_benign_size: int,
    val_ratio: float = 0.2,
    seed: int = 42
):
    """Reserve validation samples while keeping the requested training count."""

    original_indices = np.asarray(train_subset.indices, dtype=np.int64)

    val_size = calc_val_size_from_train_size(
        train_size=int(train_benign_size),
        val_ratio=val_ratio
    )

    total_needed = int(train_benign_size) + int(val_size)

    if len(original_indices) < total_needed:
        raise RuntimeError(
            f"Not enough benign samples for requested train/val split. "
            f"Need train={train_benign_size}, val={val_size}, total={total_needed}, "
            f"but train_subset has only {len(original_indices)}."
        )

    rng = np.random.default_rng(seed)
    shuffled = original_indices.copy()
    rng.shuffle(shuffled)

    train_indices = shuffled[:int(train_benign_size)]
    val_indices = shuffled[int(train_benign_size):total_needed]

    train_indices.sort()
    val_indices.sort()

    base_bin_ds = train_subset.dataset

    train_part = Subset(base_bin_ds, train_indices)
    val_part = Subset(base_bin_ds, val_indices)

    return train_part, val_part


def build_mixed_finetune_dataset_with_priority_attack_pool(
    benign_train_subset: Subset,
    target_test_subset: Subset,
    val_subset: Subset,
    attack_ratio_pct: float,
    attack_sources_ordered: List[Dict[str, object]],
    seed: int = 42,
    allow_attack_oversample: bool = False,
):
    """Build the fine-tuning dataset using source, target, then other attack samples."""

    rng = np.random.default_rng(seed)

    benign_indices = np.asarray(benign_train_subset.indices, dtype=np.int64)
    val_indices = np.asarray(val_subset.indices, dtype=np.int64)
    test_indices = np.asarray(target_test_subset.indices, dtype=np.int64)

    n_benign = len(benign_indices)
    n_attack = int(round(n_benign * attack_ratio_pct / 100.0))

    # This exclusion applies only to the target/fine-tuning dataset
    # Prevent fine-tuning benign, validation, or test samples from being reused as injected attacks
    exclude_target = set(benign_indices.tolist())
    exclude_target.update(val_indices.tolist())
    exclude_target.update(test_indices.tolist())

    selected_attack_datasets = []
    selected_attack_indices_by_source = {}
    source_attack_counts = {}
    source_attack_pool_counts = {}

    remain = n_attack

    # ========================================================
    # Sample attacks according to the order in attack_sources_ordered
    # ========================================================
    for source in attack_sources_ordered:
        if remain <= 0:
            break

        source_name = str(source["dataset_name"])
        source_binary_dataset = source["binary_ds"]
        is_target_dataset = bool(source.get("is_target_dataset", False))

        if is_target_dataset:
            exclude_indices = exclude_target
        else:
            exclude_indices = None

        source_attack_pool = collect_attack_indices_from_binary_dataset(
            binary_dataset=source_binary_dataset,
            exclude_indices=exclude_indices,
            max_count=None
        )

        source_attack_pool_counts[source_name] = int(len(source_attack_pool))

        rng.shuffle(source_attack_pool)

        take = min(remain, len(source_attack_pool))
        selected = source_attack_pool[:take]

        if take > 0:
            selected_attack_datasets.append(
                Subset(source_binary_dataset, selected.tolist())
            )

        selected_attack_indices_by_source[source_name] = selected
        source_attack_counts[source_name] = int(take)

        remain -= take

    # ========================================================
    # If all pools are insufficient, determine whether to oversample
    # ========================================================
    if remain > 0:
        if not allow_attack_oversample:
            raise ValueError(
                f"Not enough attack samples from ordered attack pools. "
                f"Need {n_attack}, selected {n_attack - remain}, still missing {remain}. "
                f"source_attack_counts={source_attack_counts}. "
                f"source_attack_pool_counts={source_attack_pool_counts}. "
                f"Set allow_attack_oversample=True if acceptable."
            )

        oversample_candidates = []

        for source in attack_sources_ordered:
            source_name = str(source["dataset_name"])
            source_binary_dataset = source["binary_ds"]
            is_target_dataset = bool(source.get("is_target_dataset", False))

            if is_target_dataset:
                exclude_indices = exclude_target
            else:
                exclude_indices = None

            source_attack_pool = collect_attack_indices_from_binary_dataset(
                binary_dataset=source_binary_dataset,
                exclude_indices=exclude_indices,
                max_count=None
            )

            if len(source_attack_pool) > 0:
                oversample_candidates.append(
                    (source_name, source_binary_dataset, source_attack_pool)
                )

        if len(oversample_candidates) == 0:
            raise RuntimeError("No attack samples available for oversampling.")

        # Oversample from the first available source according to the priority order
        oversample_name, oversample_dataset, oversample_pool = oversample_candidates[0]

        oversampled = rng.choice(
            oversample_pool,
            size=remain,
            replace=True
        )

        selected_attack_datasets.append(
            Subset(oversample_dataset, oversampled.tolist())
        )

        selected_attack_indices_by_source[f"oversample_from_{oversample_name}"] = oversampled
        source_attack_counts[f"oversample_from_{oversample_name}"] = int(remain)

        remain = 0

    # ========================================================
    # Assemble the fine-tuning dataset
    # ========================================================
    datasets_for_train = [benign_train_subset]
    datasets_for_train.extend(selected_attack_datasets)

    ft_train_dataset = ConcatDataset(datasets_for_train)

    total_attack_selected = sum(source_attack_counts.values())

    first_source_name = str(attack_sources_ordered[0]["dataset_name"])

    info = {
        "benign_train_count": int(n_benign),
        "attack_train_count": int(total_attack_selected),
        "attack_ratio_pct": float(attack_ratio_pct),
        "total_ft_train_count": int(n_benign + total_attack_selected),

        "attack_source_priority": [
            str(s["dataset_name"]) for s in attack_sources_ordered
        ],

        "source_attack_counts": source_attack_counts,
        "source_attack_pool_counts": source_attack_pool_counts,

        "primary_attack_source": first_source_name,
        "primary_attack_count": int(source_attack_counts.get(first_source_name, 0)),
        "primary_attack_pool_count": int(source_attack_pool_counts.get(first_source_name, 0)),

        "fallback_attack_used": bool(
            any(k != first_source_name and v > 0 for k, v in source_attack_counts.items())
        ),

        # Preserve compatibility with the existing field names used later for saving
        "target_attack_pool_count": int(
            sum(
                count
                for name, count in source_attack_pool_counts.items()
                if "fallback_target" in name
            )
        ),
        "common_attack_used": bool(
            any(k != first_source_name and v > 0 for k, v in source_attack_counts.items())
        ),
    }

    return ft_train_dataset, selected_attack_indices_by_source, info
# ============================================================
# Fine-tune layer unfreeze
# ============================================================

def unfreeze_aevit_by_ft_config(
    model,
    ft_cfg: Dict[str, object],
):
    """Unfreeze model parameters selected by the fine-tuning configuration."""

    for p in model.parameters():
        p.requires_grad = False

    base_modules = ft_cfg.get("base_modules", [])
    num_decoder_blocks = int(ft_cfg.get("num_decoder_blocks", 0))
    num_encoder_blocks = int(ft_cfg.get("num_encoder_blocks", 0))

    print("\nUnfreeze setting:")
    print("base_modules:", base_modules)
    print("num_decoder_blocks:", num_decoder_blocks)
    print("num_encoder_blocks:", num_encoder_blocks)

    # --------------------------------------------------------
    # 1. Unfreeze the base modules
    # --------------------------------------------------------
    for module_name in base_modules:
        if not hasattr(model, module_name):
            print(f"[Warn] Model has no module: {module_name}")
            continue

        module = getattr(model, module_name)

        if isinstance(module, torch.nn.Parameter):
            module.requires_grad = True

        elif hasattr(module, "parameters"):
            for p in module.parameters():
                p.requires_grad = True

        else:
            print(f"[Warn] {module_name} is neither a Parameter nor a Module; skipping")

    # --------------------------------------------------------
    # 2. Unfreeze the last num_decoder_blocks decoder blocks
    # --------------------------------------------------------
    decoder_block_attr_candidates = [
        "decoder_blocks",
        "decoder_block",
        "decoder"
    ]

    decoder_blocks = None

    for attr in decoder_block_attr_candidates:
        if hasattr(model, attr):
            obj = getattr(model, attr)

            if isinstance(obj, torch.nn.ModuleList):
                decoder_blocks = obj
                break

            if isinstance(obj, torch.nn.Sequential):
                decoder_blocks = obj
                break

    if num_decoder_blocks > 0:
        if decoder_blocks is None:
            print("[Warn] No decoder_blocks or decoder found to unfreeze")
        else:
            for blk in list(decoder_blocks)[-num_decoder_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

    # --------------------------------------------------------
    # 3. Unfreeze the last num_encoder_blocks encoder blocks
    # --------------------------------------------------------
    encoder_block_attr_candidates = [
        "blocks",
        "encoder_blocks",
        "encoder"
    ]

    encoder_blocks = None

    for attr in encoder_block_attr_candidates:
        if hasattr(model, attr):
            obj = getattr(model, attr)

            if isinstance(obj, torch.nn.ModuleList):
                encoder_blocks = obj
                break

            if isinstance(obj, torch.nn.Sequential):
                encoder_blocks = obj
                break

    if num_encoder_blocks > 0:
        if encoder_blocks is None:
            print("[Warn] No blocks, encoder_blocks, or encoder found to unfreeze")
        else:
            for blk in list(encoder_blocks)[-num_encoder_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True

    # --------------------------------------------------------
    # 4. Print the trainable parameters
    # --------------------------------------------------------
    trainable_params = [
        name for name, p in model.named_parameters()
        if p.requires_grad
    ]

    print("\nTrainable params:")
    for name in trainable_params:
        print("  ", name)

    n_trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    n_total = sum(p.numel() for p in model.parameters())

    print(f"\nTrainable parameter count: {n_trainable:,} / {n_total:,}")

    if n_trainable == 0:
        raise RuntimeError(
            "No trainable parameters."
            " Check the module names in FT_LAYER_CONFIGS."
        )


# ============================================================
# AEViT train / validation / evaluation
# ============================================================

def train_aevit_one_epoch(model, loader, optimizer, device):
    """Fine-tune for one epoch using the dataset's normalized features."""
    model.train()

    total_loss = 0.0
    total_n = 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        recon, _ = model(x)

        if recon.shape != x.shape:
            raise RuntimeError(
                f"AEViT recon shape mismatch: recon={recon.shape}, x={x.shape}"
            )

        loss = F.mse_loss(recon, x)

        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return {
        "loss": total_loss / max(total_n, 1)
    }
def train_aevit_one_epoch_trimmed(
    model,
    loader,
    optimizer,
    device,
    keep_ratio: float = 0.9,
):
    """Update the model using only samples with the lowest reconstruction errors."""

    if not (0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    model.train()

    total_loss = 0.0
    total_n = 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        recon, _ = model(x)

        if recon.shape != x.shape:
            raise RuntimeError(
                f"AEViT recon shape mismatch: recon={recon.shape}, x={x.shape}"
            )

        # Compute one reconstruction error per sample
        err = (recon - x).pow(2).flatten(1).mean(dim=1)

        # Retain the keep_ratio proportion of samples with the lowest errors
        batch_size = err.numel()
        k = max(1, int(round(batch_size * keep_ratio)))

        if k >= batch_size:
            kept_err = err
        else:
            kept_err = torch.topk(
                err,
                k=k,
                largest=False,
                sorted=False
            ).values

        loss = kept_err.mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * k
        total_n += k

    return {
        "loss": total_loss / max(total_n, 1),
        "keep_ratio": float(keep_ratio),
    }

def validate_aevit(model, loader, device):
    """Evaluate the model on the validation set."""
    model.eval()

    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True).float()

            recon, _ = model(x)

            if recon.shape != x.shape:
                raise RuntimeError(
                    f"AEViT recon shape mismatch: recon={recon.shape}, x={x.shape}"
                )

            loss = F.mse_loss(recon, x)

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs

    return {
        "val_loss": total_loss / max(total_n, 1)
    }


def compute_aevit_reconstruction_scores(model, loader, device):
    """Score samples by reconstruction error; higher scores indicate anomalies."""
    model.eval()

    scores = []

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True).float()

            recon, _ = model(x)

            if recon.shape != x.shape:
                raise RuntimeError(
                    f"AEViT recon shape mismatch: recon={recon.shape}, x={x.shape}"
                )

            s = (recon - x).pow(2).flatten(1).mean(dim=1)

            scores.append(s.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)


# ============================================================
# Main fine-tune function for one attack-ratio task
# ============================================================

def run_one_finetune_task(
    source_name: str,
    target_name: str,
    pretrain_run_dir: Path,
    target_data_dir: Path,
    output_root: Path,
    attack_ratio_pct: float,
    all_loaded_data: Dict[str, Dict[str, object]],
    epochs: int = 40,
    patience: int = 10,
    lr: float = 2e-4,
    weight_decay: float = 0.0,
    fixed_train_size: Optional[int] = 70000,
    max_test_size_per_class: Optional[int] = 10000,
    val_ratio: float = 0.2,
    top_k_trials: int = 1,
    fixed_trial_number: Optional[int] = None,
    upload_to_azure: bool = False,
):
    print("\n" + "=" * 100)
    print(
        f"AEViT fine-tune task: source={source_name}, target={target_name}, "
        f"attack_ratio={attack_ratio_pct}%"
    )
    print("=" * 100)

    task_start = time.perf_counter()

    if fixed_train_size is None:
        raise ValueError("fixed_train_size must specify the benign fine-tuning sample count.")

    if not pretrain_run_dir.exists():
        raise FileNotFoundError(f"Pretrain folder not found: {pretrain_run_dir}")

    if not target_data_dir.exists():
        raise FileNotFoundError(f"Target data folder not found: {target_data_dir}")

    # If not specified manually in TASKS, select the value automatically based on source_name:
    # CICIDS2018 -> trial 6
    # NB15       -> trial 2
    if fixed_trial_number is None:
        fixed_trial_number = PRETRAIN_FIXED_TRIALS.get(source_name, None)

    print("\nTrial selection setting:")
    print("source_name:", source_name)
    print("fixed_trial_number:", fixed_trial_number)

    pretrain_info = find_top_aevit_pretrain_trials(
        pretrain_run_dir=pretrain_run_dir,
        top_k=top_k_trials,
        fixed_trial_number=fixed_trial_number,
    )

    top_trials = pretrain_info["top_trials"]
    model_dir = pretrain_info["model_dir"]
    result_csv = pretrain_info["result_csv"]

    print("Pretrain result csv:", result_csv)
    print("Pretrain model dir:", model_dir)
    print("Selected trials:")
    print(top_trials)

    ratio_tag = str(attack_ratio_pct).replace(".", "p")
    task_name = f"AEViT_src_{source_name}_to_tgt_{target_name}_attack_{ratio_tag}pct"

    task_output_dir = output_root / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    final_csv = task_output_dir / "fine_tune_results_all_layers.csv"

    rows = []

    # --------------------------------------------------------
    # 1. Read target data by AEViT style
    #
    # fixed_train_size now represents the actual number of fine-tuning benign samples and excludes validation samples.
    # Therefore, the preparation stage must first load fixed_train_size + val_size benign samples.
    # --------------------------------------------------------

    val_benign_size = calc_val_size_from_train_size(
        train_size=int(fixed_train_size),
        val_ratio=val_ratio
    )

    target_total_benign_needed = int(fixed_train_size) + int(val_benign_size)

    print("\nRequested actual FT benign:", fixed_train_size)
    print("Validation benign size:", val_benign_size)
    print("Total benign needed before split:", target_total_benign_needed)

    aevit_data = prepare_aevit_target_data(
        target_name=target_name,
        target_data_dir=target_data_dir,
        input_mode=INPUT_MODE,
        packet_shape=PACKET_SHAPE,
        benign_name=BENIGN_NAME,
        split_ratio=0.7,
        train_use_ratio=1.0,
        test_use_ratio=1.0,
        train_target_size=target_total_benign_needed,
        max_test_size_per_class=max_test_size_per_class,
        seed=SPLIT_SEED,
    )

    target_train_subset = aevit_data["train_subset"]
    target_test_subset = aevit_data["test_subset"]
    base_train_ds = aevit_data["base_train_ds"]
    base_test_ds = aevit_data["base_test_ds"]
    y_test_tgt = aevit_data["y_test_tgt"]
    detected_benign_name = aevit_data["benign_name"]

    # --------------------------------------------------------
    # 2. Fix the target split and create it only once
    #    Different ft_layer settings share the same target train/validation split
    #
    # This ensures that benign_train_subset_for_ft == fixed_train_size.
    # --------------------------------------------------------

    benign_train_subset_for_ft, target_val_subset = split_train_subset_into_train_val_indices(
        train_subset=target_train_subset,
        train_benign_size=int(fixed_train_size),
        val_ratio=val_ratio,
        seed=SPLIT_SEED
    )

    # --------------------------------------------------------
    # 3. Shared attack pool
    #
    # If target attacks are insufficient, supplement them from the other dataset.
    # --------------------------------------------------------

    target_binary_dataset = target_train_subset.dataset

    # ============================================================
    # Attack source priority:
    # 1. source/pretrain dataset attack
    # 2. target/finetune dataset attack
    # 3. other common pool attack
    # ============================================================

    attack_sources_ordered = []

    # ------------------------------------------------------------
    # 1. Prioritize attacks from the pretraining/source dataset
    # ------------------------------------------------------------
    if source_name == target_name:
        # When source == target, target_binary_dataset must be used,
        # because fine-tuning benign, validation, and test indices must be excluded.
        attack_sources_ordered.append({
            "dataset_name": f"primary_source_{source_name}",
            "binary_ds": target_binary_dataset,
            "is_target_dataset": True,
        })
    else:
        attack_sources_ordered.append({
            "dataset_name": f"primary_source_{source_name}",
            "binary_ds": all_loaded_data[source_name]["binary_ds"],
            "is_target_dataset": False,
        })

    # ------------------------------------------------------------
    # 2. If source attacks are insufficient, use attacks from the fine-tuning/target dataset
    # ------------------------------------------------------------
    if target_name != source_name:
        attack_sources_ordered.append({
            "dataset_name": f"fallback_target_{target_name}",
            "binary_ds": target_binary_dataset,
            "is_target_dataset": True,
        })

    # ------------------------------------------------------------
    # 3. If they are still insufficient, use another common pool
    # ------------------------------------------------------------
    for dataset_name, loaded in all_loaded_data.items():
        if dataset_name in [source_name, target_name]:
            continue

        attack_sources_ordered.append({
            "dataset_name": f"fallback_common_{dataset_name}",
            "binary_ds": loaded["binary_ds"],
            "is_target_dataset": False,
        })

    ft_train_subset, ft_attack_indices_by_source, ft_data_info = build_mixed_finetune_dataset_with_priority_attack_pool(
        benign_train_subset=benign_train_subset_for_ft,
        target_test_subset=target_test_subset,
        val_subset=target_val_subset,
        attack_ratio_pct=attack_ratio_pct,
        attack_sources_ordered=attack_sources_ordered,
        seed=SPLIT_SEED,
        allow_attack_oversample=False,
    )

    print("Target data prepared by AEViT style.")
    print("Base train ds:", type(base_train_ds))
    print("Base test ds:", type(base_test_ds))
    print("Original benign train subset:", len(target_train_subset))
    print("Benign train for FT:", len(benign_train_subset_for_ft))
    print("Val subset:", len(target_val_subset))
    print("Mixed FT train:", len(ft_train_subset))
    print("Test:", len(target_test_subset))
    print("FT data info:", ft_data_info)
    print("FT attack indices by source:")
    for k, v in ft_attack_indices_by_source.items():
        print(f"  {k}: {len(v)}")

    check_subset_label_distribution_fast(
        subset=benign_train_subset_for_ft,
        name="benign_train_subset_for_ft"
    )

    print("\nmixed_ft_train_dataset distribution:")
    print(f"  +1 / benign: {ft_data_info['benign_train_count']}")
    print(f"  -1 / attack: {ft_data_info['attack_train_count']}")
    print(f"  source_attack_counts: {ft_data_info['source_attack_counts']}")

    # --------------------------------------------------------
    # 4. Fine-tune by ft_layer / seed / trial
    # --------------------------------------------------------

    for ft_layer, ft_cfg in FT_LAYER_CONFIGS.items():

        print("\n" + "=" * 100)
        print(f"Start fine-tuning ft_layer={ft_layer}")
        print("ft_cfg:", ft_cfg)
        print("=" * 100)

        layer_tag = f"ft_{ft_layer}layer"
        layer_output_dir = task_output_dir / layer_tag
        layer_output_dir.mkdir(parents=True, exist_ok=True)

        layer_rows = []

        for run_seed in SEEDS:
            set_global_seed(run_seed)

            worker_init_fn = make_worker(run_seed)
            g = torch.Generator()
            g.manual_seed(run_seed)

            seed_start = time.perf_counter()

            for _, trial_row in top_trials.iterrows():

                trial_number = int(trial_row["number"])
                bs = int(trial_row["params_batch_size"])

                ckpt_path = find_aevit_checkpoint_by_trial(
                    model_dir=model_dir,
                    trial_number=trial_number
                )

                print("\n" + "-" * 80)
                print(f"ft_layer={ft_layer}, seed={run_seed}, trial={trial_number}")
                print("Checkpoint:", ckpt_path)
                print("Batch size:", bs)
                print("-" * 80)

                target_train_loader = DataLoader(
                    ft_train_subset,
                    batch_size=bs,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                    worker_init_fn=worker_init_fn,
                    generator=g,
                    persistent_workers=False
                )

                target_val_loader = DataLoader(
                    target_val_subset,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    worker_init_fn=worker_init_fn,
                    generator=g,
                    persistent_workers=False
                )

                test_loader = DataLoader(
                    target_test_subset,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                    worker_init_fn=worker_init_fn,
                    generator=g,
                    persistent_workers=False
                )

                # --------------------------------------------------------
                # Load checkpoint and build config from trial row
                # --------------------------------------------------------

                ckpt = torch.load(
                    ckpt_path,
                    map_location="cuda" if torch.cuda.is_available() else "cpu"
                )

                cfg = build_aevit_config_from_trial_row(trial_row)

                print("AEViT model config from trial row:")
                print(cfg)

                model = build_model(cfg)
                base_model = unwrap_model(model)

                state_dict = get_state_dict_from_checkpoint(ckpt)
                state_dict = normalize_state_dict_keys(state_dict)

                missing, unexpected = base_model.load_state_dict(
                    state_dict,
                    strict=False
                )

                print("Missing keys:", missing)
                print("Unexpected keys:", unexpected)

                if len(missing) > 20 or len(unexpected) > 20:
                    raise RuntimeError(
                        "Checkpoint may not match the current AEViT architecture: "
                        f"missing={len(missing)}, unexpected={len(unexpected)}"
                    )

                model = model.to(DEVICE)
                base_model = unwrap_model(model)

                # --------------------------------------------------------
                # Fine-tune layer config
                # --------------------------------------------------------

                unfreeze_aevit_by_ft_config(
                    model=base_model,
                    ft_cfg=ft_cfg
                )

                # --------------------------------------------------------
                # Forward test before training
                # --------------------------------------------------------

                with torch.no_grad():
                    x_test_forward, _ = next(iter(target_train_loader))
                    x_test_forward = x_test_forward.to(DEVICE, non_blocking=True).float()
                    recon_test_forward, _ = model(x_test_forward)

                print("Forward test:")
                print("  x shape:", x_test_forward.shape)
                print("  recon shape:", recon_test_forward.shape)

                if recon_test_forward.shape != x_test_forward.shape:
                    raise RuntimeError(
                        f"AEViT forward shape mismatch: "
                        f"recon={recon_test_forward.shape}, x={x_test_forward.shape}"
                    )

                del x_test_forward
                del recon_test_forward

                opt = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr,
                    weight_decay=weight_decay
                )

                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt,
                    T_max=epochs
                )

                best = float("inf")
                best_state = None
                bad = 0

                # --------------------------------------------------------
                # Fine-tune loop
                # --------------------------------------------------------

                for epoch in range(epochs):

                    if TRAIN_LOSS_MODE == "naive":
                        meters = train_aevit_one_epoch(
                            model=model,
                            loader=target_train_loader,
                            optimizer=opt,
                            device=DEVICE
                        )

                    elif TRAIN_LOSS_MODE == "trimmed":
                        meters = train_aevit_one_epoch_trimmed(
                            model=model,
                            loader=target_train_loader,
                            optimizer=opt,
                            device=DEVICE,
                            keep_ratio=TRIM_KEEP_RATIO
                        )

                    else:
                        raise ValueError(f"Unknown TRAIN_LOSS_MODE: {TRAIN_LOSS_MODE}")

                    val = validate_aevit(
                        model=model,
                        loader=target_val_loader,
                        device=DEVICE
                    )

                    train_loss = meters["loss"]
                    val_loss = val["val_loss"]

                    print(
                        f"[{task_name} | {layer_tag} | seed {run_seed} | "
                        f"trial {trial_number} | epoch {epoch:03d}] "
                        f"mode={TRAIN_LOSS_MODE} "
                        f"keep={meters.get('keep_ratio', 1.0):.2f} "
                        f"lr={opt.param_groups[0]['lr']:.2e} "
                        f"loss={train_loss:.6f} "
                        f"val={val_loss:.6f}"
                    )
                    if val_loss < best - 1e-6:
                        best = val_loss
                        bad = 0

                        best_state = {
                            name: value.detach().cpu().clone()
                            for name, value in base_model.state_dict().items()
                        }

                    else:
                        bad += 1

                        if bad >= patience:
                            print(f"Early stopping at epoch {epoch}")
                            break

                    scheduler.step()

                if best_state is not None:
                    base_model.load_state_dict(best_state)

                # --------------------------------------------------------
                # Save model
                # --------------------------------------------------------

                ft_model_path = (
                    layer_output_dir
                    / f"AEViT_ft_{layer_tag}_src_{source_name}"
                    f"_to_{target_name}_attack_{ratio_tag}pct"
                    f"_trial_{trial_number}_seed_{run_seed}.pt"
                )

                validation_scores = compute_aevit_reconstruction_scores(
                    model=model,
                    loader=target_val_loader,
                    device=DEVICE,
                )
                threshold = select_validation_threshold(validation_scores)
                ft_ckpt = {
                    "validation_threshold": threshold,
                    "state_dict": {
                        name: value.detach().cpu().clone()
                        for name, value in base_model.state_dict().items()
                    },
                    "cfg": cfg.__dict__,
                    "ft_layer": ft_layer,
                    "ft_cfg": ft_cfg,
                    "source": source_name,
                    "target": target_name,
                    "seed": run_seed,
                    "trial_number": trial_number,
                    "fixed_trial_number": fixed_trial_number,
                    "train_loss_mode": TRAIN_LOSS_MODE,
                    "trim_keep_ratio": TRIM_KEEP_RATIO,
                    "pretrain_ckpt": str(ckpt_path),
                    "target_data_dir": str(target_data_dir),
                    "best_val_loss": best,
                    "attack_ratio_pct": attack_ratio_pct,
                    "benign_train_count": ft_data_info["benign_train_count"],
                    "attack_train_count": ft_data_info["attack_train_count"],
                    "total_ft_train_count": ft_data_info["total_ft_train_count"],
                    "target_attack_pool_count": ft_data_info["target_attack_pool_count"],
                    "source_attack_counts": ft_data_info["source_attack_counts"],
                    "common_attack_used": ft_data_info["common_attack_used"],
                }

                torch.save(ft_ckpt, ft_model_path)

                print("Saved fine-tuned AEViT model:", ft_model_path)

                # --------------------------------------------------------
                # Evaluate
                # --------------------------------------------------------

                test_score = compute_aevit_reconstruction_scores(
                    model=model,
                    loader=test_loader,
                    device=DEVICE
                )

                if len(test_score) != len(y_test_tgt):
                    raise RuntimeError(
                        f"test_score length != y_test_tgt length: "
                        f"{len(test_score)} != {len(y_test_tgt)}"
                    )

                results = evaluate_test_data(
                    y_test_tgt,
                    test_score,
                    threshold
                )

                recall = results["recall"]
                precision = results["precision"]
                f1 = results["f1"]
                au_precision_recall = results["au_precision_recall"]
                auroc = results["auroc"]

                try:
                    plot_confusion_matrix(
                        y_test_tgt,
                        test_score,
                        threshold,
                        pos_label=-1
                    )

                    fig_path = (
                        layer_output_dir
                        / f"confusion_matrix_{layer_tag}_attack_{ratio_tag}pct"
                        f"_trial_{trial_number}_seed_{run_seed}.png"
                    )

                    import matplotlib.pyplot as plt
                    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
                    plt.close()

                except Exception as e:
                    print("[Warn] plot_confusion_matrix failed:", e)

                seed_sec = time.perf_counter() - seed_start

                metrics_dict = {
                    "source": source_name,
                    "target": target_name,
                    "ft_layer": ft_layer,
                    "ft_cfg": str(ft_cfg),
                    "seed": run_seed,
                    "attack_ratio_pct": attack_ratio_pct,
                    "benign_train_count": ft_data_info["benign_train_count"],
                    "attack_train_count": ft_data_info["attack_train_count"],
                    "total_ft_train_count": ft_data_info["total_ft_train_count"],
                    "target_attack_pool_count": ft_data_info["target_attack_pool_count"],
                    "source_attack_counts": str(ft_data_info["source_attack_counts"]),
                    "common_attack_used": ft_data_info["common_attack_used"],
                    "trial_number": trial_number,
                    "fixed_trial_number": fixed_trial_number,
                    "train_loss_mode": TRAIN_LOSS_MODE,
                    "trim_keep_ratio": TRIM_KEEP_RATIO,
                    "pretrain_ckpt": str(ckpt_path),
                    "target_data_dir": str(target_data_dir),
                    "ft_model_path": str(ft_model_path),
                    "best_val_loss": best,
                    "threshold": threshold,

                    "TP": results.get("TP", None),
                    "FP": results.get("FP", None),
                    "TN": results.get("TN", None),
                    "FN": results.get("FN", None),

                    "TPR": results.get("TPR", None),
                    "FPR": results.get("FPR", None),

                    "precision": precision,
                    "recall": recall,
                    "f1": f1,

                    "au_precision_recall": au_precision_recall,
                    "auroc": auroc,
                    "seed_ft_total_sec": seed_sec,
                    "seed_ft_total_hms": fmt_hms(seed_sec),
                }

                rows.append(metrics_dict)
                layer_rows.append(metrics_dict)

                print(f"\nSEED: {run_seed}")
                print(f"Layer:              {ft_layer}")
                print(f"Trial:              {trial_number}")
                print(f"Fixed trial:        {fixed_trial_number}")
                print(f"Attack ratio:       {attack_ratio_pct}%")
                print(f"Benign train:       {ft_data_info['benign_train_count']}")
                print(f"Attack train:       {ft_data_info['attack_train_count']}")
                print(f"Total FT train:     {ft_data_info['total_ft_train_count']}")
                print(f"Attack sources:     {ft_data_info['source_attack_counts']}")
                print(f"Threshold:          {threshold:.6f}")
                print(f"Recall:             {recall:.4f}")
                print(f"Precision:          {precision:.4f}")
                print(f"F1-score:           {f1:.4f}")
                print(f"AUC PR:             {au_precision_recall:.4f}")
                print(f"AUROC:              {auroc:.4f}")
                print(f"Time:               {fmt_hms(seed_sec)}")

                # --------------------------------------------------------
                # Clean seed/trial-level objects
                # --------------------------------------------------------

                del model
                del base_model
                del opt
                del scheduler
                del target_train_loader
                del target_val_loader
                del test_loader
                del ckpt
                del state_dict

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

        # Save a separate CSV for each ft_layer
        layer_df = pd.DataFrame(layer_rows)
        layer_csv = layer_output_dir / "fine_tune_results.csv"
        layer_df.to_csv(layer_csv, index=False)

        print("Saved layer result csv:", layer_csv)

    # --------------------------------------------------------
    # Save task CSV
    # --------------------------------------------------------

    df = pd.DataFrame(rows)
    df.to_csv(final_csv, index=False)

    print("Saved task result csv:", final_csv)

    task_sec = time.perf_counter() - task_start

    summary_path = task_output_dir / "task_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"task_name: {task_name}\n")
        f.write(f"source: {source_name}\n")
        f.write(f"target: {target_name}\n")
        f.write(f"attack_ratio_pct: {attack_ratio_pct}\n")
        f.write(f"fixed_trial_number: {fixed_trial_number}\n")
        f.write(f"train_loss_mode: {TRAIN_LOSS_MODE}\n")
        f.write(f"trim_keep_ratio: {TRIM_KEEP_RATIO}\n")
        f.write(f"benign_train_count: {ft_data_info['benign_train_count']}\n")
        f.write(f"attack_train_count: {ft_data_info['attack_train_count']}\n")
        f.write(f"total_ft_train_count: {ft_data_info['total_ft_train_count']}\n")
        f.write(f"target_attack_pool_count: {ft_data_info['target_attack_pool_count']}\n")
        f.write(f"source_attack_counts: {ft_data_info['source_attack_counts']}\n")
        f.write(f"common_attack_used: {ft_data_info['common_attack_used']}\n")
        f.write(f"pretrain_run_dir: {pretrain_run_dir}\n")
        f.write(f"target_data_dir: {target_data_dir}\n")
        f.write(f"result_csv: {result_csv}\n")
        f.write(f"model_dir: {model_dir}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"patience: {patience}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"fixed_train_size_actual_ft_benign: {fixed_train_size}\n")
        f.write(f"val_benign_size: {val_benign_size}\n")
        f.write(f"target_total_benign_needed: {target_total_benign_needed}\n")
        f.write(f"max_test_size_per_class: {max_test_size_per_class}\n")
        f.write(f"val_ratio: {val_ratio}\n")
        f.write(f"top_k_trials: {top_k_trials}\n")
        f.write(f"seeds: {SEEDS}\n")
        f.write(f"ft_layer_configs: {FT_LAYER_CONFIGS}\n")
        f.write(f"pretrain_fixed_trials: {PRETRAIN_FIXED_TRIALS}\n")
        f.write(f"total_time_sec: {task_sec}\n")
        f.write(f"total_time_hms: {fmt_hms(task_sec)}\n")

    # --------------------------------------------------------
    # Upload to Azure
    # --------------------------------------------------------

    if upload_to_azure:
        from util.azure_upload import upload_directory_to_azure

        azure_prefix = f"{AZURE_BASE_PREFIX}/{task_name}"

        upload_result = upload_directory_to_azure(
            sas_url=os.environ["TRANIDS_AZURE_SAS_URL"],
            local_dir=task_output_dir,
            azure_prefix=azure_prefix,
            max_workers=4,
            overwrite=True,
            pattern="*",
            recursive=True,
            verbose=True
        )

        print("Azure upload result:", upload_result)

    # --------------------------------------------------------
    # Clean RAM / file cache
    # --------------------------------------------------------

    release_memory(
        objects_to_delete=[
            aevit_data,
            base_train_ds,
            base_test_ds,
            target_train_subset,
            target_test_subset,
            benign_train_subset_for_ft,
            target_val_subset,
            ft_train_subset,
            y_test_tgt,
        ],
        cache_roots=[target_data_dir]
    )

    return {
        "task_name": task_name,
        "source": source_name,
        "target": target_name,
        "attack_ratio_pct": attack_ratio_pct,
        "fixed_trial_number": fixed_trial_number,
        "benign_train_count": ft_data_info["benign_train_count"],
        "attack_train_count": ft_data_info["attack_train_count"],
        "total_ft_train_count": ft_data_info["total_ft_train_count"],
        "target_attack_pool_count": ft_data_info["target_attack_pool_count"],
        "source_attack_counts": str(ft_data_info["source_attack_counts"]),
        "common_attack_used": ft_data_info["common_attack_used"],
        "output_dir": str(task_output_dir),
        "result_csv": str(final_csv),
        "time_sec": task_sec,
        "time_hms": fmt_hms(task_sec),
    }


# ============================================================
# Build tasks
# ============================================================

TASKS = []


# ------------------------------------------------------------
# Example 1:
# CICIDS2018 pretrained AEViT -> NB15 fine-tune
# Always use CICIDS2018 pretraining trial 6
# ------------------------------------------------------------

# TASKS.append({
#     "source_name": "CICIDS2018",
#     "target_name": "NB15",
#     "pretrain_run_dir": PRETRAIN_RUNS["CICIDS2018"],
#     "target_data_dir": TARGET_DATA["NB15"],
#     "attack_ratio_pct": 30,
#     "fixed_trial_number": PRETRAIN_FIXED_TRIALS["CICIDS2018"],
# })


# ------------------------------------------------------------
# Example 2:
# NB15 pretrained AEViT -> CICIDS2018 fine-tune
# Always use NB15 pretraining trial 2
# Uncomment this to run it.
# ------------------------------------------------------------

# TASKS.append({
#     "source_name": "NB15",
#     "target_name": "CICIDS2018",
#     "pretrain_run_dir": PRETRAIN_RUNS["NB15"],
#     "target_data_dir": TARGET_DATA["CICIDS2018"],
#     "attack_ratio_pct": 30,
#     "fixed_trial_number": PRETRAIN_FIXED_TRIALS["NB15"],
# })


# ------------------------------------------------------------
# Use this to run all attack ratios:
# For the CICIDS2018 source, always use trial 6
# For the NB15 source, always use trial 2
# ------------------------------------------------------------

TASKS = []

# for ratio in ATTACK_RATIOS:
#     TASKS.append({
#         "source_name": "CICIDS2018",
#         "target_name": "NB15",
#         "pretrain_run_dir": PRETRAIN_RUNS["CICIDS2018"],
#         "target_data_dir": TARGET_DATA["NB15"],
#         "attack_ratio_pct": ratio,
#         "fixed_trial_number": PRETRAIN_FIXED_TRIALS["CICIDS2018"],
#     })

# for ratio in ATTACK_RATIOS:
#     TASKS.append({
#         "source_name": "NB15",
#         "target_name": "CICIDS2018",
#         "pretrain_run_dir": PRETRAIN_RUNS["NB15"],
#         "target_data_dir": TARGET_DATA["CICIDS2018"],
#         "attack_ratio_pct": ratio,
#         "fixed_trial_number": PRETRAIN_FIXED_TRIALS["NB15"],
#     })
for ratio in ATTACK_RATIOS:
    TASKS.append({
        "source_name": "CICIDS2018",
        "target_name": "NB15",
        "pretrain_run_dir": PRETRAIN_RUNS["CICIDS2018"],
        "target_data_dir": TARGET_DATA["NB15"],
        "attack_ratio_pct": ratio,
        "fixed_trial_number": PRETRAIN_FIXED_TRIALS["CICIDS2018"],
    })




print(f"Total fine-tune tasks: {len(TASKS)}")

for t in TASKS:
    print(t)


# ============================================================
# Load common attack pool datasets at the beginning
# ============================================================

print("\n" + "=" * 100)
print("Loading common attack pool datasets: NB15 + CICIDS2018")
print("=" * 100)

all_loaded_data = {}

for dataset_name, data_dir in TARGET_DATA.items():
    print("\n" + "-" * 80)
    print(f"Loading base dataset for common attack pool: {dataset_name}")
    print("data_dir:", data_dir)
    print("-" * 80)

    if not data_dir.exists():
        raise FileNotFoundError(f"Target data folder not found for {dataset_name}: {data_dir}")

    loaded = load_aevit_base_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        input_mode=INPUT_MODE,
        packet_shape=PACKET_SHAPE,
        benign_name=BENIGN_NAME,
    )

    all_loaded_data[dataset_name] = loaded

    print("Loaded:", dataset_name)
    print("  base_ds:", type(loaded["base_ds"]))
    print("  binary_ds:", type(loaded["binary_ds"]))
    print("  benign_name:", loaded["benign_name"])
    print("  benign_idx_val:", loaded["benign_idx_val"])
    print("  total samples:", len(loaded["binary_ds"]))


# ============================================================
# Run all tasks
# ============================================================

all_task_rows = []

global_start = time.perf_counter()

for task_id, task in enumerate(TASKS, 1):
    print("\n" + "#" * 120)
    print(f"Running task {task_id}/{len(TASKS)}")
    print("#" * 120)

    try:
        result = run_one_finetune_task(
            source_name=task["source_name"],
            target_name=task["target_name"],
            pretrain_run_dir=task["pretrain_run_dir"],
            target_data_dir=task["target_data_dir"],
            output_root=FT_OUTPUT_ROOT,
            attack_ratio_pct=task["attack_ratio_pct"],
            all_loaded_data=all_loaded_data,

            # For initial testing, reduce this value:
            # epochs=2
            epochs=20,
            patience=2,
            lr = 5e-6,
            weight_decay=0.0,

            # For initial testing, reduce this value:
            # fixed_train_size=1000
            # max_test_size_per_class=100
            #
            # Here, fixed_train_size is the actual number of fine-tuning benign samples.
            # It excludes validation samples and additionally injected attacks.
            fixed_train_size=1000000,
            max_test_size_per_class=10000,

            val_ratio=0.2,

            # top_k_trials affects trial selection only when fixed_trial_number=None.
            # Because fixed_trial_number is specified, this value generally does not affect trial selection.
            top_k_trials=1,

            # Fixed trials:
            # CICIDS2018 -> trial 6
            # NB15       -> trial 2
            fixed_trial_number=task.get(
                "fixed_trial_number",
                PRETRAIN_FIXED_TRIALS.get(task["source_name"], None)
            ),

            # Keep this False initially; change it to True after verifying the results
            upload_to_azure=False,
        )

        result["status"] = "success"
        result["error"] = ""

    except Exception as e:
        print(f"[ERROR] Task failed: {task}")
        print(e)

        ratio_tag = str(task["attack_ratio_pct"]).replace(".", "p")

        result = {
            "task_name": (
                f"AEViT_src_{task['source_name']}"
                f"_to_tgt_{task['target_name']}"
                f"_attack_{ratio_tag}pct"
            ),
            "source": task["source_name"],
            "target": task["target_name"],
            "attack_ratio_pct": task["attack_ratio_pct"],
            "fixed_trial_number": task.get(
                "fixed_trial_number",
                PRETRAIN_FIXED_TRIALS.get(task["source_name"], None)
            ),
            "output_dir": "",
            "result_csv": "",
            "time_sec": None,
            "time_hms": "",
            "status": "failed",
            "error": str(e),
        }

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    all_task_rows.append(result)

    summary_df = pd.DataFrame(all_task_rows)
    summary_csv = FT_OUTPUT_ROOT / "all_attack_ratio_finetune_summary_AEViT.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("Updated global summary:", summary_csv)


global_sec = time.perf_counter() - global_start

print("\n" + "=" * 100)
print("All AEViT attack-ratio fine-tune tasks finished.")
print("Total time:", fmt_hms(global_sec))
print("=" * 100)

summary_df = pd.DataFrame(all_task_rows)
summary_df["global_total_sec"] = global_sec
summary_df["global_total_hms"] = fmt_hms(global_sec)

summary_df.to_csv(
    FT_OUTPUT_ROOT / "all_attack_ratio_finetune_summary_AEViT.csv",
    index=False
)
