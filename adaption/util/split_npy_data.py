import os
import sys
import numpy as np
import torch

from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

try:
    import psutil
except ImportError:
    psutil = None


# ============================================================
# Config
# ============================================================

@dataclass
class NPYOneClassConfig:
    root: str
    benign_name: str = "Benign"
    split_seed: int = 42
    per_class_frac: float = 0.28
    test_size: float = 0.3
    valtest_test_size: float = 0.5
    ram_ratio: float = 0.6
    max_full_load_available_ratio: float = 0.8
    balanced_test: bool = True

    # Optional sample counts.
    fixed_train_size: Optional[int] = None
    fixed_test_size: Optional[int] = None

    # Optional class counts for the balanced test set.
    fixed_test_benign_size: Optional[int] = None
    fixed_test_malicious_size: Optional[int] = None

    # Allow sampling with replacement.
    allow_oversample: bool = False

# ============================================================
# Optional: project root / evaluate_results
# ============================================================

def setup_project_root(project_root: str = "/workspace") -> None:
    """Add the project root to the Python import path."""
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def import_evaluate_results(project_root: str = "/workspace"):
    """Import the evaluation helper on demand."""
    setup_project_root(project_root)

    try:
        from .data_preparing import evaluate_results
        print("Imported evaluate_results")
        return evaluate_results

    except Exception as e:
        print("Import failed")
        print("cwd:", os.getcwd())
        print("sys.path:", sys.path)
        raise e


# ============================================================
# Device
# ============================================================

def get_device(verbose: bool = True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        if torch.cuda.is_available():
            print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            print("No GPU found, using CPU")

    return device


# ============================================================
# Memory utility
# ============================================================

def get_cgroup_memory_limit():
    path = "/sys/fs/cgroup/memory.max"

    try:
        with open(path, "r") as f:
            value = f.read().strip()

        if value == "max":
            return None

        return int(value)

    except Exception:
        return None


def get_cgroup_memory_current():
    path = "/sys/fs/cgroup/memory.current"

    try:
        with open(path, "r") as f:
            return int(f.read().strip())

    except Exception:
        return None


def smart_load_x_npy(
    x_path,
    ram_ratio=0.6,
    max_full_load_available_ratio=0.8
):
    """Load features into memory when space permits, or memory-map them."""
    x_path = Path(x_path)
    file_size = os.path.getsize(x_path)
    file_size_gb = file_size / 1024**3

    cgroup_limit = get_cgroup_memory_limit()
    cgroup_current = get_cgroup_memory_current()

    if cgroup_limit is not None and cgroup_current is not None:
        usable_ram = cgroup_limit
        available_ram = cgroup_limit - cgroup_current
        source = "cgroup memory.max/current"

    elif psutil is not None:
        usable_ram = psutil.virtual_memory().total
        available_ram = psutil.virtual_memory().available
        source = "psutil"

    else:
        print("[Warn] Cannot detect RAM. Using MMAP mode.")
        return np.load(x_path, mmap_mode="r")

    usable_ram_gb = usable_ram / 1024**3
    available_ram_gb = available_ram / 1024**3

    threshold_gb = usable_ram_gb * ram_ratio
    max_full_load_gb = available_ram_gb * max_full_load_available_ratio

    print(f"X_uint8.npy file size: {file_size_gb:.2f} GB")
    print(f"RAM source: {source}")
    print(f"Usable RAM limit: {usable_ram_gb:.2f} GB")
    print(f"Current available RAM: {available_ram_gb:.2f} GB")
    print(f"Full-load threshold by ram_ratio: {threshold_gb:.2f} GB")
    print(
        f"Max allowed full-load size "
        f"({max_full_load_available_ratio * 100:.0f}% of available): "
        f"{max_full_load_gb:.2f} GB"
    )

    if file_size_gb <= max_full_load_gb and file_size < usable_ram * ram_ratio:
        print("Using FULL LOAD mode for X_uint8.npy")
        return np.load(x_path)

    else:
        print("Using MMAP mode for X_uint8.npy")
        return np.load(x_path, mmap_mode="r")


# ============================================================
# Dataset
# ============================================================

class NPYDataset(Dataset):
    """Read features and original class indices from NPY files."""

    def __init__(
        self,
        root,
        ram_ratio=0.6,
        max_full_load_available_ratio=0.8
    ):
        self.root = Path(root)

        x_path = self.root / "X_uint8.npy"
        y_path = self.root / "y.npy"
        class_map_path = self.root / "class_to_idx.npy"

        if not x_path.exists():
            raise FileNotFoundError(f"Missing file: {x_path}")
        if not y_path.exists():
            raise FileNotFoundError(f"Missing file: {y_path}")
        if not class_map_path.exists():
            raise FileNotFoundError(f"Missing file: {class_map_path}")

        self.X = smart_load_x_npy(
            x_path,
            ram_ratio=ram_ratio,
            max_full_load_available_ratio=max_full_load_available_ratio
        )

        self.y = np.load(y_path).astype(np.int64)

        class_map = np.load(
            class_map_path,
            allow_pickle=True
        )

        self.class_to_idx = (
            class_map.item()
            if class_map.shape == ()
            else dict(class_map.tolist())
        )

        self.idx_to_class = {
            v: k for k, v in self.class_to_idx.items()
        }

        if len(self.X) != len(self.y):
            raise RuntimeError(
                f"X and y length mismatch: X={len(self.X)}, y={len(self.y)}"
            )

        print("Dataset ready.")
        print("X shape:", self.X.shape, self.X.dtype)
        print("y shape:", self.y.shape, self.y.dtype)
        print("class_to_idx:", self.class_to_idx)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = int(self.y[idx])

        # Keep features in uint8 format.
        x = torch.from_numpy(
            np.array(x, copy=True)
        )

        return x, y


# ============================================================
# Step 1: per-class sampling
# ============================================================

def sample_per_class_same_as_original(
    all_labels: np.ndarray,
    per_class_frac: float,
    rng: np.random.RandomState
) -> np.ndarray:
    """Sample each class using the supplied random generator."""

    sampled_idx = []

    for c in np.unique(all_labels):

        class_indices = np.where(all_labels == c)[0]

        n_total = len(class_indices)

        n_keep = max(
            1,
            int(round(n_total * per_class_frac))
        )

        chosen = rng.choice(
            class_indices,
            size=n_keep,
            replace=False
        )

        sampled_idx.append(chosen)

    sampled_idx = np.concatenate(sampled_idx)
    sampled_idx.sort()

    return sampled_idx


# ============================================================
# Step 2: binary label
# ============================================================

def build_binary_labels_same_as_original(
    labels: np.ndarray,
    idx_to_class: Dict[int, str],
    benign_name: str
) -> np.ndarray:
    """
    Benign = +1
    Malicious = -1
    """
    binary_labels = np.array([
        1 if idx_to_class[int(y)] == benign_name else -1
        for y in labels
    ])

    return binary_labels


# ============================================================
# Step 3: train / val / test split
# ============================================================

def split_train_val_test_same_as_original(
    sampled_idx: np.ndarray,
    y_binary_sampled: np.ndarray,
    split_seed: int,
    test_size: float = 0.3,
    valtest_test_size: float = 0.5
):
    """Split indices into stratified training, validation, and test sets."""

    train_all_idx_sampled, valtest_idx_sampled, _, y_valtest_bin = train_test_split(
        sampled_idx,
        y_binary_sampled,
        test_size=test_size,
        stratify=y_binary_sampled,
        random_state=split_seed
    )

    val_idx, test_idx, y_val_bin, y_test_bin = train_test_split(
        valtest_idx_sampled,
        y_valtest_bin,
        test_size=valtest_test_size,
        stratify=y_valtest_bin,
        random_state=split_seed
    )

    return (
        train_all_idx_sampled,
        val_idx,
        test_idx,
        y_val_bin,
        y_test_bin
    )


# ============================================================
# Step 4: train only benign
# ============================================================

def keep_train_benign_only_same_as_original(
    train_all_idx_sampled: np.ndarray,
    full_dataset: NPYDataset,
    idx_to_class: Dict[int, str],
    benign_name: str
) -> np.ndarray:
    """Keep benign samples in the training set."""

    y_train_all_bin = np.array([
        1 if idx_to_class[int(full_dataset.y[i])] == benign_name else -1
        for i in train_all_idx_sampled
    ])

    train_idx = train_all_idx_sampled[
        y_train_all_bin == 1
    ]

    train_idx.sort()

    return train_idx

# ============================================================
# Utility: fixed-size sampling
# ============================================================

def sample_fixed_size_indices(
    indices: np.ndarray,
    fixed_size: Optional[int],
    rng: np.random.RandomState,
    name: str,
    allow_oversample: bool = False,
    sort_result: bool = True
) -> np.ndarray:
    """Select a fixed number of indices, with optional replacement."""

    indices = np.asarray(indices)

    if fixed_size is None:
        return indices

    if fixed_size <= 0:
        raise ValueError(f"{name}: fixed_size must be positive; got {fixed_size}")

    replace_flag = fixed_size > len(indices)

    if replace_flag and not allow_oversample:
        raise RuntimeError(
            f"{name}: Insufficient samples."
            f"Requested {fixed_size}, available {len(indices)}."
            f"Set allow_oversample=True to sample with replacement."
        )

    if replace_flag:
        print(
            f"[Warn] {name}: Requested {fixed_size}, available {len(indices)}."
            f" Sampling with replacement."
        )

    chosen = rng.choice(
        indices,
        size=fixed_size,
        replace=replace_flag
    )

    if sort_result:
        chosen.sort()

    return chosen
# ============================================================
# Step 5: balanced test
# ============================================================

def build_balanced_test_same_as_original(
    test_idx: np.ndarray,
    y_test_bin: np.ndarray,
    rng: np.random.RandomState,
    fixed_test_size: Optional[int] = None,
    fixed_test_benign_size: Optional[int] = None,
    fixed_test_malicious_size: Optional[int] = None,
    allow_oversample: bool = False
):
    pos_mal = np.where(y_test_bin == -1)[0]
    pos_ben = np.where(y_test_bin == 1)[0]

    if len(pos_mal) == 0:
        raise RuntimeError(
            "No malicious (-1) samples for the balanced test set."
        )

    if len(pos_ben) == 0:
        raise RuntimeError(
            "No benign (+1) samples for the balanced test set."
        )

    has_fixed_test = (
        fixed_test_size is not None
        or fixed_test_benign_size is not None
        or fixed_test_malicious_size is not None
    )

    # ========================================================
    # Use the default test size.
    # ========================================================
    if not has_fixed_test:

        n_mal = len(pos_mal)

        replace_flag = n_mal > len(pos_ben)

        if replace_flag:
            print(
                f"[Warn] Benign ({len(pos_ben)}) is fewer than "
                f"Malicious ({n_mal}); sampling with replacement."
            )

        pos_ben_pick = rng.choice(
            pos_ben,
            size=n_mal,
            replace=replace_flag
        )

        test_pos = np.concatenate([
            pos_mal,
            pos_ben_pick
        ])

        rng.shuffle(test_pos)

        balanced_test_idx = test_idx[test_pos]
        y_test_src = y_test_bin[test_pos]

        print(
            f"Malicious(-1): {n_mal}, "
            f"Sampled benign (+1): {len(pos_ben_pick)}"
        )

        return balanced_test_idx, y_test_src

    # ========================================================
    # Use explicit class counts.
    # ========================================================
    if fixed_test_benign_size is not None or fixed_test_malicious_size is not None:

        if fixed_test_benign_size is None or fixed_test_malicious_size is None:
            raise ValueError(
                "Set both fixed_test_benign_size and fixed_test_malicious_size."
                ""
            )

        n_ben = fixed_test_benign_size
        n_mal = fixed_test_malicious_size

    # ========================================================
    # Split the fixed total evenly across classes.
    # ========================================================
    else:
        if fixed_test_size <= 1:
            raise ValueError("fixed_test_size must exceed 1.")

        if fixed_test_size % 2 != 0:
            raise ValueError("fixed_test_size must be even for a balanced test set.")

        n_mal = fixed_test_size // 2
        n_ben = fixed_test_size // 2

    replace_mal = n_mal > len(pos_mal)
    replace_ben = n_ben > len(pos_ben)

    if replace_mal and not allow_oversample:
        raise RuntimeError(
            f"Requested {n_mal} malicious test samples; found {len(pos_mal)}."
        )

    if replace_ben and not allow_oversample:
        raise RuntimeError(
            f"Requested {n_ben} benign test samples; found {len(pos_ben)}."
        )

    pos_mal_pick = rng.choice(
        pos_mal,
        size=n_mal,
        replace=replace_mal
    )

    pos_ben_pick = rng.choice(
        pos_ben,
        size=n_ben,
        replace=replace_ben
    )

    test_pos = np.concatenate([
        pos_mal_pick,
        pos_ben_pick
    ])

    rng.shuffle(test_pos)

    balanced_test_idx = test_idx[test_pos]
    y_test_src = y_test_bin[test_pos]

    print(
        f"Fixed Balanced Test Set: "
        f"Malicious(-1): {np.sum(y_test_src == -1)}, "
        f"Benign(+1): {np.sum(y_test_src == 1)}, "
        f"Total: {len(y_test_src)}"
    )

    return balanced_test_idx, y_test_src

def build_fixed_unbalanced_test(
    test_idx: np.ndarray,
    y_test_bin: np.ndarray,
    rng: np.random.RandomState,
    fixed_test_size: Optional[int] = None,
    allow_oversample: bool = False
):
    """Sample a fixed total without imposing a class ratio."""

    if fixed_test_size is None:
        return test_idx, y_test_bin

    if fixed_test_size <= 0:
        raise ValueError("fixed_test_size must be positive.")

    all_pos = np.arange(len(test_idx))

    replace_flag = fixed_test_size > len(all_pos)

    if replace_flag and not allow_oversample:
        raise RuntimeError(
            f"Requested {fixed_test_size} test samples; found {len(all_pos)}."
            f"Set allow_oversample=True to sample with replacement."
        )

    if replace_flag:
        print(
            f"[Warn] Test set Requested {fixed_test_size} samples, available {len(all_pos)}."
            f" Sampling with replacement."
        )

    chosen_pos = rng.choice(
        all_pos,
        size=fixed_test_size,
        replace=replace_flag
    )

    rng.shuffle(chosen_pos)

    fixed_test_idx = test_idx[chosen_pos]
    fixed_y_test_src = y_test_bin[chosen_pos]

    print(
        f"Fixed Unbalanced Test Set: "
        f"Malicious(-1): {np.sum(fixed_y_test_src == -1)}, "
        f"Benign(+1): {np.sum(fixed_y_test_src == 1)}, "
        f"Total: {len(fixed_y_test_src)}"
    )

    return fixed_test_idx, fixed_y_test_src


# ============================================================
# Main reusable function
# ============================================================

def prepare_oneclass_npy_data_same_as_original(
    config: NPYOneClassConfig
) -> Dict[str, object]:
    """Prepare one-class NPY splits using a shared random generator."""

    # --------------------------------------------------------
    # rng = np.random.RandomState(SPLIT_SEED)
    # --------------------------------------------------------
    rng = np.random.RandomState(config.split_seed)

    # --------------------------------------------------------
    # Load source dataset from local
    # --------------------------------------------------------
    src_full_ds = NPYDataset(
        root=config.root,
        ram_ratio=config.ram_ratio,
        max_full_load_available_ratio=config.max_full_load_available_ratio
    )

    src_class_to_idx = src_full_ds.class_to_idx
    src_idx_to_class = src_full_ds.idx_to_class

    print("src_class_to_idx:", src_class_to_idx)

    # --------------------------------------------------------
    # Read labels directly from y.npy.
    # --------------------------------------------------------
    src_all_idx = np.arange(len(src_full_ds))
    src_all_labels = np.asarray(src_full_ds.y)

    # --------------------------------------------------------
    # Sample each class.
    # --------------------------------------------------------
    src_sampled_idx = sample_per_class_same_as_original(
        all_labels=src_all_labels,
        per_class_frac=config.per_class_frac,
        rng=rng
    )

    print(f"Original source sample count: {len(src_full_ds)}")
    print(f"Sampled source count: {len(src_sampled_idx)}")

    # --------------------------------------------------------
    # Encode the binary labels.
    # Benign = 1
    # Malicious = -1
    # --------------------------------------------------------
    src_sampled_labels = src_all_labels[src_sampled_idx]

    src_y_binary_sampled = build_binary_labels_same_as_original(
        labels=src_sampled_labels,
        idx_to_class=src_idx_to_class,
        benign_name=config.benign_name
    )

    # --------------------------------------------------------
    # Split: Train / Val / Test
    # --------------------------------------------------------
    (
        src_train_all_idx_sampled,
        src_val_idx,
        src_test_idx,
        src_y_val_bin,
        src_y_test_bin
    ) = split_train_val_test_same_as_original(
        sampled_idx=src_sampled_idx,
        y_binary_sampled=src_y_binary_sampled,
        split_seed=config.split_seed,
        test_size=config.test_size,
        valtest_test_size=config.valtest_test_size
    )

    # --------------------------------------------------------
    # Keep only benign training samples.
    # --------------------------------------------------------
    src_train_idx = keep_train_benign_only_same_as_original(
        train_all_idx_sampled=src_train_all_idx_sampled,
        full_dataset=src_full_ds,
        idx_to_class=src_idx_to_class,
        benign_name=config.benign_name
    )

        # --------------------------------------------------------
    # Apply the requested training size.
    # --------------------------------------------------------
    src_train_idx = sample_fixed_size_indices(
        indices=src_train_idx,
        fixed_size=config.fixed_train_size,
        rng=rng,
        name="SRC Train Benign Only",
        allow_oversample=config.allow_oversample,
        sort_result=True
    )

    # --------------------------------------------------------
    # Test Set
    # --------------------------------------------------------
    if config.balanced_test:
        test_idx, y_test_src = build_balanced_test_same_as_original(
            test_idx=src_test_idx,
            y_test_bin=src_y_test_bin,
            rng=rng,
            fixed_test_size=config.fixed_test_size,
            fixed_test_benign_size=config.fixed_test_benign_size,
            fixed_test_malicious_size=config.fixed_test_malicious_size,
            allow_oversample=config.allow_oversample
        )
    else:
        test_idx, y_test_src = build_fixed_unbalanced_test(
            test_idx=src_test_idx,
            y_test_bin=src_y_test_bin,
            rng=rng,
            fixed_test_size=config.fixed_test_size,
            allow_oversample=config.allow_oversample
        )

    print(
        f"SRC Train(benign only): {len(src_train_idx)}, "
        f"SRC Val(all): {len(src_val_idx)}, "
        f"SRC Test(Balanced): {len(y_test_src)}"
    )

    # --------------------------------------------------------
    # Build subsets for the data loaders.
    # --------------------------------------------------------
    src_train_subset = Subset(src_full_ds, src_train_idx)
    src_val_subset = Subset(src_full_ds, src_val_idx)
    src_test_subset = Subset(src_full_ds, test_idx)

    return {
        # dataset
        "src_full_ds": src_full_ds,
        "src_train_subset": src_train_subset,
        "src_val_subset": src_val_subset,
        "src_test_subset": src_test_subset,

        # class mapping
        "src_class_to_idx": src_class_to_idx,
        "src_idx_to_class": src_idx_to_class,

        # all labels
        "src_all_idx": src_all_idx,
        "src_all_labels": src_all_labels,

        # sampled
        "src_sampled_idx": src_sampled_idx,
        "src_sampled_labels": src_sampled_labels,
        "src_y_binary_sampled": src_y_binary_sampled,

        # split before train benign filtering
        "src_train_all_idx_sampled": src_train_all_idx_sampled,
        "src_val_idx": src_val_idx,
        "src_test_idx_before_balance": src_test_idx,
        "src_y_val_bin": src_y_val_bin,
        "src_y_test_bin_before_balance": src_y_test_bin,

        # final split
        "src_train_idx": src_train_idx,
        "test_idx": test_idx,
        "y_test_src": y_test_src,
    }


# ============================================================
# Optional DataLoader builder
# ============================================================

def build_dataloaders_from_prepared_data(
    prepared_data: Dict[str, object],
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    train_shuffle: bool = True
):
    """Build data loaders for the prepared splits."""

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        prepared_data["src_train_subset"],
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        prepared_data["src_val_subset"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        prepared_data["src_test_subset"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader
    }


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    device = get_device(verbose=True)

    config = NPYOneClassConfig(
        root="/workspace/data/NB15/train_224_npy_100%_k3_merged",
        benign_name="Benign",
        split_seed=42,
        per_class_frac=0.28,
        test_size=0.3,
        valtest_test_size=0.5,
        ram_ratio=0.6,
        max_full_load_available_ratio=0.8,
        balanced_test=True
    )

    data = prepare_oneclass_npy_data_same_as_original(config)

    loaders = build_dataloaders_from_prepared_data(
        prepared_data=data,
        batch_size=128,
        num_workers=0,
        train_shuffle=True
    )

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]

    src_full_ds = data["src_full_ds"]
    src_train_idx = data["src_train_idx"]
    src_val_idx = data["src_val_idx"]
    test_idx = data["test_idx"]
    y_test_src = data["y_test_src"]

    print("DataLoader ready.")
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
