import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import  DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
from transformer_AEViT_1D import ViTAutoencoderConfig, build_model, pretrain_one_epoch, validate,SplitFolderNpyDataset,build_split_and_balance_dataset
from util.data_preparing import (
    evaluate_test_data,
    plot_confusion_matrix,
    score_reconstruction,
    select_validation_threshold,
)
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, ThresholdPruner
import time
from pathlib import Path
import os
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
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


SPLIT_SEED = 42
SEEDS = [7,17, 27, 37, 47]
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
        np.random.seed(s); random.seed(s)
    return _seed_worker


def _fmt_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU found, using CPU")




# tgt_dir = '../data/New Data/CICIDS2018/data_pcaps_5'
tgt_dir = '../data/New Data/NB15/feature_out_5/_MERGED_ALL_NPY'

# =========================
# Adjustable input format
# =========================
INPUT_MODE = "packet_2d"   # "flat_1d" or "packet_2d"
# INPUT_MODE = "flat_1d"
PACKET_SHAPE = (5, 96)     # 5 packets, each 96 bytes
# train_ds = SplitFolderNpyDataset(
#     Path(tgt_dir) / "train",
#     use_label_file=False,
#     input_mode=INPUT_MODE,
#     packet_shape=PACKET_SHAPE,
# )

# test_ds = SplitFolderNpyDataset(
#     Path(tgt_dir) / "test",
#     use_label_file=False,
#     input_mode=INPUT_MODE,
#     packet_shape=PACKET_SHAPE,
# )


# train_subset, test_subset, benign_idx_train = build_split_and_balance_dataset(
#     full_ds=None,
#     train_ds=train_ds,
#     test_ds=test_ds,
#     benign_name="benign",
#     split_ratio=0.7,
#     train_use_ratio=1.0,
#     test_use_ratio=1.0,
#     train_target_size= 700000,
#     max_test_size_per_class= 10000,
#     seed=42
# )

full_ds = SplitFolderNpyDataset(
    Path(tgt_dir),
    use_label_file=False,
    input_mode=INPUT_MODE,
    packet_shape=PACKET_SHAPE,
)

train_subset, test_subset, benign_idx_train = build_split_and_balance_dataset(
    full_ds=full_ds,
    train_ds=None,
    test_ds=None,
    benign_name="benign",
    split_ratio=0.7,
    train_use_ratio=1.0,
    test_use_ratio=1.0,
    train_target_size= 150000,
    max_test_size_per_class= 10000,
    seed=42
)



pretrain_root = os.getenv(
    "TRANIDS_PRETRAIN_ROOT",
    "results_AEViT_image_2018_96_5_1000000_1/SVDD_image/optuna",
)
result_csv = os.path.join(pretrain_root, "result_imgSVDD.csv")
model_dir = os.path.join(pretrain_root, "models")

df = pd.read_csv(result_csv)
top_trials = df.head(1)

print(f"Trials: {len(df)} total, {len(top_trials)} retained")

rows = []
final_csv = "results_AEViT_image_2018_96_5_1000000_1/SVDD_image/optuna/models/ft_5layer_150000/fine_tune_results.csv"
os.makedirs(os.path.dirname(final_csv), exist_ok=True)

# pretrain_save_dir = 'results_SVDD_image_2018_v2_1/SVDD_image/optuna/models/pretrain_backup_1'
# os.makedirs(pretrain_save_dir, exist_ok=True)

ft_save_dir = 'results_AEViT_image_2018_96_5_1000000_1/SVDD_image/optuna/models/ft_5layer_150000'
os.makedirs(ft_save_dir, exist_ok=True)

all_idx = np.arange(len(train_subset))
train_idx, val_idx = train_test_split(
    all_idx, test_size=0.2, random_state=42, shuffle=True
)

seed_time = {}

for run_seed in SEEDS:
    set_global_seed(run_seed)
    worker_init_fn = make_worker(run_seed)
    g = torch.Generator()
    g.manual_seed(run_seed)

    seed_ft_accum = 0.0

    for _, trial_row in top_trials.iterrows():

        trial_number = int(trial_row["number"])
        bs = int(trial_row["params_batch_size"])
        # lr = float(trial_row["params_lr"])
        epochs = 50
        t0 = time.perf_counter()


        # cfg = ViTAutoencoderConfig(
        #     input_mode="packet_2d",
        #     packet_shape=(5, 96),
        #     patch_size=12,      # 96 / 12 = 8 patches
        #     embed_dim=192,
        #     latent_dim=64,
        #     depth=4,
        #     decoder_depth=2,
        #     num_heads=6,
        #     mlp_ratio=3.0,
        #     dropout=0.1,
        #     drop_path_rate=0.1,
        # )

        embed_dim = int(trial_row["params_embed_dim"])
        head_dim = int(trial_row["params_head_dim"])
        latent_dim = int(trial_row["params_latent_dim"])
        depth = int(trial_row["params_depth"])
        decoder_depth = int(trial_row["params_decoder_depth"])
        num_heads = embed_dim // head_dim

        cfg = ViTAutoencoderConfig(
            input_mode="packet_2d",
            packet_shape=(5, 96),
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


        # Create DataLoaders
        train_loader = DataLoader(
            Subset(train_subset, train_idx),
            batch_size=bs,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=g
        )

        val_loader = DataLoader(
            Subset(train_subset, val_idx),
            batch_size=bs,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=g
        )

        test_loader = DataLoader(
            test_subset,
            batch_size=bs,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            generator=g
        )

        model = build_model(cfg)
        model = model.to(cfg.device)

        model_path = os.path.join(model_dir, f"AE_trial_{trial_number}.pt")
        ckpt = torch.load(model_path, map_location=cfg.device)
        model.load_state_dict(ckpt["state_dict"])

        # =========================================
        # 1. Freeze all parameters first
        # =========================================
        for p in model.parameters():
            p.requires_grad = False


        # =========================================
        # 2. Set the number of decoder blocks to unfreeze
        # =========================================
        num_decoder_blocks = 2 # Change this value (1, 2, 3, ...)
        num_encoder_blocks = 1

        # =========================================
        # 3. Unfreeze patch_head and decoder_norm
        # =========================================
        base_modules = ["patch_head", "decoder_norm","bottleneck"]

        for module_name in base_modules:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                for p in module.parameters():
                    p.requires_grad = True
            else:
                print(f"[Warning] model has no module named: {module_name}")


        # =========================================
        # 4. Unfreeze only the last N decoder blocks
        # =========================================
        if hasattr(model, "decoder"):
            total_blocks = len(model.decoder)

            if num_decoder_blocks > total_blocks:
                raise ValueError(f"num_decoder_blocks={num_decoder_blocks} > total_blocks={total_blocks}")

            for i in range(total_blocks - num_decoder_blocks, total_blocks):
                print(f"[Info] Unfreezing decoder block {i}")
                for p in model.decoder[i].parameters():
                    p.requires_grad = True
        else:
            print("[Warning] model has no module named: decoder")

        if hasattr(model, "encoder"):
            total_blocks = len(model.encoder)

            if num_encoder_blocks > total_blocks:
                raise ValueError(f"num_encoder_blocks={num_encoder_blocks} > total_blocks={total_blocks}")

            for i in range(total_blocks - num_encoder_blocks, total_blocks):
                print(f"[Info] Unfreezing encoder block {i}")
                for p in model.encoder[i].parameters():
                    p.requires_grad = True
        else:
            print("[Warning] model has no module named: encoder")





        # =========================================
        # 5. Confirm which parameters will be trained
        # =========================================
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]

        print("Trainable params:")
        for n in trainable_params:
            print("  ", n)


        # =========================================
        # 6. Safety check
        # =========================================
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]

        if len(trainable_parameters) == 0:
            raise ValueError("No trainable parameters found after unfreezing.")
        opt = torch.optim.AdamW(trainable_parameters, lr=2e-4, weight_decay=0.0)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs * len(train_loader)
        )
        # Training
        best = float("inf")
        best_state = None
        patience = 10
        bad = 0

        for epoch in range(epochs):
            meters = pretrain_one_epoch(model, train_loader, opt, cfg, scheduler=scheduler)
            val = validate(model, val_loader, cfg)

            print(f"[{epoch:03d}] MSE_Loss={meters['loss']:.6f}  Val_MSE={val['val_loss']:.6f}")

            if val["val_loss"] < best - 1e-6:
                best = val["val_loss"]
                bad = 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
            else:
                bad += 1
                if bad >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break


        seed_ft_accum += (time.perf_counter() - t0)

        if best_state is not None:
            model.load_state_dict(best_state)

        validation_scores, _ = score_reconstruction(model, val_loader, cfg.device)
        threshold = select_validation_threshold(validation_scores)
        ft_ckpt = {
            "validation_threshold": threshold,
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "cfg": model.cfg.__dict__,
        }
        torch.save(ft_ckpt, os.path.join(ft_save_dir, f"SVDD_ft_trial_{trial_number}_{run_seed}.pt"))

        test_score, y_true = score_reconstruction(model, test_loader, cfg.device)
        results = evaluate_test_data(y_true, test_score, threshold)


        recall = results['recall']
        precision = results['precision']
        f1 = results['f1']
        au_precision_recall = results['au_precision_recall']
        auroc = results["auroc"]


        plot_confusion_matrix(y_true, test_score,threshold, pos_label=-1)

        metrics_dict = {
        "seed": run_seed,
        "trial_number":trial_number,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "au_precision_recall": au_precision_recall,
        "auroc": auroc,
        "threshold": threshold
        }

        rows.append(metrics_dict)

        print(f"SEED:{run_seed}")
        print(f"Evaluation Metrics for  trial {trial_number}")
        print(f"Threshold:            {threshold:.4f}")
        print(f"Recall:               {recall:.4f}")
        print(f"Precision:            {precision:.4f}")
        print(f"F1-score:             {f1:.4f}")
        print(f"AUC (PR Curve):       {au_precision_recall:.4f}")
        print(f"AUROC (ROC Curve):    {auroc:.4f}")

    seed_time[run_seed] = seed_ft_accum

df = pd.DataFrame(rows)
df["seed_ft_total_sec"] = df["seed"].map(seed_time)
df["seed_ft_total_hms"] = df["seed_ft_total_sec"].apply(_fmt_hms)
df.to_csv(final_csv, index=False)


import numpy as np, pandas as pd
df_out = pd.DataFrame(rows)
summary = (
    df_out.groupby("trial_number")[["precision","recall","f1","auroc","au_precision_recall"]]
          .agg(["mean","std"])
          .round(4)
)
print("\n=== Seed-wise summary (per trial): mean +/- std ===")
print(summary)
# It can also be saved separately:
summary.to_csv(final_csv.replace(".csv","_summary.csv"))
