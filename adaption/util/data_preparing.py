import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


def plot_confusion_matrix(y_true, y_score, threshold, pos_label=-1, save_path=None):

    # Convert scores to class labels.
    y_pred = np.where(np.asarray(y_score, dtype=np.float64) >= threshold, pos_label, 1 if pos_label == -1 else -1)

    # Count predictions by true class.
    cm = confusion_matrix(y_true, y_pred, labels=[1, pos_label])
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100

    # Show counts and row percentages.
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_perc[i, j]:.1f}%\n{cm[i, j]:,}"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues",
                xticklabels=["Benign", "Malicious"],
                yticklabels=["Benign", "Malicious"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()




def stratified_sample(df, label_col, frac, random_state=42):
    return df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=random_state)
    ).reset_index(drop=True)


def prepare_parquet_splits(
    parquet_path: str,
    label_col: str = "Attack",
    usage_ratio: float = 0.3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_state: int = 42,
    test_balance="downsample",
    verbose: bool = True,
):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0."

    df = pd.read_parquet(parquet_path).reset_index(drop=True)

    df_use = stratified_sample(df, label_col, usage_ratio, random_state)

    # train/val/test
    df_trainval, df_test = train_test_split(
        df_use, test_size=test_ratio, stratify=df_use["Label"], random_state=random_state
    )
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_ratio/(train_ratio+val_ratio),
        stratify=df_trainval["Label"],
        random_state=random_state
    )


    # train/val Benign
    X_train_benign = df_train[df_train[label_col] == "Benign"].drop(columns=[label_col,"Label"])
    X_val_benign = df_val[df_val[label_col] == "Benign"].drop(columns=[label_col,"Label"])


    if test_balance in ("downsample", "upsample"):
        df_ben = df_test[df_test[label_col] == "Benign"]
        df_mal = df_test[df_test[label_col] != "Benign"]

        if test_balance == "downsample":
            n = min(len(df_ben), len(df_mal))
            df_ben_bal = df_ben.sample(n=n, random_state=random_state, replace=False)
            df_mal_bal = df_mal.sample(n=n, random_state=random_state, replace=False)
        else:  # upsample
            n = max(len(df_ben), len(df_mal))
            df_ben_bal = resample(df_ben, replace=True, n_samples=n, random_state=random_state)
            df_mal_bal = resample(df_mal, replace=True, n_samples=n, random_state=random_state)

        df_test = pd.concat([df_ben_bal, df_mal_bal], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # X_test / y_test
    y_test = df_test[label_col].astype(str).map(lambda x: 1 if x == "Benign" else -1).to_numpy(dtype=np.int64)
    X_test = df_test.drop(columns=[label_col,"Label"])



    n_train = X_train_benign.shape[0]
    n_val   = X_val_benign.shape[0]
    n_test  = X_test.shape[0]

    n_ben = int((y_test == 1).sum())
    n_mal = int((y_test == -1).sum())

    if verbose:
        uniq, cnt = np.unique(y_test, return_counts=True)
        print(">>> Final Dataset Splits")
        print("Train Shape (Benign only):", X_train_benign.shape)
        print("Val Shape(Benign only):", X_val_benign.shape)
        print("Test Shape(All classes) :", X_test.shape, "  y_test(+1/-1)=", dict(zip(uniq, cnt)))
        print(f"SRC Train(benign only): {n_train}, SRC Val(all): {n_val}, SRC Test(Balanced): {n_test}")
        print(f"Malicious(-1): {n_mal}, Sampled benign (+1): {n_ben}")

    return X_train_benign, X_val_benign, X_test, y_test


def anomaly_scores(original, transformed):
    sse = np.sum((original - transformed)**2, axis=1)
    return sse

def select_validation_threshold(validation_scores, *, false_positive_rate=0.05):
    """Calibrate an anomaly threshold from benign validation scores."""
    scores = np.asarray(validation_scores, dtype=np.float64)
    if scores.ndim != 1 or not scores.size or not np.isfinite(scores).all():
        raise ValueError("Expected a nonempty vector of finite validation scores.")
    if not 0 <= false_positive_rate < 1:
        raise ValueError("false_positive_rate must be in [0, 1).")

    # Use a strict cutoff so tied benign scores cannot exceed the target rate.
    allowed = int(np.floor(scores.size * false_positive_rate))
    cutoff_index = scores.size - allowed - 1
    cutoff = np.partition(scores, cutoff_index)[cutoff_index]
    return float(np.nextafter(cutoff, np.inf))


def score_reconstruction(model, loader, device):
    """Return reconstruction MSE and labels in loader order."""
    import torch

    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device, non_blocking=True)
            reconstruction, _ = model(features)
            error = (reconstruction - features).square().flatten(1).mean(1)
            scores.append(error.cpu().numpy())
            labels.append(target.cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels)


def evaluate_results(y_true, score, *, threshold):
    """Evaluate scores using a separately calibrated threshold."""
    return evaluate_test_data(y_true, score, threshold)


def safe_divide(a, b):
    """Divide safely when the denominator is zero."""
    return a / b if b != 0 else 0.0


def compute_confusion_metrics(y_true, y_pred):
    """Compute binary classification metrics with attack as the positive class."""

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[1, -1]
    ).ravel()

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    tpr = recall
    fpr = safe_divide(fp, fp + tn)

    tnr = safe_divide(tn, tn + fp)
    fnr = safe_divide(fn, fn + tp)

    f1 = safe_divide(2 * precision * recall, precision + recall)

    results = {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "TPR": tpr,
        "FPR": fpr,
        "TNR": tnr,
        "FNR": fnr,
    }

    return results


def evaluate_predictions(y_true, y_pred):
    """Compute metrics from predicted class labels."""

    results = compute_confusion_metrics(y_true, y_pred)

    results["recall_sklearn"] = recall_score(
        y_true,
        y_pred,
        pos_label=-1,
        zero_division=0
    )

    results["precision_sklearn"] = precision_score(
        y_true,
        y_pred,
        pos_label=-1,
        zero_division=0
    )

    results["f1_sklearn"] = f1_score(
        y_true,
        y_pred,
        pos_label=-1,
        zero_division=0
    )

    return results


def evaluate_test_data(y_true, score, threshold):
    """Evaluate test scores using a fixed threshold."""

    # score >= threshold -> Attack(-1)
    # score < threshold  -> Benign(1)

    y_pred = np.where(np.asarray(score, dtype=np.float64) < threshold, 1, -1)

    results = evaluate_predictions(y_true, y_pred)

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_true,
        score,
        pos_label=-1
    )

    results["au_precision_recall"] = auc(recall_curve, precision_curve)

    fpr_curve, tpr_curve, _ = roc_curve(
        y_true,
        score,
        pos_label=-1
    )

    results["auroc"] = auc(fpr_curve, tpr_curve)
    results["threshold"] = threshold

    return results

def split_data_DANN(data_dir_src, filename_src, data_dir_tgt,filename_tgt,label_col="Attack",
                         usage_ratio_src = 0.25, usage_ratio_tgt = 0.15,train_ratio_tgt=0.7,val_ratio_tgt=0.2,test_ratio_tgt=0.1,
                         random_state=42, verbose=True):


    # Read parquet inputs.
    df_src = stratified_sample(pd.read_parquet(f"{data_dir_src}/{filename_src}"), label_col, usage_ratio_src)
    df_tgt = stratified_sample(pd.read_parquet(f"{data_dir_tgt}/{filename_tgt}"), label_col, usage_ratio_tgt)


    # Split the target data.
    df_trainval_tgt, df_test_tgt = train_test_split(df_tgt, test_size=test_ratio_tgt , stratify=df_tgt[label_col],random_state=random_state)
    df_train_tgt, df_val_tgt = train_test_split(df_trainval_tgt, test_size=val_ratio_tgt / (train_ratio_tgt + val_ratio_tgt),
                                                    stratify=df_trainval_tgt[label_col],random_state=random_state)

    # Keep benign training and validation samples.
    df_train_benign_tgt = df_train_tgt[df_train_tgt[label_col] == "Benign"].drop(columns=[label_col,"Label"])
    df_val_benign_tgt = df_val_tgt[df_val_tgt[label_col] == "Benign"].drop(columns=[label_col,"Label"])

    df_train_benign_src =  df_src[df_src[label_col] == "Benign"].drop(columns=[label_col,"Label"])

    # Encode test labels.
    y_test_tgt = df_test_tgt[label_col].map(lambda x: 1 if x == "Benign" else -1)
    X_test_tgt = df_test_tgt.drop(columns=[label_col,"Label"])

    if verbose:
        print(">>> Final Dataset Splits")
        print(f"Train shape SRC(Benign only): {df_train_benign_src.shape}")
        print(f"Train shape TGT(Benign only): {df_train_benign_tgt.shape}")
        print(f"Val shape TGT  (Benign only): {df_val_benign_tgt.shape}")
        print(f"Test shape TGT (All classes): {X_test_tgt.shape}")
        print("Test label distribution:")
        print(y_test_tgt.value_counts())

    return df_train_benign_src, df_train_benign_tgt, df_val_benign_tgt, X_test_tgt ,y_test_tgt

def split_data_DANN_balanced(
    data_dir_src, filename_src,
    data_dir_tgt, filename_tgt,
    label_col="Attack",
    usage_ratio_src=0.25, usage_ratio_tgt=0.15,
    train_ratio_tgt=0.7, val_ratio_tgt=0.2, test_ratio_tgt=0.1,
    test_balance="downsample",
    random_state=42, verbose=True
):


    # Read parquet inputs.
    df_src = stratified_sample(pd.read_parquet(f"{data_dir_src}/{filename_src}"), label_col, usage_ratio_src)
    df_tgt = stratified_sample(pd.read_parquet(f"{data_dir_tgt}/{filename_tgt}"), label_col, usage_ratio_tgt)


    # Split the target data.
    df_trainval_tgt, df_test_tgt = train_test_split(df_tgt, test_size=test_ratio_tgt , stratify=df_tgt[label_col],random_state=random_state)
    df_train_tgt, df_val_tgt = train_test_split(df_trainval_tgt, test_size=val_ratio_tgt / (train_ratio_tgt + val_ratio_tgt),
                                                    stratify=df_trainval_tgt[label_col],random_state=random_state)

    # Keep benign training and validation samples.
    df_train_benign_tgt = df_train_tgt[df_train_tgt[label_col] == "Benign"].drop(columns=[label_col,"Label"])
    df_val_benign_tgt = df_val_tgt[df_val_tgt[label_col] == "Benign"].drop(columns=[label_col,"Label"])

    df_train_benign_src =  df_src[df_src[label_col] == "Benign"].drop(columns=[label_col,"Label"])

    # Balance benign and malicious test samples.
    if test_balance in ("downsample", "upsample"):
        df_ben = df_test_tgt[df_test_tgt[label_col] == "Benign"]
        df_mal = df_test_tgt[df_test_tgt[label_col] != "Benign"]

        if test_balance == "downsample":
            n = min(len(df_ben), len(df_mal))
            df_ben_bal = df_ben.sample(n=n, random_state=random_state, replace=False)
            df_mal_bal = df_mal.sample(n=n, random_state=random_state, replace=False)
        else:  # upsample
            n = max(len(df_ben), len(df_mal))
            df_ben_bal = resample(df_ben, replace=True, n_samples=n, random_state=random_state)
            df_mal_bal = resample(df_mal, replace=True, n_samples=n, random_state=random_state)

        df_test_tgt = pd.concat([df_ben_bal, df_mal_bal], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Prepare test features and labels.
    y_test_tgt = df_test_tgt[label_col].map(lambda x: 1 if x == "Benign" else -1)
    X_test_tgt = df_test_tgt.drop(columns=[label_col,"Label"])

    if verbose:
        print(">>> Final Dataset Splits")
        print(f"Train shape SRC(Benign only): {df_train_benign_src.shape}")
        print(f"Train shape TGT(Benign only): {df_train_benign_tgt.shape}")
        print(f"Val shape TGT  (Benign only): {df_val_benign_tgt.shape}")
        print(f"Test shape TGT (All classes): {X_test_tgt.shape}")
        print("Test label distribution (1=Benign, -1=Malicious):")
        print(y_test_tgt.value_counts().sort_index())

    return df_train_benign_src, df_train_benign_tgt, df_val_benign_tgt, X_test_tgt, y_test_tgt
