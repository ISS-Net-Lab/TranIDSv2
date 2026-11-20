from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, matthews_corrcoef, average_precision_score, roc_auc_score, precision_recall_curve, auc, roc_curve
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np



# def split_data_train_val(data_dir, filename, label_col="Label",
#                                    val_size=0.2,  random_state=42, verbose=True):
#     """

#     """

#     df_train = pd.read_parquet(f"{data_dir}/{filename}")
#     df_train = df_train[df_train[label_col] == "Benign"]
#     X = df_train.drop(columns=[label_col])

#     X_train, X_val = train_test_split(X, test_size=val_size, random_state=random_state)

#     return X_train,X_val

# def get_data_test(data_dir, filename, label_col="Label"):


#     df_test = pd.read_parquet(f"{data_dir}/{filename}")
#     y_test = df_test[label_col].map(lambda x: 1 if x == "Benign" else -1)
#     X_test = df_test.drop(columns=[label_col])


#     return X_test,y_test

def split_data(data_dir, filename, label_col="Label",
                         train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                         random_state=42, verbose=True):
    """Split sampled parquet data and keep only benign rows for training and validation."""

    # Read the parquet data.
    df = pd.read_parquet(f"{data_dir}/{filename}")

    # Keep one half of the input.
    df_half, _ = train_test_split(df, test_size=0.5, random_state=random_state)

    # Split the sampled data.
    df_trainval, df_test = train_test_split(df_half, test_size=test_ratio , random_state=random_state)
    df_train, df_val = train_test_split(df_trainval, test_size=val_ratio / (train_ratio + val_ratio), random_state=random_state)

    # Keep benign training and validation samples.
    df_train_benign = df_train[df_train[label_col] == "Benign"].drop(columns=[label_col])
    df_val_benign = df_val[df_val[label_col] == "Benign"].drop(columns=[label_col])

    # Encode the test labels.
    y_test = df_test[label_col].map(lambda x: 1 if x == "Benign" else -1)
    X_test = df_test.drop(columns=[label_col])

    if verbose:
        print(">>> Final Dataset Splits")
        print(f"Train shape (Benign only): {df_train_benign.shape}")
        print(f"Val shape   (Benign only): {df_val_benign.shape}")
        print(f"Test shape (All classes): {X_test.shape}")
        print("Test label distribution:")
        print(y_test.value_counts())

    return df_train_benign, df_val_benign, X_test, y_test




def anomaly_scores(original, transformed):
    sse = np.sum((original - transformed)**2, axis=1)
    return sse

def evaluate_results(y_true, score, *, threshold):
    """Evaluate scores using a separately calibrated threshold."""
    return evaluate_test_data(y_true, score, threshold)

def evaluate_predictions(y_true, y_pred):
    results = {}
    results['recall'] = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
    results['precision'] = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
    results['f1'] = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
    return results

def evaluate_test_data(y_true, score, threshold):
    y_pred = np.where(np.asarray(score, dtype=np.float64) < threshold, 1, -1)
    results = evaluate_predictions(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, score, pos_label=-1)
    results['au_precision_recall'] = auc(recall, precision)
    fpr, tpr, thresholds = roc_curve(y_true, score, pos_label=-1)
    results["auroc"] = auc(fpr, tpr)
    return results
