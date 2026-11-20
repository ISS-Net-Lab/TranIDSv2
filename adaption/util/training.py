from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, matthews_corrcoef, average_precision_score, roc_auc_score, precision_recall_curve, auc, roc_curve
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np

# def load_data(data_dir, filename, verbose=True, train_size=0.8, test_size=none, data2017=False):
#     df = pd.read_csv(f"{data_dir}/{filename}")
#     if data2017:
#         Y = df["Label"].map(lambda x: 1 if (x == "BENIGN") else -1)
#         labels = df["Label"]
#         df.drop(columns=["Label"], inplace=True)
#     else:
#         Y = df["Label"].map(lambda x: 1 if (x == "Benign") else -1)
#         labels = df["Label"]
#         df.drop(columns=["Label"], inplace=True)

#     X_train, X_test, y_train, y_test = train_test_split(df, Y, train_size=train_size, test_size=test_size, shuffle=True, stratify=Y, random_state=42)
#     if verbose:
#         print("***** Train Data *****")
#         print(labels.loc[y_train.index].value_counts())
#         print("***** Test Data *****")
#         print(labels.loc[y_test.index].value_counts())
#     return X_train, X_test, y_train, y_test, labels.loc[y_train.index], labels.loc[y_test.index]

# def load_data_oneclass(data_dir, filename, verbose=True, train_size=0.8):
#     df = pd.read_csv(f"{data_dir}/{filename}")
#     Y = df["Label"].map(lambda x: 1 if (x == "Benign") else -1)
#     labels = df["Label"]
#     df.drop(columns=["Label"], inplace=True)
#     X_train, X_test, y_train, y_test = train_test_split(df[Y==1], Y[Y==1], train_size=train_size, shuffle=True, random_state=42)
#     X_test = pd.concat([X_test, df[Y==-1]])
#     y_test = pd.concat([y_test, Y[Y==-1]])
#     if verbose:
#         print("***** Train Data *****")
#         print(labels.loc[y_train.index].value_counts())
#         print("***** Test Data *****")
#         print(labels.loc[y_test.index].value_counts())
#     return X_train, X_test, y_train, y_test, labels.loc[y_train.index], labels.loc[y_test.index]



def split_data_oneclass(data_dir, filename, label_col="Label", normal_label=1, anomaly_label=-1,
                                   train_size=0.8, val_size=0.1, test_size=0.1, random_state=42, verbose=True):
    """Stratify the splits and retain only benign rows for training."""

    assert train_size + val_size + test_size == 1.0, "Split ratios must sum to 1.0."

    df = pd.read_csv(f"{data_dir}/{filename}")

    # Encode the binary labels.
    y = df[label_col].map(lambda x: normal_label if x == "Benign" else anomaly_label)
    X = df.drop(columns=[label_col])

    # Stratify the initial split by class.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), stratify=y, random_state=random_state
    )

    # Stratify the validation and test splits.
    val_relative_size = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_relative_size, stratify=y_temp, random_state=random_state
    )

    # Keep only benign training samples.
    X_train = X_train[y_train == normal_label]
    y_train = y_train[y_train == normal_label]

    if verbose:
        print("***** Train Data *****")
        print(f"Train shape: {X_train.shape}")
        print(y_train.value_counts())
        print("\n***** Validation Data *****")
        print(f"Val shape: {X_val.shape}")
        print(y_val.value_counts())
        print("\n***** Test Data *****")
        print(f"Test shape: {X_test.shape}")
        print(y_test.value_counts())

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_data_fraud(data_dir, verbose=True, data2017=False):
    df = pd.read_csv(f"{data_dir}/all_malicious.csv")
    if data2017:
        Y = df["Label"].map(lambda x: 1 if (x == "BENIGN") else -1)
        labels = df["Label"]
        df.drop(columns=["Label", "Timestamp", "Destination Port"], inplace=True)
    else:
        Y = df["Label"].map(lambda x: 1 if (x == "Benign") else -1)
        labels = df["Label"]
        df.drop(columns=["Label", "Timestamp", "Dst Port"], inplace=True)
    if verbose:
        print("***** Data *****")
        print(labels.value_counts())
    return df, Y, labels

# def tune_hyperparam(X, y, pipe, hyper_param, folds=5, verbose=1, n_jobs=-1, refit="f1", scoring=None):
#     if scoring is None:
#         scoring = {
#             "accuracy": "accuracy",
#             "recall": make_scorer(recall_score, pos_label=-1, zero_division=0),
#             "precision": make_scorer(precision_score, pos_label=-1, zero_division=0),
#             "f1": make_scorer(f1_score, pos_label=-1, zero_division=0),
#             "mcc": make_scorer(matthews_corrcoef),
#             "auroc": make_scorer(roc_auc_score, needs_threshold=True),
#             "average_precision": make_scorer(average_precision_score, needs_threshold=True)
#         }
#     cv = GridSearchCV(pipe, hyper_param, scoring=scoring, cv=folds, refit=refit, return_train_score=False, n_jobs=n_jobs, verbose=verbose)
#     cv.fit(X, y)
#     if refit:
#         print(f"Best parameters set found on development set: {cv.best_params_}")
#         print(f"Mean score of best model ({refit}): {cv.best_score_}")
#     return cv

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
