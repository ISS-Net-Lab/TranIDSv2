import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, f1_score, make_scorer
from sklearn.model_selection import learning_curve
import os

random.seed(0)

def get_results(cv, orderby='f1', save_dir=None, return_summary=True):
    df = pd.DataFrame(cv.cv_results_).sort_values(by=[f'rank_test_{orderby}'])
    df_summary = df[[c for c in df.columns if 'mean_test_' in c or 'param_' in c]]
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_csv(f'{save_dir}/results.csv')
        df_summary.to_csv(f'{save_dir}/results_summary.csv')
    if return_summary:
        return df_summary
    else:
        return df

def plot_graphs(cv, X, y, save_dir=None, learning_curve=True, suffix=None):
    y_pred = cv.predict(X)
    y_proba = cv.decision_function(X)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if suffix is None:
        suffix = "_train" if learning_curve else "_test"

    # Confusion Matrix
    plt = plot_confusion_matrix(y, y_pred)
    if save_dir:
        plt.savefig(f'{save_dir}/confusion_matrix{suffix}', bbox_inches='tight')
    plt.show()
    plt.clf()

    # AUROC
    plt = plot_roc(y, y_proba)
    if save_dir:
        plt.savefig(f'{save_dir}/auroc{suffix}', bbox_inches='tight')
    plt.show()
    plt.clf()

    # Precision-Recall Curve
    plt = plot_precision_recall(y, y_proba)
    if save_dir:
        plt.savefig(f'{save_dir}/precision_recall{suffix}', bbox_inches='tight')
    plt.show()
    plt.clf()

    # Learning Curve
    if learning_curve:
        plt = plot_learning_curve(cv, X, y, cv=5, n_jobs=-1, scoring=make_scorer(f1_score, pos_label=-1))
        if save_dir:
            plt.savefig(f'{save_dir}/learning_curve', bbox_inches='tight')
        plt.show()
        plt.clf()

def plot_confusion_matrix(y_true, y_pred, figsize=(7,7), cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_pred))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float)
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            annot[i, j] = '%.1f%%\n%d' % (p * 100, c)
    cm_perc = pd.DataFrame(cm_perc, index=["fraud", "benign"], columns=["fraud", "benign"])
    cm_perc.index.name = 'Actual'
    cm_perc.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_perc, cmap=cmap, annot=annot, fmt='', ax=ax, vmin=0, vmax=1)
    return plt

def plot_roc(y_true, y_proba):
    # low proba is outlier, high proba is inlier
    fpr, tpr, thresholds = roc_curve(y_true, -y_proba, pos_label=-1)
    plt.plot(fpr, tpr, 'k-', lw=2, label='AUROC = %.2f' % auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristics")
    plt.legend(loc='lower right')
    return plt

def plot_precision_recall(y_true, y_proba):
    # low proba is outlier, high proba is inlier
    precision, recall, thresholds = precision_recall_curve(y_true, -y_proba, pos_label=-1)
    plt.plot(recall, precision, 'k-', lw=2, label='AUC = %.2f' % auc(recall, precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.title('Precision-Recall curve')
    return plt

def plot_distribution_scores(y_true, score, hist=True, bins=None, kde=True, labels=None, norm_hist=None, min_count=0):
    if labels is None:
        sns.distplot(score[y_true == 1], bins=bins, kde=kde, color='g', label='Benign')
        sns.distplot(score[y_true == -1], bins=bins, kde=kde, color='r', label='Fraud')
    else:
        plt.figure(figsize=(10,7))
        for label, count in zip(*np.unique(labels, return_counts=True)):
            if count > min_count:
                sns.distplot(score[labels == label], hist=hist, bins=bins, kde=kde, label=label, norm_hist=norm_hist)
        plt.legend(fontsize='small', loc='best')
    plt.xlabel('Squared Reconstruction Error')
    plt.title("Squared Error Distribution")
    return plt

def plot_learning_curve(estimator, X, y, title='Learning Curve', axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), scoring=None):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    scoring : string, callable or None, optional (default=None)
        A string (see model evaluation documentation) or a scorer callable
        object / function with signature scorer(estimator, X, y).
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(f'{title}')
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples (k)")
    axes[0].set_ylabel("Score")

    train_sizes_absolute, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    x = train_sizes_absolute / 1000
    axes[0].grid()
    axes[0].fill_between(x, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes[0].fill_between(x, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes[0].plot(x, train_scores_mean, 'o-', color="r", label="Training score")
    axes[0].plot(x, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(x, fit_times_mean, 'o-')
    axes[1].fill_between(x, fit_times_mean - fit_times_std, fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples (k)")
    axes[1].set_ylabel("fit_times (s)")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times (s)")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
