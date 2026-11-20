import torch
from sklearn.metrics import auc, roc_curve
import numpy as np
import optuna

class AUROCEarlyStoppingPruneCallback:
    """Utility class for early stopping and Optuna pruning based on AUROC."""

    def __init__(self, x_val, y_val, trial, min_delta=0.0, patience=0,
                 verbose=0, baseline=None, restore_best_weights=False):
        self.x_val = torch.tensor(x_val, dtype=torch.float32)
        self.y_val = y_val
        self.trial = trial
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights

        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = baseline if baseline is not None else -np.inf
        self.best_state = None

    def evaluate(self, model, device):
        model.eval()
        with torch.no_grad():
            inputs = self.x_val.to(device)
            outputs = model(inputs)
            sse = torch.sum((inputs - outputs) ** 2, dim=1).cpu().numpy()
        fpr, tpr, _ = roc_curve(self.y_val, sse, pos_label=-1)
        return auc(fpr, tpr)

    def check(self, epoch, model, device):
        current = self.evaluate(model, device)
        if self.verbose:
            print(f"Epoch #{epoch}\tValidation AUROC: {current}\tBest AUROC: {self.best_score}")
        self.trial.report(float(current), step=epoch)
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned(f"Trial was pruned at epoch {epoch}.")

        if current > self.best_score + self.min_delta:
            self.best_score = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_state is not None:
                    model.load_state_dict(self.best_state)
                return True
        return False
