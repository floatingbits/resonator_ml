from collections import defaultdict
import torch
import numpy as np


class PerSampleLossTracker:
    """
    Tracks per-sample losses across training steps / epochs.

    Assumes each sample has a stable integer ID.
    """

    def __init__(self):
        # sample_id -> list[loss]
        self.loss_history = defaultdict(list)
        # optional: step index for debugging / plotting
        self.step_history = defaultdict(list)
        self.prediction_history = defaultdict(list)
        self.global_step = 0

    def update(self, sample_ids, per_sample_losses, y_pred):
        """
        sample_ids: Tensor or list, shape [B]
        per_sample_losses: Tensor, shape [B]
        """
        sample_ids = sample_ids.detach().cpu().tolist()
        losses = per_sample_losses.detach().cpu().tolist()
        predictions = y_pred.detach().cpu().tolist()

        for sid, loss, prediction in zip(sample_ids, losses, predictions):
            self.loss_history[sid].append(loss)
            self.prediction_history[sid].append(prediction)
            self.step_history[sid].append(self.global_step)

        self.global_step += 1

    # --------------------------------------------------
    # Basic statistics
    # --------------------------------------------------
    def last_prediction(self, sample_id):
        return float(self.prediction_history[sample_id][-1][0])
    def mean_prediction(self, sample_id):
        return float(np.mean(self.prediction_history[sample_id]))
    def max_prediction(self, sample_id):
        return float(np.max(self.prediction_history[sample_id]))
    def min_prediction(self, sample_id):
        return float(np.min(self.prediction_history[sample_id]))
    def mean_loss(self, sample_id):
        return float(np.mean(self.loss_history[sample_id]))

    def max_loss(self, sample_id):
        return float(np.max(self.loss_history[sample_id]))

    def quantile_loss(self, sample_id, q=0.95):
        return float(np.quantile(self.loss_history[sample_id], q))

    # --------------------------------------------------
    # Dataset-level diagnostics
    # --------------------------------------------------

    def worst_samples(self, k=20, by="quantile", q=0.95):
        """
        Returns top-k worst samples.

        by: 'mean', 'max', 'quantile'
        """
        scores = []

        for sid in self.loss_history:
            if by == "mean":
                score = self.mean_loss(sid)
            elif by == "max":
                score = self.max_loss(sid)
            elif by == "quantile":
                score = self.quantile_loss(sid, q)
            else:
                raise ValueError(f"Unknown criterion: {by}")

            prediction = None
            max_prediction = None
            min_prediction = None
            if self.prediction_history[sid]:
                prediction = self.mean_prediction(sid)
                max_prediction = self.max_prediction(sid)
                min_prediction = self.min_prediction(sid)
                last_prediction = self.last_prediction(sid)

            scores.append((sid, score, prediction, max_prediction, min_prediction, last_prediction))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

    # --------------------------------------------------
    # Oscillation / persistence diagnostics
    # --------------------------------------------------

    def loss_autocorrelation(self, sample_id, lag=1):
        """
        Measures whether a sample stays hard over time.
        High value -> persistent difficulty.
        """
        losses = np.array(self.loss_history[sample_id])
        if len(losses) <= lag:
            return np.nan

        x = losses[:-lag]
        y = losses[lag:]

        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0

        return float(np.corrcoef(x, y)[0, 1])

    def persistent_hard_samples(self, k=20, q=0.95):
        """
        Samples with high loss AND high autocorrelation.
        """
        scored = []

        for sid in self.loss_history:
            score = self.quantile_loss(sid, q)
            corr = self.loss_autocorrelation(sid)

            if not np.isnan(corr):
                scored.append((sid, score, corr))

        scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return scored[:k]
