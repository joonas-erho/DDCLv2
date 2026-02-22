import pickle
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from util import (
    get_dataset_fp_list,
    label_to_vect_dict,
    make_onset_feature_context_range,
    windowize,
)

def micro_prf(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Micro precision/recall/F1 for multi-label predictions."""
    y_true_b = y_true.astype(bool)
    y_pred_b = y_pred.astype(bool)

    tp = np.sum(y_true_b & y_pred_b)
    fp = np.sum(~y_true_b & y_pred_b)
    fn = np.sum(y_true_b & ~y_pred_b)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(precision), float(recall), float(f1)


def iter_fixed_onset_windows(
    test_txt_fp: str,
    stream_labels_fp: str,
    memlen: int = 15,
    use_all_charts: bool = True,
) -> Iterator[Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]]:
    """
    Deterministically enumerate ALL onset windows in the test split.

    Yields:
      ((ac, ac2, sd, sd2), y) per sample (window), where
        ac, ac2: (memlen+1, nframes, nmelbands, nchannels)
        sd, sd2: (memlen+1, 2)
        y:       (48,) (vectorized label)
    """
    test_files = get_dataset_fp_list(test_txt_fp)

    with open(stream_labels_fp, "rb") as f:
        labels = pickle.load(f)
    enc_dict = label_to_vect_dict(labels)

    for song_pkl in test_files:
        with open(song_pkl, "rb") as f:
            charts = pickle.load(f)

        feats_fp = charts[3]
        with open(feats_fp, "rb") as f:
            feats = pickle.load(f)

        n_charts = len(charts[0])
        chart_indices = range(n_charts) if use_all_charts else range(min(1, n_charts))

        for chart_num in chart_indices:
            beat_ranges = charts[0][chart_num]
            streams = charts[1][chart_num]
            label_strs = charts[2][chart_num]

            # Audio contexts per beat (deterministic)
            audio_ctx = [make_onset_feature_context_range(feats, x[0], x[1]) for x in beat_ranges]
            audio_ctx = np.asarray(audio_ctx)

            # Per-chart normalization (matches training intent, but deterministic)
            mean = np.mean(audio_ctx, axis=0)
            std = np.std(audio_ctx, axis=0)
            audio_ctx = (audio_ctx - mean) / (std + 1e-12)

            # Stream features: keep first 2 cols (difficulty, bpm or similar)
            streams = np.asarray([[a[0], a[1]] for a in streams])

            # Windowize deterministically
            ac = windowize(audio_ctx, front_set="min", frames=memlen, return_type="numpy")
            ac2 = windowize(audio_ctx, front_set="min", go_backwards=True, frames=memlen, return_type="numpy")
            sd = windowize(streams, frames=memlen, return_type="numpy")
            sd2 = windowize(streams, go_backwards=True, frames=memlen, return_type="numpy")

            y = np.asarray([enc_dict[s] for s in label_strs])

            # Yield sample-by-sample; batching happens in evaluator
            n = y.shape[0]
            for i in range(n):
                yield (ac[i], ac2[i], sd[i], sd2[i]), y[i]


def evaluate_onset_fixed_streaming(
    model_path: str,
    test_txt_fp: str,
    stream_labels_fp: str,
    batch_size: int = 32,
    memlen: int = 15,
    threshold: float = 0.5,
    threshold_grid: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float, float, float]]]:
    """
    Deterministic, single-pass evaluation over the full enumerated test set.

    - No random sampling
    - Memory bounded: stores only one batch at a time
    - Computes:
        * Log Loss (binary CE)
        * PR-AUC micro, ROC-AUC micro (flattened)
        * Precision/Recall/F1 at fixed threshold
        * Max F1 / Max Precision / Max Recall over threshold grid
    """
    if threshold_grid is None:
        threshold_grid = np.linspace(0.0, 1.0, 101).astype(np.float32)
    else:
        threshold_grid = np.asarray(threshold_grid, dtype=np.float32)

    model = load_model(model_path)

    auc_pr = tf.keras.metrics.AUC(curve="PR", from_logits=False)
    auc_roc = tf.keras.metrics.AUC(curve="ROC", from_logits=False)

    ll_sum = 0.0
    ll_count = 0

    tp = fp = fn = 0

    tp_t = np.zeros_like(threshold_grid, dtype=np.int64)
    fp_t = np.zeros_like(threshold_grid, dtype=np.int64)
    fn_t = np.zeros_like(threshold_grid, dtype=np.int64)

    buf_ac: list[np.ndarray] = []
    buf_ac2: list[np.ndarray] = []
    buf_sd: list[np.ndarray] = []
    buf_sd2: list[np.ndarray] = []
    buf_y: list[np.ndarray] = []

    def flush_batch() -> int:
        nonlocal ll_sum, ll_count, tp, fp, fn, tp_t, fp_t, fn_t

        if not buf_y:
            return 0

        ac_b = np.asarray(buf_ac)
        ac2_b = np.asarray(buf_ac2)
        sd_b = np.asarray(buf_sd)
        sd2_b = np.asarray(buf_sd2)
        y_true = np.asarray(buf_y)

        y_prob = model.predict((ac_b, ac2_b, sd_b, sd2_b), batch_size=len(y_true), verbose=0)

        # --- Log loss (binary CE over all labels) ---
        eps = 1e-7
        y_prob_clip = np.clip(y_prob, eps, 1.0 - eps)
        loss = -(y_true * np.log(y_prob_clip) + (1.0 - y_true) * np.log(1.0 - y_prob_clip))
        ll_sum += float(np.sum(loss))
        ll_count += int(np.prod(loss.shape))

        # --- AUC (micro by flattening) ---
        y_true_f = tf.convert_to_tensor(y_true.reshape(-1).astype(np.float32))
        y_prob_f = tf.convert_to_tensor(y_prob.reshape(-1).astype(np.float32))
        auc_pr.update_state(y_true_f, y_prob_f)
        auc_roc.update_state(y_true_f, y_prob_f)

        # --- Fixed-threshold PRF counts ---
        y_true_b = y_true.astype(bool)
        y_pred_b = (y_prob >= threshold)
        tp += int(np.sum(y_true_b & y_pred_b))
        fp += int(np.sum(~y_true_b & y_pred_b))
        fn += int(np.sum(y_true_b & ~y_pred_b))

        # --- Threshold grid counts (for max metrics) ---
        for j, t in enumerate(threshold_grid):
            y_pred_t = (y_prob >= t)
            tp_t[j] += int(np.sum(y_true_b & y_pred_t))
            fp_t[j] += int(np.sum(~y_true_b & y_pred_t))
            fn_t[j] += int(np.sum(y_true_b & ~y_pred_t))

        buf_ac.clear()
        buf_ac2.clear()
        buf_sd.clear()
        buf_sd2.clear()
        buf_y.clear()

        return int(y_true.shape[0])

    n_samples = 0
    for (x, y) in iter_fixed_onset_windows(
        test_txt_fp=test_txt_fp,
        stream_labels_fp=stream_labels_fp,
        memlen=memlen,
        use_all_charts=True,
    ):
        ac_i, ac2_i, sd_i, sd2_i = x
        buf_ac.append(ac_i)
        buf_ac2.append(ac2_i)
        buf_sd.append(sd_i)
        buf_sd2.append(sd2_i)
        buf_y.append(y)

        if len(buf_y) >= batch_size:
            n_samples += flush_batch()

    n_samples += flush_batch()

    logloss = ll_sum / max(1, ll_count)

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    p_grid = tp_t / (tp_t + fp_t + 1e-12)
    r_grid = tp_t / (tp_t + fn_t + 1e-12)
    f1_grid = 2 * p_grid * r_grid / (p_grid + r_grid + 1e-12)

    j_f1 = int(np.argmax(f1_grid))
    j_p = int(np.argmax(p_grid))
    j_r = int(np.argmax(r_grid))

    best = {
        "max_f1": (float(f1_grid[j_f1]), float(threshold_grid[j_f1]), float(p_grid[j_f1]), float(r_grid[j_f1])),
        "max_precision": (float(p_grid[j_p]), float(threshold_grid[j_p]), float(r_grid[j_p]), float(f1_grid[j_p])),
        "max_recall": (float(r_grid[j_r]), float(threshold_grid[j_r]), float(p_grid[j_r]), float(f1_grid[j_r])),
    }

    metrics = {
        "n_samples": float(n_samples),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "log_loss": float(logloss),
        "auc_pr_micro": float(auc_pr.result().numpy()),
        "auc_roc_micro": float(auc_roc.result().numpy()),
    }
    return metrics, best


def evaluate_onset(
    model_path: str,
    test_txt_fp: str,
    stream_labels_fp: str,
    batch_size: int = 32,
    memlen: int = 15,
    threshold: float = 0.5,
    threshold_grid: Optional[np.ndarray] = None,
):
    """
    Backwards-friendly wrapper name.
    (Previously this file contained extra unused parameters; they were removed.)
    """
    return evaluate_onset_fixed_streaming(
        model_path=model_path,
        test_txt_fp=test_txt_fp,
        stream_labels_fp=stream_labels_fp,
        batch_size=batch_size,
        memlen=memlen,
        threshold=threshold,
        threshold_grid=threshold_grid,
    )


if __name__ == "__main__":
    metrics, best = evaluate_onset_fixed_streaming(
        model_path="trained_models/onset_model.keras",
        test_txt_fp="onset/songs/songs_test.txt",
        stream_labels_fp="onset/songs/stream_labels.pkl",
        batch_size=32,
        memlen=15,
        threshold=0.5,
    )

    print(f"Samples evaluated: {int(metrics['n_samples'])}")
    print(f"AUC (PR, micro): {metrics['auc_pr_micro']:.6f}")
    print(f"AUC (ROC, micro): {metrics['auc_roc_micro']:.6f}")
    print(f"Log Loss (binary CE): {metrics['log_loss']:.6f}")
    print(
        f"Precision/Recall/F1 @ t={metrics['threshold']:.2f}: "
        f"P={metrics['precision']:.6f} R={metrics['recall']:.6f} F1={metrics['f1']:.6f}"
    )

    f1v, t, p, r = best["max_f1"]
    print(f"Max F1: {f1v:.6f} at threshold={t:.2f} (P={p:.6f}, R={r:.6f})")

    pv, t, r, f1v = best["max_precision"]
    print(f"Max Precision: {pv:.6f} at threshold={t:.2f} (R={r:.6f}, F1={f1v:.6f})")

    rv, t, p, f1v = best["max_recall"]
    print(f"Max Recall: {rv:.6f} at threshold={t:.2f} (P={p:.6f}, F1={f1v:.6f})")