from scipy.ndimage import distance_transform_edt

import numpy as np
import utils

def confusion_metrics(pred, target):
    """Returns confusion matrix metrics"""

    TP = np.sum((pred == 1) & (target == 1))
    TN = np.sum((pred == 0) & (target == 0))
    FP = np.sum((pred == 1) & (target == 0))
    FN = np.sum((pred == 0) & (target == 1))

    return TP, TN, FP, FN

def calc_fom(ref_img, img, alpha=1.0 / 9.0):
    """
    Computes Pratt's Figure of Merit for the given image img, using a gold
    standard image as source of the ideal edge pixels.
    """

    # Compute the distance transform for the gold standard image.
    dist = distance_transform_edt(1 - ref_img)

    N, M = img.shape
    fom = 0
    for i in range(N):
        for j in range(M):
            if img[i, j]:
                fom += 1.0 / (1.0 + dist[i, j] * dist[i, j] * alpha)

    fom /= np.maximum(np.count_nonzero(img), np.count_nonzero(ref_img))

    return fom


def eval_metrics(targets, preds):
    """Evaluate model performance on test set"""

    r_accuracy = []
    r_balanced_accuracy = []
    r_precision = []
    r_recall = []
    r_f1 = []
    r_mse = []
    r_fom = []
    # Calculate metrics for each image
    for i in range(len(targets)):
        target = np.array(targets[i])
        pred = np.array(
            preds[i],
        )

        # segmentation metrics
        TP_, TN_, FP_, FN_ = confusion_metrics(pred, target)

        accuracy = (TP_ + TN_) / (TP_ + TN_ + FP_ + FN_)
        balanced_accuracy = 0.5 * (TP_ / (TP_ + FN_) + TN_ / (TN_ + FP_)) if (TP_ + FN_) > 0 and (TN_ + FP_) > 0 else np.nan
        precision = TP_ / (TP_ + FP_) if (TP_ + FP_) > 0 else np.nan
        recall = TP_ / (TP_ + FN_) if (TP_ + FN_) > 0 else np.nan
        f1 = 2 * (precision * recall) / (precision + recall) if (not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0) else np.nan

        r_accuracy.append(accuracy)
        r_balanced_accuracy.append(balanced_accuracy)
        r_precision.append(precision)
        r_recall.append(recall)
        r_f1.append(f1)

        # Edge detection metrics
        target_edge = utils.edge_from_mask(target)
        pred_edge = utils.edge_from_mask(pred)

        TP_, TN_, FP_, FN_ = confusion_metrics(pred_edge, target_edge)

        mse = (FP_ + FN_) / (TP_ + TN_ + FP_ + FN_)
        fom = calc_fom(target_edge, pred_edge)

        r_mse.append(mse)
        r_fom.append(fom)

    accuracy = np.nanmean(r_accuracy)
    balanced_accuracy = np.nanmean(r_balanced_accuracy)
    precision = np.nanmean(r_precision)
    recall = np.nanmean(r_recall)
    f1 = np.nanmean(r_f1)
    mse = np.nanmean(r_mse)
    fom = np.nanmean(r_fom)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mse": mse,
        "fom": fom,
    }, {"accuracy": r_accuracy, "fom": r_fom}