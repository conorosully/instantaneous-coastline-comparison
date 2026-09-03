from scipy.ndimage import distance_transform_edt

import numpy as np
import pandas as pd
import utils

def confusion_metrics(target, pred):
    """Returns confusion matrix metrics for a single image."""
    TP = np.sum((pred == 1) & (target == 1))
    TN = np.sum((pred == 0) & (target == 0))
    FP = np.sum((pred == 1) & (target == 0))
    FN = np.sum((pred == 0) & (target == 1))
    return TP, TN, FP, FN


def confusion_metrics_per_image(targets, preds):
    """Returns per-image confusion matrix counts as four lists: TPs, TNs, FPs, FNs."""
    TPs, TNs, FPs, FNs = [], [], [], []
    for target, pred in zip(targets, preds):
        tp, tn, fp, fn = confusion_metrics(np.array(target), np.array(pred))
        TPs.append(int(tp))
        TNs.append(int(tn))
        FPs.append(int(fp))
        FNs.append(int(fn))
    return TPs, TNs, FPs, FNs

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

    denom = np.maximum(np.count_nonzero(img), np.count_nonzero(ref_img))
    if denom == 0:
        return np.nan
    fom /= denom

    return fom

def fom_per_image(targets, preds):
    """Return per-image TP, TN, FP, FN and FOM as a list of dicts."""
    FOM = []
    for i in range(len(targets)):
        target = np.array(targets[i])
        pred   = np.array(preds[i])


        target_edge = utils.edge_from_mask(target)
        pred_edge   = utils.edge_from_mask(pred)
        fom = calc_fom(target_edge, pred_edge)

        FOM.append(fom)
    return FOM


def aggregate_metrics(TPs, TNs, FPs, FNs, foms=None):
    """
    Compute micro-averaged metrics from per-image confusion matrix counts.

    TPs, TNs, FPs, FNs : sequences of per-image counts (lists, arrays, or pandas Series)
    foms               : optional sequence of per-image FOM values; included in output if provided
    """
    total_TP = sum(TPs)
    total_TN = sum(TNs)
    total_FP = sum(FPs)
    total_FN = sum(FNs)

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 1.0
    recall    = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 1.0
    pos_iou   = total_TP / (total_TP + total_FP + total_FN)
    neg_iou   = total_TN / (total_TN + total_FP + total_FN)

    result = {
        "accuracy":  (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN),
        "precision": precision,
        "recall":    recall,
        "f1":        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0,
        "pos_iou":   pos_iou,
        "neg_iou":   neg_iou,
        "mean_iou":  (pos_iou + neg_iou) / 2,
    }

    if foms is not None:
        fps = list(FPs)
        fns = list(FNs)
        fom_vals = []
        for i, f in enumerate(foms):
            if not np.isnan(f):
                fom_vals.append(f)
            elif fps[i] == 0 and fns[i] == 0:
                fom_vals.append(1.0)
            else:
                fom_vals.append(0.0)
        a = np.array(fom_vals, dtype=float)
        result["fom"] = np.nan if len(a) == 0 else np.nanmean(a)

    return result


def calculate_reporting_metrics(model_rows):
    """Aggregate per-image rows for a single model into one reporting-metrics dict."""
    rows = model_rows.copy().reset_index(drop=True)

    model_name = rows['model_name'][0]
    model_type = rows['model_type'][0]
    dataset = rows['dataset'][0]
    satellite = rows['satellite'][0]

    if "NDWI" in model_type:
        optimizer = "N/A"
        n_params = 0
        augmentation = "N/A"
    else:
        optimizer = rows['optimizer'][0]
        n_params = rows['n_params'][0]
        augmentation = rows['augmentation'][0]

    TP = rows['TP']
    TN = rows['TN']
    FP = rows['FP']
    FN = rows['FN']
    FOM = rows['fom']

    metrics = aggregate_metrics(TP, TN, FP, FN, FOM)

    return {
        "model_name": model_name,
        "model_type": model_type,
        "dataset": dataset,
        "satellite": satellite,
        "optimizer": optimizer,
        "n_params": n_params,
        "augmentation": augmentation,
        **metrics
    }


def add_metrics_to_rows(df):
    """Compute accuracy/f1/iou/fom columns for a results dataframe of per-image confusion counts."""
    df = df.copy()
    df['accuracy'] = (df['TP'] + df['TN']) / (df['TP'] + df['TN'] + df['FP'] + df['FN'])

    denom_f1 = 2 * df['TP'] + df['FP'] + df['FN']
    df['f1'] = np.where(denom_f1 > 0, 2 * df['TP'] / denom_f1, 1.0)

    denom_pos = df['TP'] + df['FP'] + df['FN']
    denom_neg = df['TN'] + df['FP'] + df['FN']
    df['pos_iou']  = np.where(denom_pos > 0, df['TP'] / denom_pos, 1.0)
    df['neg_iou']  = np.where(denom_neg > 0, df['TN'] / denom_neg, 1.0)
    df['mean_iou'] = (df['pos_iou'] + df['neg_iou']) / 2

    FOM_ = []
    for i, f in enumerate(df['fom']):
        if not np.isnan(f):
            FOM_.append(f)
        elif df['FP'].iloc[i] + df['FN'].iloc[i] == 0:
            FOM_.append(1.0)
        else:
            FOM_.append(0.0)
    df['fom'] = FOM_

    return df


def aggregate_experiment_metrics(model_results, experiment, extra_cols=None):
    """Aggregate per-image rows into one reporting-metrics row per model for a given experiment.

    model_results : the full per-image results dataframe
    experiment    : experiment number to filter on
    extra_cols    : optional column names to carry over from each model's first row
                     (e.g. encoder/pretrained/freeze_encoder for Exp4)
    """
    results = model_results[model_results['experiment'] == experiment]

    metrics_list = []
    for model_name in results['model_name'].unique():
        model_rows = results[results['model_name'] == model_name]
        metrics = calculate_reporting_metrics(model_rows)
        if extra_cols:
            row0 = model_rows.iloc[0]
            for col in extra_cols:
                metrics[col] = row0[col]
        metrics_list.append(metrics)

    return pd.DataFrame(metrics_list)
