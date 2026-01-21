import numpy as np
import numba as nb
import logging
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.absolute()))
sys.path.append(str(pathlib.Path(__file__).parent.parent.absolute()))

from utils.utils import setup_logger

logger = logging.getLogger(__name__)

# TODO: this is tested only for window_size == 3
@nb.jit(nopython=True)
def calculate_fpr_tpr_window(ground_truth, prediction, thresh, num_windows, window_size):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(num_windows):
            if ground_truth[i+(window_size//2)] == prediction[i+(window_size//2)]:  # If the center of the window is the same
                if ground_truth[i+(window_size//2)] == 1:
                        tp += 1
                else:
                        tn += 1
            else:
                if ground_truth[i+(window_size//2)] == 1:
                        if prediction[i:i+window_size].sum() == 0:   # 010 || 000
                            fn += 1
                        elif prediction[i:i+window_size].sum() == 1: # 010 || 001/100
                            tp += 1
                        else: # prediction[i:i+window_size].sum() > 1: # 010 || 101
                            fn += 1
                else:   # ground_truth[i+(window_size//2)] == 0:
                        if ground_truth[i:i+window_size].sum() == 0:   # 000 || 010/011/110/111
                            fp += 1
                        else:   # ground_truth[i:i+window_size].sum() == 1:
                            if prediction[i+(window_size//2)] == 1:   # 001/100 || 010
                                    tp += 1
                            else:   # prediction[i+(window_size//2)].sum > 1:   # 001/100 || 011/110/111
                                    fp += 1


    # logger.info(f'thresh: {thresh}, tp: {tp}, fp: {fp}, tn: {tn}, fn: {fn}')
            
    # Calculate the true positive rate
    if tp == 0 and fn == 0:
            tpr = 0
    else:
            tpr = tp / (tp + fn)

    # Calculate the false positive rate
    if fp == 0 and tn == 0:
            fpr = 0
    else:
            fpr = fp / (fp + tn)

    return tpr, fpr

# TODO: this is tested only for window_size == 3
def window_roc_curve(ground_truth, prediction, window_size=3, prediction_round=3):
    prediction = np.array(prediction).round(prediction_round)
    thresholds = set(prediction)
    thresholds.add(0)
    thresholds.add(1)
    thresholds = sorted(thresholds)
#     logger.info(f'number of thresholds: {len(thresholds)}')

    tpr = []
    fpr = []
    thresholds1 = []
    ground_truth = [0] + list(ground_truth) + [0]
    prediction = [0] + list(prediction) + [0]
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)
    num_windows = len(ground_truth) - window_size + 1

    for thresh in thresholds:
        output_preds = np.array(prediction.copy())
        output_preds[output_preds>=thresh] = 1
        output_preds[output_preds<thresh] = 0

        # Calculate the true positive rate
        p_tpr, p_fpr = calculate_fpr_tpr_window(ground_truth, output_preds, thresh, num_windows, window_size)

        if not(p_tpr in tpr and p_fpr in fpr):
                tpr.append(p_tpr)
                fpr.append(p_fpr)
                thresholds1.append(thresh)

    return np.array(fpr), np.array(tpr), np.array(thresholds1)