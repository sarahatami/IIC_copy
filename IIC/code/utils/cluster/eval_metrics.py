from __future__ import print_function

import numpy as np
import torch
from sklearn import metrics
# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment as linear_assignment


def _original_match(flat_preds, flat_targets, preds_k, targets_k):
    # map each output channel to the best matching ground truth (many to one)
    print("_original_match started ***********************************")

    assert (isinstance(flat_preds, torch.Tensor) and
            isinstance(flat_targets, torch.Tensor) and
            flat_preds.is_cuda and flat_targets.is_cuda)

    out_to_gts = {}
    out_to_gts_scores = {}
    for out_c in range(preds_k):
        for gt_c in range(targets_k):
            # the amount of out_c at all the gt_c samples
            tp_score = int(((flat_preds == out_c) * (flat_targets == gt_c)).sum())
            if (out_c not in out_to_gts) or (tp_score > out_to_gts_scores[out_c]):
                out_to_gts[out_c] = gt_c
                out_to_gts_scores[out_c] = tp_score
    # return list(out_to_gts.iteritems())

    # sara wrote
    # print("out_to_gts: ",out_to_gts)
    # print("list(out_to_gts.values())",list(out_to_gts.values()))
    return list(out_to_gts.values())


def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    print("_hungarian_match started ***********************************")
    assert (isinstance(flat_preds, torch.Tensor) and
            isinstance(flat_targets, torch.Tensor) and
            flat_preds.is_cuda and flat_targets.is_cuda)

    num_samples = flat_targets.shape[0]  # 16181138

    # print("flat_preds:",type(flat_preds)) #tensor 1*[16181138]
    # print("flat_targets:",type(flat_targets)) #tensor 1*[16181138]
    # print("config.output_k",preds_k) #3
    # print("config.gt_k",targets_k) #3

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))  # matrix 3*3

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_assignment(num_samples - num_correct)  # num_samples - num_correct = 3*3 matrix

    print("num_samples - num_correct",num_samples - num_correct)
    print("match", match)  # tuple:(array([0, 1, 2], dtype=int64), array([1, 0, 2], dtype=int64))
    print("match[0]",match[0])  # [0, 1, 2]
    print("match[1]",match[1])  # [1, 0, 2]
    # match=[(1,2),(3,2),(1,3)]
    # match=[match]

    # return as list of tuples, out_c to gt_c
    res = []
    print("#################################################3")
    for out_c, gt_c in match:
        print("out_c, gt_c: ", out_c, gt_c)
        res.append((out_c, gt_c))

    return res


def _acc(preds, targets, num_k, verbose=0):
    assert (isinstance(preds, torch.Tensor) and
            isinstance(targets, torch.Tensor) and
            preds.is_cuda and targets.is_cuda)

    if verbose >= 2:
        print("calling acc...")

    assert (preds.shape == targets.shape)
    assert (preds.max() < num_k and targets.max() < num_k)

    acc = int((preds == targets).sum()) / float(preds.shape[0])

    return acc


def _nmi(preds, targets):
    return metrics.normalized_mutual_info_score(targets, preds)


def _ari(preds, targets):
    return metrics.adjusted_rand_score(targets, preds)
