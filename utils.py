import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    V = true.size(0)
    n_classes = pred.shape[1] if pred.ndim > 1 else 2
    label_count = torch.bincount(true)
    label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
    cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
    cluster_sizes[torch.unique(true)] = label_count
    weight = (V - cluster_sizes).float() / V
    weight *= (cluster_sizes > 0).float()
    pred = F.log_softmax(pred, dim=-1)
    return F.nll_loss(pred, true, weight=weight,reduction='none')

def Entropy(x, min_val=1e-32):
    p = F.softmax(x, dim=1)
    entropy = -(p * p.clamp(min=min_val).log()).sum(dim=1)
    return entropy

def calculate_confusion_matrix_from_tensors(y_pred, y_true, num_classes):
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix

def f1_score_from_confusion_matrix(confusion_matrix):
    precision = torch.diag(confusion_matrix) / confusion_matrix.sum(0)
    precision[torch.isnan(precision)] = 0
    recall = torch.diag(confusion_matrix) / confusion_matrix.sum(1)
    recall[torch.isnan(recall)] = 0
    f1 = 2. * precision * recall / (precision + recall)
    f1[torch.isnan(f1)] = 0
    return torch.mean(precision), torch.mean(recall), torch.mean(f1)

def evalution(y_true, y_pred, num_classes):
    confusion_matrix = calculate_confusion_matrix_from_tensors(y_pred, y_true, num_classes)
    precision, recall, macro_f1_score = f1_score_from_confusion_matrix(confusion_matrix)
    return  macro_f1_score
