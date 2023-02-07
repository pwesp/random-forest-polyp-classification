import numpy as np
from   sklearn.metrics import confusion_matrix, roc_curve



def adjusted_prediction(y_score, threshold=0.5, positive_label=1):
    return [1 if y >= threshold else 0 for y in y_score[:,positive_label]]



def eval_fpr_tpr(y_test, y_score, positive_label=1):
    fpr = []
    tpr = []
    for i in range(2):
        fpr_, tpr_, thr_ = roc_curve(y_true = y_test, y_score = y_score[:, i], pos_label = positive_label)
        fpr.append(fpr_)
        tpr.append(tpr_)
    return fpr, tpr



def eval_sensitivity_specificity(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity    = float(tp) / float(tp + fn)
    specifity      = float(tn) / float(tn + fp)
    return sensitivity, specifity