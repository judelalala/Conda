import torch
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score,accuracy_score
import numpy as np

def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    # predicts=np.around(predicts,0).astype('int')
    # labels = labels.cpu()
    # print(labels.shape,sum(labels))
    # print(np.around(predicts,0).astype('int'))
    # acc=accuracy_score(labels,np.around(predicts,0).astype('int'))
    # f1 = f1_score(labels, np.around(predicts,0).astype('int'))
    # recall = recall_score(labels, np.around(predicts,0).astype('int'))
    # average_precision = average_precision_score(y_true=labels, y_score=predicts)
    # roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
    binarized_predicts = np.around(predicts, 0).astype('int')

    # acc = accuracy_score(labels, binarized_predicts)
    # f1 = f1_score(labels, binarized_predicts)
    # recall = recall_score(labels, binarized_predicts)
    average_precision = average_precision_score(y_true=labels, y_score=predicts)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    f1_micro = f1_score(labels, binarized_predicts, average='micro')
    f1_macro = f1_score(labels, binarized_predicts, average='macro')

    return {
        
        'average_precision': average_precision,
        'roc_auc': roc_auc,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro
    }

    # return {'average_precision': average_precision, 'roc_auc': roc_auc}
    # return {'average_precision': average_precision, 'roc_auc': roc_auc, 'recall': recall, 'f1_score': f1,'acc': acc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}
