import sklearn.metrics
import numpy as np
def cal_map(labels,scores):
    pass
    top_items_index=np.argsort(-scores)
    labels=labels[top_items_index]
    precision=0
    #number of true sample
    true_sample=0
    for i in xrange(labels.shape[0]):
        if labels[i]==0:
            continue
        else:
            true_sample+=1
            precision+=float(true_sample)/(i+1)
    ap=precision/np.sum(labels)
    return ap

def dcg_at_k(y_score, k=5, method=0):
    y_score = np.asfarray(y_score)[:k]
    if y_score.size:
        if method == 0:
            return y_score[0] + np.sum(y_score[1:] / np.log2(np.arange(2, y_score.size + 1)))
        elif method == 1:
            return np.sum(y_score / np.log2(np.arange(2, y_score.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0

def cal_NDCG(item_scores,  method=0):
    k=item_scores.shape[0]
    amin, amax = item_scores.min(), item_scores.max()
    y_score = (item_scores - amin) / (amax - amin)

    dcg_max = dcg_at_k(sorted(y_score, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(y_score, k, method) / dcg_max

def cal_auc(labels,scores):
    return sklearn.metrics.roc_auc_score(y_true=labels,y_score=scores)
def cal_precision(labels,scores,top_k=50):
    top_items_index=np.argpartition(-scores, top_k)[:top_k]
    hits=0
    for i in top_items_index:
        if labels[i]==1:
            hits+=1
    return hits/float(top_k)
def cal_recall(labels,scores,top_k=50):
    top_items_index=np.argpartition(-scores, top_k)[:top_k]
    hits=0
    for i in top_items_index:
        if labels[i]==1:
            hits+=1
    #print np.sum(labels)
    return hits/float(np.sum(labels))