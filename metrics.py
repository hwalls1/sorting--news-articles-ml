from sklearn import metrics

def calculate_accuracy(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)
