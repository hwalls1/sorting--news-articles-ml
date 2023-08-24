from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics
from metrics import calculate_accuracy

def benchmark_classifiers(X_train, y_train, X_test, y_test):
    classifiers = [MultinomialNB(alpha=.01), BernoulliNB(alpha=.01)]
    for clf in classifiers:
        result = benchmark_classifier(clf, X_train, y_train, X_test, y_test)
        print(f"{result['name']} Accuracy: {result['score']:.4f}")

def benchmark_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = calculate_accuracy(y_test, pred)
    return {'name': clf.__class__.__name__, 'score': score}
