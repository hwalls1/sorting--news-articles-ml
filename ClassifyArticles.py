import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics


def fetch_dataset(categories, subset='train', remove=('header', 'footers', 'quotes')):
    """Fetch 20 newsgroups dataset."""
    return fetch_20newsgroups(subset=subset, categories=categories, shuffle=True,
                              random_state=42, remove=remove)


def benchmark(clf, X_train, y_train, X_test, y_test):
    """Train and test a classifier; return metrics."""
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)
    
    # You can add more metrics here as needed
    return {'name': clf.__class__.__name__, 'score': score}


def main():
    # Define categories
    categories = [
        'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
        'sci.space', 'sci.crypt', 'sci.electronics', 'sci.med',
        'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'
    ]

    print("Loading 20 newsgroups dataset for categories:", categories)
    
    # Fetch data
    data_train = fetch_dataset(categories, subset='train')
    data_test = fetch_dataset(categories, subset='test')

    # Vectorize data
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)

    print("Training samples:", X_train.shape[0], ", Features:", X_train.shape[1])
    print("Testing samples:", X_test.shape[0], ", Features:", X_test.shape[1])
    
    # Train and benchmark classifiers
    classifiers = [MultinomialNB(alpha=.01), BernoulliNB(alpha=.01)]
    for clf in classifiers:
        result = benchmark(clf, X_train, data_train.target, X_test, data_test.target)
        print(f"{result['name']} Accuracy: {result['score']:.4f}")


if __name__ == '__main__':
    main()
