# Harrison Walls Junior IS
# Requires Python >= 2.7
# NumPy >= 1.8.2
# SciPy >= 0.13.3
# Refernece: http://scikit-learn.org/stable/index.htm


from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics

def main():

# Uploads categories from the "20 Newsgroups" dataset.
    categories = ['rec.motorcycles','rec.sport.baseball','rec.sport.hockey',
        'sci.space', 'sci.crypt', 'sci.electronics', 'sci.med',
        'talk.politics.misc','talk.politics.guns','talk.politics.mideast'
        ]

    remove = ('header','footers','quotes')

    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

# Loads training dataset from "20 newsgroups"
    data_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42,
                                remove=remove)

# Loads test dataset from "20 newsgroups"
    data_test = fetch_20newsgroups(subset='test', categories=categories,
                               shuffle=True, random_state=42,
                               remove=remove)

    print("%d categories" % len(categories))

    print("Extracting features from the training data using a sparse vectorizer")

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
    print("Number of samples: %d, Number of features: %d" % X_train.shape)

    print("Extracting features from the test data using the same vectorizer")
    X_test = vectorizer.transform(data_test.data)
    print("Number of samples: %d, Number of features: %d" % X_test.shape)
    print()

    results = []


# Train Naive Bayes classifiers
    print("Naive Bayes")
    results.append(benchmark(MultinomialNB(alpha=.01)))
    results.append(benchmark(BernoulliNB(alpha=.01)))


if __name__ == '__main__':
    main()
