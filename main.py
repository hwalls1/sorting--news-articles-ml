from data_loader import fetch_dataset
from vectorizer import create_tfidf_vectorizer
from classifiers import benchmark_classifiers

def main():
    categories = [
        'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
        'sci.space', 'sci.crypt', 'sci.electronics', 'sci.med',
        'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'
    ]

    print("Loading 20 newsgroups dataset for categories:", categories)
    
    data_train = fetch_dataset(categories, subset='train')
    data_test = fetch_dataset(categories, subset='test')

    vectorizer = create_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)

    print("Training samples:", X_train.shape[0], ", Features:", X_train.shape[1])
    print("Testing samples:", X_test.shape[0], ", Features:", X_test.shape[1])
    
    benchmark_classifiers(X_train, data_train.target, X_test, data_test.target)

if __name__ == '__main__':
    main()
