from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_vectorizer():
    return TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
