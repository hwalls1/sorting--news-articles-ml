from sklearn.datasets import fetch_20newsgroups

def fetch_dataset(categories, subset='train', remove=('header', 'footers', 'quotes')):
    return fetch_20newsgroups(subset=subset, categories=categories, shuffle=True,
                              random_state=42, remove=remove)
