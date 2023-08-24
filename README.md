# 20 Newsgroups Classifier

This program classifies articles from the "20 Newsgroups" dataset into predefined categories using Naive Bayes classifiers.

## Dependencies

- **Python 3**
- **NumPy** (>= 1.8.2)
- **SciPy** (>= 0.13.3)
- **scikit-learn**

For visualization (if implemented in future iterations):
- **matplotlib**

Reference for `scikit-learn`: [Official Documentation](http://scikit-learn.org/stable/index.html)

## Overview

The script fetches articles from the "20 Newsgroups" dataset and focuses on classifying them into the following categories:
- `rec.motorcycles`
- `rec.sport.baseball`
- `rec.sport.hockey`
- `sci.space`
- `sci.crypt`
- `sci.electronics`
- `sci.med`
- `talk.politics.misc`
- `talk.politics.guns`
- `talk.politics.mideast`

The classifiers utilized for this task are:
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

## Usage

To run the script, navigate to the directory containing the script and execute:

```bash
python python main.py
