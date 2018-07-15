# Run with recent versions of scipy, numpy and scikit-learn to avoid bugs in train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pathlib


def validate_na_fraction(input=None):
    """
    Fxn to check whether input is between 0 and 1
    :param input: float in range [0,1]
    :return: None
    """
    if not 0 <= input <= 1:
        raise ValueError('na_fraction must be between 0 and 1')


def generate_classification_data(na_fraction=None):
    """
    Fxn to generate data for classification problem
    :param na_fraction: float in range [0,1] denoting the fraction of missing values to be introduced when generating the data
    :return: None
    """
    if na_fraction is not None:
        validate_na_fraction(input=na_fraction)

    print('Generating classification data... with na_fraction={}.'.format(na_fraction))
    data = make_classification(n_samples=5000,
                               n_features=100,
                               n_informative=20,
                               n_redundant=2,
                               n_repeated=0,
                               n_classes=2,
                               n_clusters_per_class=2,
                               weights=None,
                               flip_y=0.01,
                               class_sep=1.0,
                               hypercube=True,
                               shift=0.0,
                               scale=1.0,
                               shuffle=True,
                               random_state=None)
    df = pd.DataFrame(data[0])
    target = pd.DataFrame(data[1])
    target.columns = ['target']
    df['target'] = target.target
    df.to_csv('data/clf_dta.csv', index=False)

    y = df.target
    X = df.drop(['target'], axis=1)
    print('Train/test split ...')
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=0.2,
                                                          random_state=42)
    print('Saving training test to file ...')
    if na_fraction is not None:
        X_train = X_train.mask(np.random.random(X_train.shape) < na_fraction)
        X_valid = X_valid.mask(np.random.random(X_valid.shape) < na_fraction)

    X_train.join(y_train).to_csv('data/X_train_clf.csv', index=False)
    X_valid.join(y_valid).to_csv('data/X_valid_clf.csv', index=False)


def generate_regression_data(na_fraction=None):
    """
  Fxn to generate data for regression problem
  :param na_fraction: float in range [0,1] denoting the fraction of missing values to be introduced when generating the data
  :return: None
  """
    if na_fraction is not None:
        validate_na_fraction(input=na_fraction)

    print('Generating regression data... with na_fraction={}.'.format(na_fraction))
    data = make_regression(n_samples=5000,
                           n_features=100,
                           n_informative=20,
                           n_targets=1,
                           bias=0.0,
                           effective_rank=None,
                           tail_strength=0.5,
                           noise=0.0,
                           shuffle=True,
                           coef=False,
                           random_state=None)

    df = pd.DataFrame(data[0])
    target = pd.DataFrame(data[1])
    target.columns = ['target']
    df['target'] = target.target
    df.to_csv('data/reg_dta.csv', index=False)

    y = df.target
    X = df.drop(['target'], axis=1)

    print('Train/test split ...')
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Saving training test to file ...')
    if na_fraction is not None:
        X_train = X_train.mask(np.random.random(X_train.shape) < na_fraction)
        X_valid = X_valid.mask(np.random.random(X_valid.shape) < na_fraction)

    X_train.join(y_train).to_csv('data/X_train_reg.csv', index=False)
    X_valid.join(y_valid).to_csv('data/X_valid_reg.csv', index=False)


if __name__ == '__main__':
    pathlib.Path('./data').mkdir(parents=True, exist_ok=True)
    generate_classification_data(na_fraction=None)
    generate_regression_data(na_fraction=None)

