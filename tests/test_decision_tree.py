import os
import sys
import numpy as np

# ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from decision_tree import DecisionTreeClassifier


def test_perfect_stump():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert np.array_equal(preds, y)


def test_all_same_label():
    X = np.random.randn(10, 3)
    y = np.zeros(10, dtype=int)
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert np.all(preds == 0)


def test_depth_limit_vs_full_tree():
    # XOR dataset — needs depth 2 to represent perfectly
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    clf_shallow = DecisionTreeClassifier(max_depth=1)
    clf_shallow.fit(X, y)
    preds_shallow = clf_shallow.predict(X)
    # shallow should not represent XOR perfectly
    assert not np.array_equal(preds_shallow, y)

    clf_deep = DecisionTreeClassifier(max_depth=2)
    clf_deep.fit(X, y)
    preds_deep = clf_deep.predict(X)
    assert np.array_equal(preds_deep, y)


def test_multiclass_small():
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 2, 2])
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert np.array_equal(preds, y)


def test_predict_proba_shape():
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 1])
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (3, clf.n_classes_)
