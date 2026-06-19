import numpy as np
from typing import Optional


class _Node:
	__slots__ = ("is_leaf", "prediction", "proba", "feature", "threshold", "left", "right")

	def __init__(self):
		self.is_leaf = False
		self.prediction = None
		self.proba = None
		self.feature = None
		self.threshold = None
		self.left: Optional[_Node] = None
		self.right: Optional[_Node] = None


class DecisionTreeClassifier:
	"""A minimal CART-style decision tree sufficient for unit tests.

	Supports `max_depth`, `fit`, `predict` and `predict_proba` and exposes
	`n_classes_` after fitting.
	"""

	def __init__(self, max_depth: int = 2):
		self.max_depth = max_depth
		self.root: Optional[_Node] = None
		self.n_classes_ = 0

	def _gini(self, y):
		if len(y) == 0:
			return 0.0
		_, counts = np.unique(y, return_counts=True)
		probs = counts / counts.sum()
		return 1.0 - np.sum(probs ** 2)

	def _best_split(self, X, y):
		n_samples, n_features = X.shape
		if n_samples <= 1:
			return None

		best = None
		best_imp = 1.0

		parent_gini = self._gini(y)

		for feat in range(n_features):
			values = X[:, feat]
			uniq = np.unique(values)
			if uniq.shape[0] == 1:
				continue
			thresholds = (uniq[:-1] + uniq[1:]) / 2.0
			for thr in thresholds:
				left_mask = values <= thr
				right_mask = ~left_mask
				if left_mask.sum() == 0 or right_mask.sum() == 0:
					continue
				g_left = self._gini(y[left_mask])
				g_right = self._gini(y[right_mask])
				w = left_mask.sum() / n_samples
				imp = w * g_left + (1 - w) * g_right
				if imp < best_imp:
					best_imp = imp
					best = (feat, thr, left_mask, right_mask)

		return best

	def _build(self, X, y, depth=0):
		node = _Node()
		classes, counts = np.unique(y, return_counts=True)
		node.prediction = classes[np.argmax(counts)]
		probs = np.zeros(len(classes)) if len(classes) > 0 else np.array([1.0])
		# build proba vector for all global classes later in predict_proba
		node.proba = (classes, counts / counts.sum())

		# stopping conditions
		if depth >= self.max_depth or len(classes) == 1 or X.shape[0] <= 1:
			node.is_leaf = True
			return node

		split = self._best_split(X, y)
		if split is None:
			node.is_leaf = True
			return node

		feat, thr, left_mask, right_mask = split
		node.feature = feat
		node.threshold = thr
		node.left = self._build(X[left_mask], y[left_mask], depth + 1)
		node.right = self._build(X[right_mask], y[right_mask], depth + 1)
		return node

	def fit(self, X, y):
		X = np.asarray(X)
		y = np.asarray(y)
		if X.ndim == 1:
			X = X.reshape(-1, 1)

		classes = np.unique(y)
		self.n_classes_ = classes.shape[0]
		self._classes_ = classes
		self.root = self._build(X, y, depth=0)

	def _traverse(self, x):
		node = self.root
		while node is not None and not node.is_leaf:
			if x[node.feature] <= node.threshold:
				node = node.left
			else:
				node = node.right
		return node

	def predict(self, X):
		X = np.asarray(X)
		if X.ndim == 1:
			X = X.reshape(-1, 1)
		preds = []
		for x in X:
			node = self._traverse(x)
			preds.append(node.prediction)
		return np.array(preds)

	def predict_proba(self, X):
		X = np.asarray(X)
		if X.ndim == 1:
			X = X.reshape(-1, 1)
		proba = np.zeros((X.shape[0], self.n_classes_))
		class_index = {c: i for i, c in enumerate(self._classes_)}
		for i, x in enumerate(X):
			node = self._traverse(x)
			classes, probs = node.proba
			for c, p in zip(classes, probs):
				proba[i, class_index[c]] = p
		return proba


if __name__ == "__main__":
	# quick smoke
	X = np.array([[0], [1], [2], [3]])
	y = np.array([0, 0, 1, 1])
	clf = DecisionTreeClassifier(max_depth=1)
	clf.fit(X, y)
	print(clf.predict(X))