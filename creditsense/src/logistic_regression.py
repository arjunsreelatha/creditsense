import numpy as np
from numpy.typing import NDArray



class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: NDArray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []
    def _sigmoid(self, z: NDArray) -> NDArray:
        z = np.asarray(z, dtype=float)
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
    def fit(self, X: NDArray, y: NDArray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0
        self.loss_history = []
        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias
            p = self._sigmoid(z)
            p = np.clip(p, 1e-9, 1 - 1e-9)
            loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
            self.loss_history.append(float(loss))
            dw = (X.T @ (p - y)) / n_samples
            db = np.mean(p - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def predict_proba(self, X: NDArray) -> NDArray:
        X = np.asarray(X, dtype=float)
        z = X @ self.weights + self.bias
        p = self._sigmoid(z)
        return np.clip(p, 1e-9, 1 - 1e-9)
    def predict(self, X: NDArray, threshold: float = 0.5) -> NDArray:
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    y_proba = model.predict_proba(X)
    y_pred = model.predict(X)
    accuracy = np.mean(y == y_pred)
    precision = np.sum((y == 1) & (y_pred == 1)) / max(np.sum(y_pred == 1), 1)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Final loss:", model.loss_history[-1])