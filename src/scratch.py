"""PRD section 10 mathematics written directly in NumPy, checked against sklearn in the tests."""

import numpy as np


def normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares weights including an intercept (first element), via lstsq on the design matrix."""
    design = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def mse_gradient(design: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gradient of the MSE cost J(w) = (1/n) ||Xw - y||^2, which is (2/n) X^T (Xw - y)."""
    return 2 / len(y) * design.T @ (design @ w - y)


def stable_learning_rate(design: np.ndarray) -> float:
    """Largest step that guarantees descent: 1 / L, where L = 2 * lambda_max(X^T X) / n is the gradient's Lipschitz constant."""
    return float(len(design) / (2 * np.linalg.eigvalsh(design.T @ design)[-1]))


def gradient_descent_lr(X: np.ndarray, y: np.ndarray, lr: float | None = None, epochs: int = 5000, tol: float = 1e-10) -> tuple:
    """Train linear regression by batch gradient descent. X should be standardised.

    Collinear features make a fixed step unstable, so the default step comes from the largest eigenvalue.
    Returns the weights (intercept first) and the MSE at each epoch.
    """
    design = np.column_stack([np.ones(len(X)), X])
    lr = stable_learning_rate(design) if lr is None else lr
    w = np.zeros(design.shape[1])
    history = []
    for _ in range(epochs):
        residual = design @ w - y
        history.append(float(residual @ residual / len(y)))
        step = lr * mse_gradient(design, y, w)
        w = w - step
        if np.max(np.abs(step)) < tol:
            break
    return w, np.array(history)


def predict_linear(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Linear prediction Xw with an intercept in w[0]."""
    return w[0] + X @ w[1:]


def mse(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((np.asarray(y, float) - p) ** 2))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mse(y, p)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y, float) - p)))


def mape(y: np.ndarray, p: np.ndarray) -> float:
    """Mean absolute percentage error as a fraction, ignoring zero actuals (no percentage exists there)."""
    y = np.asarray(y, float)
    nonzero = y != 0
    return float(np.mean(np.abs((y[nonzero] - p[nonzero]) / y[nonzero])))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot."""
    y = np.asarray(y, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def binary_cross_entropy(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    """Binary cross-entropy: -mean(y log p + (1 - y) log(1 - p)), with p kept away from 0 and 1."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def confusion_matrix(y: np.ndarray, p: np.ndarray, n_classes: int) -> np.ndarray:
    """Counts with actual classes on the rows and predicted classes on the columns."""
    return np.bincount(n_classes * np.asarray(y, int) + np.asarray(p, int), minlength=n_classes**2).reshape(n_classes, n_classes)


def precision_recall_f1(y: np.ndarray, p: np.ndarray, n_classes: int) -> tuple:
    """Per-class precision, recall and F1. A class with an empty denominator scores 0."""
    cm = confusion_matrix(y, p, n_classes)
    tp = np.diag(cm).astype(float)
    predicted, actual = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(tp), where=total > 0)
    return precision, recall, f1


def macro_f1(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of the per-class F1 scores."""
    return float(precision_recall_f1(y, p, n_classes)[2].mean())


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve by the trapezoid rule, with tied scores grouped into one step."""
    y = np.asarray(y, int)
    order = np.argsort(-score, kind="stable")
    y_sorted, score_sorted = y[order], score[order]
    last_of_tie = np.r_[np.where(np.diff(score_sorted))[0], len(y) - 1]
    tp = np.cumsum(y_sorted)[last_of_tie]
    fp = last_of_tie + 1 - tp
    tpr = np.r_[0, tp] / y.sum()
    fpr = np.r_[0, fp] / (len(y) - y.sum())
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def zscore_fit(X: np.ndarray) -> tuple:
    """Per-column mean and standard deviation, treating a constant column's deviation as 1."""
    mean, std = X.mean(axis=0), X.std(axis=0)
    return mean, np.where(std == 0, 1.0, std)


def zscore_transform(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardise: (X - mean) / std."""
    return (X - mean) / std


def minmax_fit(X: np.ndarray) -> tuple:
    """Per-column minimum and maximum, keeping a constant column's range non-zero."""
    low, high = X.min(axis=0), X.max(axis=0)
    return low, np.where(high == low, low + 1.0, high)


def minmax_transform(X: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Rescale into [0, 1] over the fitted range: (X - low) / (high - low)."""
    return (X - low) / (high - low)


def covariance(X: np.ndarray) -> np.ndarray:
    """Sample covariance: C = (X - mean)^T (X - mean) / (n - 1)."""
    centred = X - X.mean(axis=0)
    return centred.T @ centred / (len(X) - 1)


def correlation(X: np.ndarray) -> np.ndarray:
    """Correlation matrix from the covariance: C_ij / (sigma_i sigma_j)."""
    cov = covariance(X)
    sigma = np.sqrt(np.diag(cov))
    return cov / np.outer(sigma, sigma)


def variance_inflation(X: np.ndarray) -> np.ndarray:
    """VIF per column: the diagonal of the inverse correlation matrix."""
    return np.diag(np.linalg.pinv(correlation(X)))
