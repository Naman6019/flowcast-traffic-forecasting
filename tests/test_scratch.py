import numpy as np
import pytest
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src import scratch


@pytest.fixture
def regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    y = 3 + X @ np.array([1.5, -2.0, 0.5, 0.0]) + rng.normal(scale=0.3, size=500)
    return X, y


@pytest.fixture
def scores():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 1000)
    return y, np.round(np.clip(y * 0.3 + rng.random(1000) * 0.7, 0, 1), 2)


def test_gradient_descent_matches_the_normal_equation(regression):
    X, y = regression
    Xs = scratch.zscore_transform(X, *scratch.zscore_fit(X))
    weights, history = scratch.gradient_descent_lr(Xs, y)
    assert np.allclose(weights, scratch.normal_equation(Xs, y), atol=1e-4)
    assert history[-1] < history[0]
    assert np.all(np.diff(history) <= 1e-12)


def test_normal_equation_matches_sklearn(regression):
    X, y = regression
    model = LinearRegression().fit(X, y)
    weights = scratch.normal_equation(X, y)
    assert weights[0] == pytest.approx(model.intercept_)
    assert np.allclose(weights[1:], model.coef_)
    assert np.allclose(scratch.predict_linear(X, weights), model.predict(X))


def test_gradient_formula_matches_finite_differences(regression):
    X, y = regression
    design = np.column_stack([np.ones(len(X)), X])
    w = np.array([0.5, -1.0, 2.0, 0.3, 0.1])
    numeric = np.zeros_like(w)
    for i in range(len(w)):
        step = np.zeros_like(w)
        step[i] = 1e-6
        numeric[i] = (scratch.mse(y, design @ (w + step)) - scratch.mse(y, design @ (w - step))) / 2e-6
    assert np.allclose(scratch.mse_gradient(design, y, w), numeric, atol=1e-5)


def test_regression_metrics_match_sklearn(regression):
    X, y = regression
    p = LinearRegression().fit(X[:, :2], y).predict(X[:, :2])
    assert scratch.mse(y, p) == pytest.approx(metrics.mean_squared_error(y, p), abs=1e-9)
    assert scratch.rmse(y, p) == pytest.approx(metrics.root_mean_squared_error(y, p), abs=1e-9)
    assert scratch.mae(y, p) == pytest.approx(metrics.mean_absolute_error(y, p), abs=1e-9)
    assert scratch.mape(y, p) == pytest.approx(metrics.mean_absolute_percentage_error(y, p), abs=1e-9)
    assert scratch.r2(y, p) == pytest.approx(metrics.r2_score(y, p), abs=1e-9)


def test_mape_ignores_zero_actuals():
    assert scratch.mape(np.array([0.0, 100.0]), np.array([5.0, 110.0])) == pytest.approx(0.1)


def test_classification_metrics_match_sklearn():
    rng = np.random.default_rng(2)
    y, p = rng.integers(0, 4, 800), rng.integers(0, 4, 800)
    assert np.array_equal(scratch.confusion_matrix(y, p, 4), metrics.confusion_matrix(y, p))
    precision, recall, f1 = scratch.precision_recall_f1(y, p, 4)
    assert np.allclose(precision, metrics.precision_score(y, p, average=None), atol=1e-9)
    assert np.allclose(recall, metrics.recall_score(y, p, average=None), atol=1e-9)
    assert np.allclose(f1, metrics.f1_score(y, p, average=None), atol=1e-9)
    assert scratch.macro_f1(y, p, 4) == pytest.approx(metrics.f1_score(y, p, average="macro"), abs=1e-9)


def test_empty_class_scores_zero_rather_than_nan():
    precision, recall, f1 = scratch.precision_recall_f1(np.array([0, 0]), np.array([0, 0]), 2)
    assert precision[1] == recall[1] == f1[1] == 0.0


def test_roc_auc_matches_sklearn_with_ties(scores):
    y, score = scores
    assert scratch.roc_auc(y, score) == pytest.approx(metrics.roc_auc_score(y, score), abs=1e-9)


def test_cross_entropy_matches_sklearn(scores):
    y, score = scores
    p = np.clip(score, 0.01, 0.99)
    assert scratch.binary_cross_entropy(y, p) == pytest.approx(metrics.log_loss(y, p), abs=1e-9)


def test_scalers_match_sklearn(regression):
    X, _ = regression
    assert np.allclose(scratch.zscore_transform(X, *scratch.zscore_fit(X)), StandardScaler().fit_transform(X))
    assert np.allclose(scratch.minmax_transform(X, *scratch.minmax_fit(X)), MinMaxScaler().fit_transform(X))


def test_constant_column_does_not_divide_by_zero():
    X = np.column_stack([np.ones(5), np.arange(5.0)])
    assert np.isfinite(scratch.zscore_transform(X, *scratch.zscore_fit(X))).all()
    assert np.isfinite(scratch.minmax_transform(X, *scratch.minmax_fit(X))).all()


def test_covariance_and_correlation_match_numpy(regression):
    X, _ = regression
    assert np.allclose(scratch.covariance(X), np.cov(X, rowvar=False))
    assert np.allclose(scratch.correlation(X), np.corrcoef(X, rowvar=False))


def test_vif_is_one_for_uncorrelated_and_large_for_collinear():
    rng = np.random.default_rng(3)
    independent = rng.normal(size=(5000, 3))
    assert np.allclose(scratch.variance_inflation(independent), 1.0, atol=0.02)
    collinear = np.column_stack([independent, independent[:, 0] + rng.normal(scale=0.01, size=5000)])
    assert scratch.variance_inflation(collinear)[0] > 100
