import numpy as np
import pandas as pd
import pytest

from src.evaluate import accident_metrics, accident_threshold, coverage, day_bootstrap, day_folds, top_share_lift
from src.ml_models import (
    ScratchLinearRegression,
    choose_winner,
    make_model,
    predict_scores,
    sample_params,
    subsample_by_day,
    tune_band_cuts,
)
from src.uncertainty import apply_conformal, band_probabilities, conformal_quantile, fit_conformal, predictive_entropy


def test_day_bootstrap_brackets_the_truth_and_is_repeatable():
    rng = np.random.default_rng(0)
    dates = np.repeat([f"d{i}" for i in range(60)], 20)
    values = rng.normal(5.0, 1.0, len(dates))
    interval = day_bootstrap(lambda idx: values[idx].mean(), dates, n_resamples=300)
    assert interval[0] < 5.0 < interval[1]
    assert interval == day_bootstrap(lambda idx: values[idx].mean(), dates, n_resamples=300)


def test_day_bootstrap_is_wider_when_days_differ():
    dates = np.repeat([f"d{i}" for i in range(30)], 10)
    rng = np.random.default_rng(1)
    day_effect = np.repeat(rng.normal(0, 3, 30), 10)
    noise = rng.normal(0, 1, 300)
    narrow = day_bootstrap(lambda idx: noise[idx].mean(), dates, n_resamples=300)
    wide = day_bootstrap(lambda idx: (noise + day_effect)[idx].mean(), dates, n_resamples=300)
    assert wide[1] - wide[0] > narrow[1] - narrow[0]


def test_day_folds_are_expanding_whole_days_and_never_look_back():
    dates = pd.Series(pd.date_range("2025-01-01", periods=60, freq="D").strftime("%Y-%m-%d").repeat(3))
    folds = day_folds(dates, 5)
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert dates.iloc[train_idx].max() < dates.iloc[test_idx].min()
        assert dates.iloc[test_idx].value_counts().eq(3).all()
    assert all(len(a[0]) < len(b[0]) for a, b in zip(folds, folds[1:], strict=False))


def test_sample_params_is_repeatable_and_distinct():
    space = {"a": [1, 2, 3], "b": [4, 5]}
    first = sample_params(space, 4, 0)
    assert first == sample_params(space, 4, 0)
    assert len({tuple(sorted(p.items())) for p in first}) == 4
    assert len(sample_params(space, 100, 0)) == 6


def test_preprocessor_is_fit_on_training_rows_only():
    columns = ["vol_t", "occ_t", "road_code"]
    train = pd.DataFrame({"vol_t": [1.0, 2.0, 3.0], "occ_t": [0.0, 50.0, 100.0], "road_code": [1, 2, 1]})
    model = make_model("linear", "regression", {}, columns, 0).fit(train, [1.0, 2.0, 3.0])
    scaler = model.named_steps["prep"].named_transformers_["zscore"]
    assert scaler.mean_[0] == pytest.approx(2.0)


def test_scratch_regression_learns_a_linear_rule():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3))
    y = 2 + X @ np.array([1.0, -3.0, 0.5])
    model = ScratchLinearRegression(epochs=2000).fit(X, y)
    assert np.allclose(model.predict(X), y, atol=1e-3)


@pytest.mark.parametrize("family", ["linear_scratch", "linear", "tree", "forest", "xgboost"])
def test_every_regressor_family_fits_and_predicts_through_its_real_pipeline(family):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"vol_t": rng.normal(size=300), "occ_t": rng.uniform(0, 100, 300), "road_code": rng.integers(1, 4, 300)})
    y = 2 * frame["vol_t"] + frame["occ_t"] / 50
    params = {"n_estimators": 10} if family in ("forest", "xgboost") else {}
    model = make_model(family, "regression", params, list(frame.columns), 0, scratch_epochs=500).fit(frame, y)
    assert np.corrcoef(model.predict(frame), y)[0, 1] > 0.9


@pytest.mark.parametrize("family", ["logistic", "tree", "forest", "xgboost", "svm_rbf", "svm_linear"])
def test_every_classifier_family_fits_and_scores_through_its_real_pipeline(family):
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"vol_t": rng.normal(size=300), "occ_t": rng.uniform(0, 100, 300), "road_code": rng.integers(1, 4, 300)})
    y = (frame["vol_t"] > 0).astype(int)
    params = {"n_estimators": 10} if family in ("forest", "xgboost") else {}
    model = make_model(family, "accident", params, list(frame.columns), 0).fit(frame, y)
    assert predict_scores(model, frame, list(frame.columns), "accident").shape == (300,)


def test_subsample_covers_every_day():
    ts = pd.date_range("2025-01-01", periods=48 * 10, freq="30min", tz="Asia/Kolkata")
    frame = pd.DataFrame({"road_id": "NL-001", "ts": ts})
    sample = subsample_by_day(frame, 100, 0)
    assert sample["ts"].dt.strftime("%Y-%m-%d").value_counts().eq(10).all()


def test_winner_prefers_a_faster_model_when_statistically_tied():
    rows = [
        {"model": "slow", "val_rmse": 50.0, "cv_scores": [60, 70, 50, 65, 55], "infer_ms_per_1k": 10.0},
        {"model": "fast", "val_rmse": 51.0, "cv_scores": [64, 68, 54, 62, 58], "infer_ms_per_1k": 1.0},
        {"model": "fastest_but_bad", "val_rmse": 60.0, "cv_scores": [70, 80, 60, 75, 65], "infer_ms_per_1k": 0.1},
    ]
    assert choose_winner(rows, "rmse")["model"] == "fast"


def test_a_consistent_gap_is_not_a_tie_even_when_folds_vary_widely():
    # Fold scores swing by 20 for both models, but one is better by exactly 1 in every fold.
    rows = [
        {"model": "better", "val_rmse": 50.0, "cv_scores": [40, 60, 45, 55, 50], "infer_ms_per_1k": 10.0},
        {"model": "faster", "val_rmse": 51.0, "cv_scores": [41, 61, 46, 56, 51], "infer_ms_per_1k": 1.0},
    ]
    assert choose_winner(rows, "rmse")["model"] == "better"


def test_a_small_validation_gap_is_not_a_tie_when_cv_is_clearly_worse():
    # Close on validation, but about 6 worse on average across folds, with a standard error near 2.
    rows = [
        {"model": "forest", "val_rmse": 55.0, "cv_scores": [60, 70, 50, 65, 55], "infer_ms_per_1k": 2.0},
        {"model": "tree", "val_rmse": 55.6, "cv_scores": [64, 79, 54, 71, 62], "infer_ms_per_1k": 0.3},
    ]
    assert choose_winner(rows, "rmse")["model"] == "forest"


def test_a_row_without_fold_scores_can_only_win_outright():
    rows = [{"model": "best", "val_rmse": 50.0, "cv_scores": [1, 2, 3], "infer_ms_per_1k": 9.0}, {"model": "derived", "val_rmse": 50.5, "cv_scores": [], "infer_ms_per_1k": 1.0}]
    assert choose_winner(rows, "rmse")["model"] == "best"
    rows[1]["val_rmse"] = 49.0
    assert choose_winner(rows, "rmse")["model"] == "derived"


def test_winner_reads_fold_scores_back_from_csv_text():
    rows = [
        {"model": "slow", "val_rmse": 50.0, "cv_scores": "[60, 70, 50, 65, 55]", "infer_ms_per_1k": 10.0},
        {"model": "fast", "val_rmse": 51.0, "cv_scores": "[64, 68, 54, 62, 58]", "infer_ms_per_1k": 1.0},
    ]
    assert choose_winner(rows, "rmse")["model"] == "fast"


def test_winner_direction_for_higher_is_better():
    rows = [{"model": "a", "val_roc_auc": 0.6, "cv_scores": [], "infer_ms_per_1k": 1.0}, {"model": "b", "val_roc_auc": 0.7, "cv_scores": [], "infer_ms_per_1k": 5.0}]
    assert choose_winner(rows, "roc_auc")["model"] == "b"


def test_band_cut_search_finds_a_shifted_boundary(config):
    # Predictions run 10% low, so the best cut-offs sit about 10% below the published ones.
    capacity = np.full(4000, 2000.0)
    rng = np.random.default_rng(0)
    actual_ratio = rng.uniform(0, 1.5, 4000)
    y = np.digitize(actual_ratio, config["congestion"]["bands"])
    cuts = tune_band_cuts(actual_ratio * 0.9 * 1000, capacity, y, config)
    assert cuts == pytest.approx([0.45, 0.72, 0.92], abs=0.021)


def test_top_share_lift_measures_concentration():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    assert top_share_lift(y, np.arange(10, 0, -1), share=0.2) == pytest.approx(5.0)
    assert top_share_lift(y, np.arange(1, 11), share=0.2) == 0.0


def test_accident_threshold_meets_the_recall_floor():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.05
    score = y * 0.2 + rng.random(5000)
    threshold = accident_threshold(y, score, 0.6)
    assert accident_metrics(y, score, threshold)["recall"] >= 0.6


def test_conformal_quantile_uses_the_finite_sample_rank():
    assert conformal_quantile(np.arange(1, 11, dtype=float), 0.8) == 9.0
    assert conformal_quantile(np.arange(1, 11, dtype=float), 0.95) == 10.0


def test_conformal_covers_new_data_at_the_nominal_level():
    rng = np.random.default_rng(0)
    y_cal, y_new = rng.normal(size=20000), rng.normal(size=20000)
    p_cal, p_new = np.zeros(20000), np.zeros(20000)
    cells_cal, cells_new = pd.Series(["a"] * 20000), pd.Series(["a"] * 20000)
    table = fit_conformal(y_cal, p_cal, cells_cal, [0.8, 0.95])
    for level in (0.8, 0.95):
        lower, upper = apply_conformal(p_new, cells_new, table, level, floor=-np.inf)
        assert coverage(y_new, lower, upper) == pytest.approx(level, abs=0.015)


def test_conformal_widths_differ_by_cell_and_respect_the_floor():
    y = np.r_[np.zeros(100), np.arange(100.0)]
    cells = pd.Series(["narrow"] * 100 + ["wide"] * 100)
    table = fit_conformal(y, np.zeros(200), cells, [0.8]).set_index("cell")
    assert table.loc["wide", "q80"] > table.loc["narrow", "q80"]
    lower, upper = apply_conformal(np.array([5.0]), pd.Series(["wide"]), table.reset_index(), 0.8)
    assert lower[0] == 0.0
    assert upper[0] > 5.0


def test_unseen_cell_takes_the_widest_band():
    table = pd.DataFrame({"cell": ["a", "b"], "q80": [1.0, 3.0]})
    _, upper = apply_conformal(np.zeros(1), pd.Series(["new"]), table, 0.8)
    assert upper[0] == 3.0


def test_band_probabilities_sum_to_one_and_respond_to_the_forecast():
    residuals = np.random.default_rng(0).normal(scale=50, size=5000)
    capacity = np.array([2000.0, 2000.0])
    probabilities = band_probabilities(np.array([100.0, 1200.0]), capacity, residuals, [0.5, 0.8, 1.0])
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert probabilities[0].argmax() == 0
    assert probabilities[1].argmax() == 3


def test_entropy_is_zero_for_certainty_and_two_bits_for_four_equal_classes():
    assert predictive_entropy(np.array([[1.0, 0, 0, 0]]))[0] == pytest.approx(0.0, abs=1e-6)
    assert predictive_entropy(np.array([[0.25] * 4]))[0] == pytest.approx(2.0)
