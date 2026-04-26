import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from dto.results import RegressionMetrics


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_diff = np.diff(np.asarray(y_true).reshape(-1)) > 0
    pred_diff = np.diff(np.asarray(y_pred).reshape(-1)) > 0
    if len(true_diff) == 0:
        return 0.0
    return float(np.mean(true_diff == pred_diff) * 100)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    y_true_array = np.asarray(y_true).reshape(-1)
    y_pred_array = np.asarray(y_pred).reshape(-1)

    mae = float(mean_absolute_error(y_true_array, y_pred_array))
    rmse = float(np.sqrt(mean_squared_error(y_true_array, y_pred_array)))
    mape = float(mean_absolute_percentage_error(y_true_array, y_pred_array) * 100)
    r2 = float(r2_score(y_true_array, y_pred_array))
    directional_accuracy = calculate_directional_accuracy(y_true_array, y_pred_array)

    return RegressionMetrics(
        mae=mae,
        rmse=rmse,
        mape=mape,
        r2=r2,
        directional_accuracy=directional_accuracy,
    )