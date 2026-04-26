import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_multivariate_features(df: pd.DataFrame) -> tuple[np.ndarray, MinMaxScaler, MinMaxScaler, list[str]]:
    prepared = df.copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"])
    prepared = prepared.sort_values("Date").reset_index(drop=True)

    gold_scaler = MinMaxScaler(feature_range=(0, 1))
    gold_scaler.fit(prepared[["Gold_Close"]])

    feature_columns = [column for column in prepared.columns if column != "Date"]
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = feature_scaler.fit_transform(prepared[feature_columns])

    return scaled_data, gold_scaler, feature_scaler, feature_columns


def create_multivariate_dataset(data: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    if look_back <= 0:
        raise ValueError("look_back phải lớn hơn 0.")

    x_values: list[np.ndarray] = []
    y_values: list[float] = []

    for index in range(len(data) - look_back):
        x_values.append(data[index : index + look_back, :])
        y_values.append(data[index + look_back, 0])

    return np.array(x_values), np.array(y_values)


def split_windows_by_index(
    x_values: np.ndarray,
    y_values: np.ndarray,
    split_index: int,
    look_back: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_end = split_index - look_back
    if train_end <= 0:
        raise ValueError("Không đủ dữ liệu để tách train/test với look_back hiện tại.")

    return (
        x_values[:train_end],
        y_values[:train_end],
        x_values[train_end:],
        y_values[train_end:],
    )