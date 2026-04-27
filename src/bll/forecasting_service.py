from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyswarms as ps
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import tensorflow as tf

from .dataset import create_multivariate_dataset, prepare_multivariate_features, split_windows_by_index
from .model_factory import build_stacked_lstm_model
from dal.gold_market_data import GoldMarketDataProvider
from dto.configs import ExperimentConfig
from dto.results import LSTMHyperParameters, RegressionMetrics
from gui.visualizer import plot_correlation_heatmap, plot_prediction_comparison
from util.metrics import evaluate_regression
from util.persistence import save_json, save_joblib_object, save_keras_model


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    cost: float
    params: LSTMHyperParameters
    raw_position: np.ndarray


class PSOLSTMExperiment:
    def __init__(
        self,
        config: ExperimentConfig | None = None,
        data_provider: GoldMarketDataProvider | None = None,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.data_provider = data_provider or GoldMarketDataProvider()
        self.raw_df: pd.DataFrame | None = None
        self.prepared_df: pd.DataFrame | None = None
        self.scaled_data: np.ndarray | None = None
        self.scaler_gold = None
        self.feature_scaler = None
        self.feature_columns: list[str] = []
        self.split_index: int | None = None
        self.num_features: int | None = None

    def load_and_prepare_data(self) -> pd.DataFrame:
        self.raw_df = self.data_provider.fetch_benchmark(self.config.data.start_date)
        return self._prepare_filtered_data(self.raw_df)

    def plot_correlation(self) -> None:
        if self.prepared_df is None:
            raise RuntimeError("Dữ liệu chưa được tải.")

        corr_df = self.prepared_df.drop(columns=["Date"]).corr(numeric_only=True)
        plot_correlation_heatmap(corr_df)

    def optimize_hyperparameters(
        self,
        progress_callback: Callable[[int, int, float | None], None] | None = None,
    ) -> OptimizationResult:
        self._ensure_prepared()
        total_iterations = max(1, int(self.config.pso.iterations))
        optimizer = ps.single.GlobalBestPSO(
            n_particles=self.config.pso.n_particles,
            dimensions=self.config.pso.dimensions,
            options=self.config.pso.options,
            bounds=self.config.pso.bounds,
        )

        current_iteration = 0

        def objective_with_progress(particles: np.ndarray) -> np.ndarray:
            nonlocal current_iteration
            costs = self._objective_function(particles)
            current_iteration += 1
            if progress_callback is not None:
                best_cost = float(np.min(costs)) if len(costs) > 0 else None
                progress_callback(current_iteration, total_iterations, best_cost)
            return costs

        cost, raw_position = optimizer.optimize(
            objective_with_progress,
            iters=total_iterations,
            verbose=False,
        )
        if progress_callback is not None:
            progress_callback(total_iterations, total_iterations, float(cost))

        params = self._position_to_params(raw_position)
        return OptimizationResult(cost=float(cost), params=params, raw_position=np.asarray(raw_position))

    def get_window_stats(self, look_back: int) -> dict[str, int | bool]:
        self._ensure_prepared()
        if self.prepared_df is None or self.split_index is None:
            raise RuntimeError("Dữ liệu chưa được chuẩn bị.")
        if look_back <= 0:
            raise ValueError("look_back phải lớn hơn 0.")

        total_rows = len(self.prepared_df)
        total_windows = max(0, total_rows - look_back)
        train_windows = min(max(self.split_index - look_back, 0), total_windows)
        test_windows = max(0, total_windows - train_windows)

        return {
            "look_back": int(look_back),
            "total_rows": int(total_rows),
            "split_index": int(self.split_index),
            "total_windows": int(total_windows),
            "train_windows": int(train_windows),
            "test_windows": int(test_windows),
            "is_valid": bool(total_windows > 0 and train_windows > 0 and test_windows > 0),
        }

    def validate_split_and_look_back(self, look_back: int) -> tuple[bool, str]:
        stats = self.get_window_stats(look_back)
        if stats["total_windows"] <= 0:
            return (
                False,
                "Không tạo được window dữ liệu (total_windows=0). Hãy giảm look_back hoặc chọn ngày bắt đầu sớm hơn.",
            )
        if stats["train_windows"] <= 0:
            return (
                False,
                (
                    "Không có mẫu train (train_windows=0). "
                    "Hãy đặt split_date muộn hơn hoặc giảm look_back."
                ),
            )
        if stats["test_windows"] <= 0:
            return (
                False,
                (
                    "Không có mẫu test (test_windows=0). "
                    "Hãy đặt split_date sớm hơn hoặc chọn ngày bắt đầu sớm hơn."
                ),
            )

        return (
            True,
            (
                f"Hợp lệ: train_windows={stats['train_windows']}, "
                f"test_windows={stats['test_windows']}, total_windows={stats['total_windows']}."
            ),
        )

    def train_final_model(
        self,
        params: LSTMHyperParameters,
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
    ):
        model_result = self._train_model(
            params=params,
            epochs=self.config.final_training_epochs,
            validation_split=0.15,
            patience=self.config.baseline.patience,
            verbose=1,
            callbacks=callbacks,
        )
        return model_result

    def fine_tune_model(
        self,
        params: LSTMHyperParameters,
        base_model,
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
    ):
        return self._train_model(
            params=params,
            epochs=self.config.fine_tune.epochs,
            validation_split=self.config.fine_tune.validation_split,
            patience=self.config.fine_tune.patience,
            verbose=1,
            callbacks=callbacks,
            initial_model=base_model,
            learning_rate=params.learning_rate * self.config.fine_tune.learning_rate_factor,
        )

    def train_baseline_model(self, callbacks: list[tf.keras.callbacks.Callback] | None = None):
        baseline_params = LSTMHyperParameters(
            look_back=self.config.baseline.look_back,
            hidden_units=self.config.baseline.hidden_units,
            learning_rate=self.config.baseline.learning_rate,
            batch_size=self.config.baseline.batch_size,
            num_layers=self.config.baseline.num_layers,
            dropout_rate=self.config.baseline.dropout_rate,
        )
        return self._train_model(
            params=baseline_params,
            epochs=self.config.baseline.epochs,
            validation_split=self.config.baseline.validation_split,
            patience=self.config.baseline.patience,
            verbose=1,
            callbacks=callbacks,
        )

    def save_artifacts(
        self,
        model,
        best_result: OptimizationResult | None,
        metrics: RegressionMetrics,
    ) -> None:
        artifact_dir = self.config.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

        save_keras_model(artifact_dir / "model_PSO_LSTM_final.h5", model)
        save_joblib_object(artifact_dir / "scaler_gold.pkl", self.scaler_gold)
        if best_result is not None:
            save_json(
                artifact_dir / "best_params.json",
                {
                    **asdict(best_result.params),
                    "cost": best_result.cost,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
        save_json(
            artifact_dir / "model_metrics.json",
            {
                **metrics.to_dict(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    def run(self) -> dict[str, object]:
        self.load_and_prepare_data()
        self.plot_correlation()

        optimization_result = self.optimize_hyperparameters()
        final_result = self.train_final_model(optimization_result.params)
        baseline_result = self.train_baseline_model()

        self.save_artifacts(final_result["model"], optimization_result, final_result["metrics"])

        plot_prediction_comparison(
            final_result["y_true"],
            final_result["predictions"],
            title="PSO-LSTM Multivariate - Dự đoán giá vàng trên Benchmark",
            true_label="Giá vàng thực tế (Test set)",
            prediction_label="Dự đoán PSO-LSTM",
        )

        return {
            "optimization": optimization_result,
            "final": final_result,
            "baseline": baseline_result,
        }

    def load_data_for_app(self, refresh: bool = False) -> pd.DataFrame:
        csv_path = Path(__file__).resolve().parents[1] / "gia_vang_benchmark.csv"
        if csv_path.exists() and not refresh:
            self.raw_df = pd.read_csv(csv_path)
            return self._prepare_filtered_data(self.raw_df)

        return self.load_and_prepare_data()

    def forecast_next_days(
        self,
        model,
        params: LSTMHyperParameters,
        days: int,
    ) -> pd.DataFrame:
        self._ensure_prepared()
        if days <= 0:
            raise ValueError("days phải lớn hơn 0.")

        if self.scaled_data is None or self.prepared_df is None:
            raise RuntimeError("Dữ liệu chưa được chuẩn bị.")

        history = self.scaled_data.copy()
        last_date = pd.to_datetime(self.prepared_df["Date"]).iloc[-1]
        future_rows: list[dict[str, object]] = []
        last_known_features = history[-1, :].copy()

        for step in range(days):
            x_input = history[-params.look_back :, :].reshape(1, params.look_back, self.num_features)
            predicted_scaled = model.predict(x_input, verbose=0).reshape(-1)[0]

            next_row = last_known_features.copy()
            next_row[0] = predicted_scaled
            history = np.vstack([history, next_row])
            last_known_features = next_row

            future_rows.append(
                {
                    "Date": (last_date + pd.offsets.BDay(step + 1)).date(),
                    "Predicted_Gold_Close": float(self.scaler_gold.inverse_transform([[predicted_scaled]])[0, 0]),
                }
            )

        return pd.DataFrame(future_rows)

    def _objective_function(self, particles: np.ndarray) -> np.ndarray:
        self._ensure_prepared()
        fitness_values: list[float] = []
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.config.pso.objective_patience,
            restore_best_weights=True,
        )

        for particle in particles:
            params = self._position_to_params(particle)
            try:
                x_values, y_values = create_multivariate_dataset(self.scaled_data, params.look_back)
                x_train_full, y_train_full, _, _ = split_windows_by_index(
                    x_values,
                    y_values,
                    self.split_index,
                    params.look_back,
                )

                if len(x_train_full) < 2:
                    fitness_values.append(1e6)
                    continue

                validation_size = max(1, int(len(x_train_full) * (1 - self.config.pso.train_eval_ratio)))
                x_train = x_train_full[:-validation_size]
                y_train = y_train_full[:-validation_size]
                x_val = x_train_full[-validation_size:]
                y_val = y_train_full[-validation_size:]

                if len(x_train) == 0 or len(x_val) == 0:
                    fitness_values.append(1e6)
                    continue

                model = build_stacked_lstm_model(
                    input_shape=(params.look_back, self.num_features),
                    hidden_units=params.hidden_units,
                    num_layers=params.num_layers,
                    dropout_rate=params.dropout_rate,
                    learning_rate=params.learning_rate,
                )

                model.fit(
                    x_train,
                    y_train,
                    epochs=self.config.pso.objective_epochs,
                    batch_size=params.batch_size,
                    validation_data=(x_val, y_val),
                    callbacks=[early_stopping],
                    verbose=0,
                    shuffle=False,
                )

                predictions_scaled = model.predict(x_val, verbose=0)
                y_true = self.scaler_gold.inverse_transform(y_val.reshape(-1, 1)).flatten()
                y_pred = self.scaler_gold.inverse_transform(predictions_scaled).flatten()

                rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                mape = float(mean_absolute_percentage_error(y_true, y_pred))
                fitness_values.append(rmse * (1 + 0.6 * mape))
            except Exception:
                fitness_values.append(1e6)

        return np.asarray(fitness_values)

    def _train_model(
        self,
        params: LSTMHyperParameters,
        epochs: int,
        validation_split: float,
        patience: int,
        verbose: int,
        callbacks: list[tf.keras.callbacks.Callback] | None = None,
        initial_model=None,
        learning_rate: float | None = None,
    ) -> dict[str, object]:
        self._ensure_prepared()
        x_values, y_values = create_multivariate_dataset(self.scaled_data, params.look_back)
        if len(x_values) == 0:
            raise ValueError(
                "Không đủ dữ liệu để tạo chuỗi huấn luyện. Hãy giảm look_back hoặc mở rộng khoảng thời gian dữ liệu."
            )

        x_train, y_train, x_test, y_test = split_windows_by_index(
            x_values,
            y_values,
            self.split_index,
            params.look_back,
        )

        if len(x_train) == 0:
            raise ValueError(
                "Không có mẫu train sau khi tách dữ liệu. Hãy điều chỉnh split_date hoặc giảm look_back."
            )
        if len(x_test) == 0:
            raise ValueError(
                "Không có mẫu test sau khi tách dữ liệu. Hãy điều chỉnh split_date để còn dữ liệu đánh giá."
            )

        validation_split = float(np.clip(validation_split, 0.0, 0.99))
        validation_samples = int(len(x_train) * validation_split)
        effective_validation_split = validation_split if validation_samples > 0 else 0.0
        monitor_metric = "val_loss" if effective_validation_split > 0 else "loss"
        effective_batch_size = min(int(params.batch_size), len(x_train))

        effective_learning_rate = learning_rate if learning_rate is not None else params.learning_rate
        if initial_model is None:
            model = build_stacked_lstm_model(
                input_shape=(params.look_back, self.num_features),
                hidden_units=params.hidden_units,
                num_layers=params.num_layers,
                dropout_rate=params.dropout_rate,
                learning_rate=effective_learning_rate,
            )
        else:
            model = initial_model
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=effective_learning_rate), loss="mse")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
        )
        fit_callbacks = [early_stopping, *(callbacks or [])]
        model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=effective_batch_size,
            validation_split=effective_validation_split,
            callbacks=fit_callbacks,
            verbose=verbose,
            shuffle=False,
        )

        predictions_scaled = model.predict(x_test, verbose=0)
        y_true = self.scaler_gold.inverse_transform(y_test.reshape(-1, 1)).flatten()
        predictions = self.scaler_gold.inverse_transform(predictions_scaled).flatten()
        metrics = evaluate_regression(y_true, predictions)
        test_dates = pd.to_datetime(self.prepared_df["Date"]).iloc[self.split_index : self.split_index + len(y_test)].to_list()

        return {
            "model": model,
            "metrics": metrics,
            "y_true": y_true,
            "predictions": predictions,
            "test_dates": test_dates,
            "train_size": len(x_train),
            "test_size": len(x_test),
        }

    def _position_to_params(self, position: np.ndarray) -> LSTMHyperParameters:
        values = np.asarray(position, dtype=float)
        return LSTMHyperParameters(
            look_back=int(np.clip(values[0], 30, 120)),
            hidden_units=int(np.clip(values[1], 80, 280)),
            learning_rate=float(np.clip(values[2], 0.0005, 0.008)),
            batch_size=int(np.clip(values[3], 16, 256)),
            num_layers=int(np.clip(values[4], 1, 3)),
            dropout_rate=float(np.clip(values[5], 0.05, 0.45)),
        )

    def _resolve_split_index(self, df: pd.DataFrame, split_date: str) -> int:
        split_timestamp = pd.to_datetime(split_date)
        split_matches = np.flatnonzero((df["Date"] >= split_timestamp).to_numpy())
        if len(split_matches) == 0:
            raise ValueError(f"split_date {split_date} nằm ngoài phạm vi dữ liệu hiện có.")
        return int(split_matches[0])

    def _prepare_filtered_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        prepared_df = raw_df.copy()
        self._normalize_date_column(prepared_df)

        min_allowed_start = pd.Timestamp("2005-01-01")
        requested_start = pd.to_datetime(self.config.data.start_date)
        if requested_start < min_allowed_start:
            raise ValueError("Ngày bắt đầu dữ liệu không được trước 2005-01-01.")

        prepared_df = prepared_df.loc[prepared_df["Date"] >= requested_start].copy()
        if prepared_df.empty:
            raise ValueError("Không có dữ liệu sau ngày bắt đầu được chọn.")

        prepared_df = prepared_df.sort_values("Date").reset_index(drop=True)

        split_timestamp = pd.to_datetime(self.config.data.split_date)
        min_date = prepared_df["Date"].min()
        max_date = prepared_df["Date"].max()
        if split_timestamp < min_date or split_timestamp > max_date:
            raise ValueError("Ngày chia train/test phải nằm trong phạm vi dữ liệu đã lọc.")

        self.prepared_df = prepared_df
        self.scaled_data, self.scaler_gold, self.feature_scaler, self.feature_columns = (
            prepare_multivariate_features(self.prepared_df)
        )
        self.num_features = int(self.scaled_data.shape[1])
        self.split_index = self._resolve_split_index(self.prepared_df, self.config.data.split_date)
        return self.prepared_df

    @staticmethod
    def _normalize_date_column(df: pd.DataFrame) -> None:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    def _ensure_prepared(self) -> None:
        if self.scaled_data is None or self.split_index is None or self.num_features is None:
            raise RuntimeError("Dữ liệu chưa được chuẩn bị. Hãy gọi load_and_prepare_data() trước.")