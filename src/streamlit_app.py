from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from bll.dataset import create_multivariate_dataset
from bll.forecasting_service import OptimizationResult, PSOLSTMExperiment
from dto.results import LSTMHyperParameters


APP_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = APP_ROOT / "artifacts"
CSV_PATH = APP_ROOT / "src" / "gia_vang_benchmark.csv"
MODEL_PATH = ARTIFACTS_DIR / "model_PSO_LSTM_final.h5"
BEST_PARAMS_PATH = ARTIFACTS_DIR / "best_params.json"
METRICS_PATH = ARTIFACTS_DIR / "model_metrics.json"


st.set_page_config(
    page_title="Gold Forecast Studio",
    page_icon="\U0001F3C6",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(223, 181, 93, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(28, 76, 99, 0.14), transparent 24%),
                linear-gradient(180deg, #f8f4ec 0%, #f3efe6 45%, #eef2f6 100%);
            color: #13212b;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            padding: 1.2rem 1.4rem;
            border-radius: 22px;
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(19, 33, 43, 0.08);
            box-shadow: 0 16px 50px rgba(19, 33, 43, 0.08);
            backdrop-filter: blur(12px);
        }
        .subtle-note {
            color: #4c5b66;
            font-size: 0.95rem;
        }
        .section-title {
            margin-top: 1.4rem;
            margin-bottom: 0.3rem;
            font-size: 1.25rem;
            font-weight: 700;
            color: #182734;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(19, 33, 43, 0.08);
            border-radius: 18px;
            padding: 0.6rem 0.8rem;
            box-shadow: 0 10px 30px rgba(19, 33, 43, 0.05);
        }
        .stButton>button {
            border-radius: 999px;
            padding: 0.65rem 1.1rem;
            border: 0;
            background: linear-gradient(90deg, #1f556e 0%, #d39f3f 100%);
            color: white;
            font-weight: 700;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 20px rgba(19, 33, 43, 0.15);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_experiment() -> PSOLSTMExperiment:
    return PSOLSTMExperiment()


@st.cache_data(show_spinner=False)
def load_market_data(start_date: str, split_date: str, refresh: bool = False) -> pd.DataFrame:
    experiment = get_experiment()
    experiment.config = type(experiment.config)(
        data=type(experiment.config.data)(start_date=start_date, split_date=split_date),
        pso=experiment.config.pso,
        baseline=experiment.config.baseline,
        final_training_epochs=experiment.config.final_training_epochs,
        artifact_dir=experiment.config.artifact_dir,
    )
    return experiment.load_data_for_app(refresh=refresh)


def clear_cached_data() -> None:
    load_market_data.clear()


def load_saved_model() -> tuple[tf.keras.Model, LSTMHyperParameters] | None:
    params = load_saved_params()
    if params is None or not MODEL_PATH.exists():
        return None
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model, params


def load_saved_params() -> LSTMHyperParameters | None:
    if not BEST_PARAMS_PATH.exists():
        return None

    with BEST_PARAMS_PATH.open("r", encoding="utf-8") as file_handle:
        raw_params = json.load(file_handle)

    return LSTMHyperParameters(
        look_back=int(raw_params["look_back"]),
        hidden_units=int(raw_params["hidden_units"]),
        learning_rate=float(raw_params["learning_rate"]),
        batch_size=int(raw_params["batch_size"]),
        num_layers=int(raw_params["num_layers"]),
        dropout_rate=float(raw_params["dropout_rate"]),
    )


class EpochProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, status_box, label: str, total_epochs: int) -> None:
        super().__init__()
        self.progress_bar = progress_bar
        self.status_box = status_box
        self.label = label
        self.total_epochs = max(1, total_epochs)

    def on_train_begin(self, logs=None):
        self.progress_bar.progress(0, text=f"{self.label}: bắt đầu")
        self.status_box.info(f"{self.label}: đang khởi tạo huấn luyện...")

    def on_epoch_end(self, epoch, logs=None):
        progress_value = int(((epoch + 1) / self.total_epochs) * 100)
        metrics_text = []
        if logs:
            loss = logs.get("loss")
            val_loss = logs.get("val_loss")
            if loss is not None:
                metrics_text.append(f"loss={loss:.6f}")
            if val_loss is not None:
                metrics_text.append(f"val_loss={val_loss:.6f}")
        detail_text = " | ".join(metrics_text)
        self.progress_bar.progress(progress_value, text=f"{self.label}: epoch {epoch + 1}/{self.total_epochs}")
        if detail_text:
            self.status_box.info(f"{self.label}: epoch {epoch + 1}/{self.total_epochs} - {detail_text}")
        else:
            self.status_box.info(f"{self.label}: epoch {epoch + 1}/{self.total_epochs}")

    def on_train_end(self, logs=None):
        self.progress_bar.progress(100, text=f"{self.label}: hoàn tất")
        self.status_box.success(f"{self.label}: hoàn tất huấn luyện.")


def build_correlation_chart(corr_df: pd.DataFrame) -> go.Figure:
    labels = list(corr_df.columns)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_df.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            text=corr_df.round(2).astype(str).values,
            texttemplate="%{text}",
            hovertemplate="%{y} x %{x}<br>corr=%{z:.2f}<extra></extra>",
            colorbar=dict(title="Corr"),
        )
    )
    fig.update_layout(
        height=650,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
        title="Ma trận tương quan giữa các biến",
    )
    return fig


def build_prediction_chart(
    history_df: pd.DataFrame,
    pso_forecast_df: pd.DataFrame,
    baseline_forecast_df: pd.DataFrame | None = None,
    pso_in_sample_df: pd.DataFrame | None = None,
    baseline_in_sample_df: pd.DataFrame | None = None,
    split_date: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["Date"],
            y=history_df["Gold_Close"],
            name="Lịch sử giá vàng",
            line=dict(color="#1f556e", width=2.5),
        )
    )
    if pso_in_sample_df is not None and not pso_in_sample_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pso_in_sample_df["Date"],
                y=pso_in_sample_df["Predicted_Gold_Close"],
                name="Đường dự báo PSO-LSTM",
                line=dict(color="#b87d2c", width=2),
                opacity=0.85,
            )
        )
    fig.add_trace(
        go.Scatter(
            x=pso_forecast_df["Date"],
            y=pso_forecast_df["Predicted_Gold_Close"],
            name="Dự báo PSO-LSTM (2 tháng)",
            line=dict(color="#d39f3f", width=3, dash="dash"),
        )
    )
    if baseline_in_sample_df is not None and not baseline_in_sample_df.empty:
        fig.add_trace(
            go.Scatter(
                x=baseline_in_sample_df["Date"],
                y=baseline_in_sample_df["Predicted_Gold_Close"],
                name="Đường dự báo LSTM Baseline",
                line=dict(color="#1f7346", width=1.8, dash="dot"),
                opacity=0.85,
            )
        )
    if baseline_forecast_df is not None:
        fig.add_trace(
            go.Scatter(
                x=baseline_forecast_df["Date"],
                y=baseline_forecast_df["Predicted_Gold_Close"],
                name="Dự báo LSTM Baseline (2 tháng)",
                line=dict(color="#2a8f5b", width=2.5, dash="dot"),
            )
        )
    forecast_start = pso_forecast_df["Date"].iloc[0]
    fig.add_vline(x=forecast_start, line_width=2, line_dash="dot", line_color="#6d7a84")
    if split_date is not None:
        fig.add_vline(
            x=pd.to_datetime(split_date),
            line_width=1.5,
            line_dash="dash",
            line_color="#93a1ab",
        )
    fig.update_layout(
        height=580,
        template="plotly_white",
        title="Giá vàng thực tế và dự báo 2 tháng tới",
        xaxis_title="Ngày",
        yaxis_title="Giá vàng (USD)",
        legend_title_text="Chú thích",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def build_test_comparison_chart(final_result: dict[str, object], baseline_result: dict[str, object] | None) -> go.Figure:
    chart_df = pd.DataFrame(
        {
            "Giá vàng thực tế": pd.Series(final_result["y_true"]).astype(float),
            "PSO-LSTM (đề xuất)": pd.Series(final_result["predictions"]).astype(float),
        }
    )

    if baseline_result is not None:
        baseline_values = pd.Series(baseline_result["predictions"]).astype(float)
        min_len = min(len(chart_df), len(baseline_values))
        chart_df = chart_df.iloc[:min_len].copy()
        chart_df["LSTM Baseline"] = baseline_values.iloc[:min_len].to_numpy()

    chart_df = chart_df.reset_index(drop=True)
    chart_df["Test_Index"] = chart_df.index

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["Test_Index"],
            y=chart_df["Giá vàng thực tế"],
            name="Giá vàng thực tế",
            line=dict(color="#0000FF", width=2.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chart_df["Test_Index"],
            y=chart_df["PSO-LSTM (đề xuất)"],
            name="PSO-LSTM (đề xuất)",
            line=dict(color="#FF0000", width=2.7, dash="dash"),
        )
    )
    if baseline_result is not None and "LSTM Baseline" in chart_df.columns:
        fig.add_trace(
            go.Scatter(
                x=chart_df["Test_Index"],
                y=chart_df["LSTM Baseline"],
                name="LSTM Baseline",
                line=dict(color="#1E9E2E", width=2.5, dash="dot"),
            )
        )

    fig.update_layout(
        height=580,
        template="plotly_white",
        title="So sánh PSO-LSTM và LSTM Baseline trên tập test",
        xaxis_title="Thời gian (ngày trong tập test)",
        yaxis_title="Giá vàng (USD)",
        legend_title_text="Chú thích",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#d9d9d9")
    fig.update_yaxes(showgrid=True, gridcolor="#d9d9d9")
    return fig


def get_baseline_params(experiment: PSOLSTMExperiment) -> LSTMHyperParameters:
    cfg = experiment.config.baseline
    return LSTMHyperParameters(
        look_back=cfg.look_back,
        hidden_units=cfg.hidden_units,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        num_layers=cfg.num_layers,
        dropout_rate=cfg.dropout_rate,
    )


def build_full_history_prediction_df(
    experiment: PSOLSTMExperiment,
    model,
    params: LSTMHyperParameters,
) -> pd.DataFrame:
    experiment._ensure_prepared()
    if experiment.scaled_data is None or experiment.prepared_df is None or experiment.scaler_gold is None:
        return pd.DataFrame(columns=["Date", "Predicted_Gold_Close"])

    all_dates = pd.to_datetime(experiment.prepared_df["Date"]).reset_index(drop=True)
    if len(all_dates) <= params.look_back:
        return pd.DataFrame(columns=["Date", "Predicted_Gold_Close"])

    x_values, _ = create_multivariate_dataset(experiment.scaled_data, params.look_back)
    if len(x_values) == 0:
        return pd.DataFrame(columns=["Date", "Predicted_Gold_Close"])

    predictions_scaled = model.predict(x_values, verbose=0)
    predictions = experiment.scaler_gold.inverse_transform(predictions_scaled).flatten()

    full_predictions = pd.Series([float("nan")] * len(all_dates), dtype="float64")
    full_predictions.iloc[params.look_back : params.look_back + len(predictions)] = predictions

    return pd.DataFrame(
        {
            "Date": all_dates,
            "Predicted_Gold_Close": full_predictions,
        }
    )


def render_future_forecast_preview(
    experiment: PSOLSTMExperiment,
    history_df: pd.DataFrame,
    forecast_horizon: int,
    chart_placeholder,
    pso_model,
    pso_params: LSTMHyperParameters,
    pso_test_df: pd.DataFrame | None = None,
    baseline_model=None,
    baseline_params: LSTMHyperParameters | None = None,
    baseline_test_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame | None]:
    pso_future_df = experiment.forecast_next_days(pso_model, pso_params, forecast_horizon)

    baseline_future_df = None
    if baseline_model is not None and baseline_params is not None:
        baseline_future_df = experiment.forecast_next_days(baseline_model, baseline_params, forecast_horizon)

    chart_placeholder.plotly_chart(
        build_prediction_chart(
            history_df,
            pso_future_df,
            baseline_future_df,
            pso_test_df,
            baseline_test_df,
            split_date=experiment.config.data.split_date,
        ),
        use_container_width=True,
        key='forecast_preview_chart',
    )
    return {
        'pso_future': pso_future_df,
        'baseline_future': baseline_future_df,
        'pso_in_sample': pso_test_df,
        'baseline_in_sample': baseline_test_df,
    }


def train_or_load_model(
    experiment: PSOLSTMExperiment,
    run_pso: bool,
    fine_tune_from_saved: bool,
    progress_bar,
    status_box,
    history_df: pd.DataFrame,
    forecast_horizon: int,
    forecast_chart_placeholder,
) -> dict[str, object]:
    def pso_progress(iteration: int, total_iterations: int, best_cost: float | None) -> None:
        safe_total = max(1, total_iterations)
        safe_iteration = min(max(1, iteration), safe_total)
        progress_value = int((safe_iteration / safe_total) * 100)
        progress_bar.progress(
            progress_value,
            text=f"PSO GlobalBest: vong lap {safe_iteration}/{safe_total}",
        )
        if best_cost is None:
            status_box.info(f"PSO GlobalBest: vong lap {safe_iteration}/{safe_total}")
        else:
            status_box.info(
                f"PSO GlobalBest: vong lap {safe_iteration}/{safe_total} | best_cost={best_cost:.6f}"
            )

    def train_baseline_optional() -> object | None:
        try:
            return experiment.train_baseline_model(
                callbacks=[
                    EpochProgressCallback(
                        progress_bar,
                        status_box,
                        'Hu?n luy?n baseline',
                        experiment.config.baseline.epochs,
                    )
                ]
            )
        except Exception as baseline_error:
            status_box.warning(f"Huấn luyện baseline không thành công: {baseline_error}")
            return None

    if run_pso or not MODEL_PATH.exists():
        optimization_result: OptimizationResult = experiment.optimize_hyperparameters(progress_callback=pso_progress)
        final_result = experiment.train_final_model(
            optimization_result.params,
            callbacks=[
                EpochProgressCallback(
                    progress_bar,
                    status_box,
                    'Hu?n luy?n model m?i',
                    experiment.config.final_training_epochs,
                )
            ],
        )
        preview_payload = render_future_forecast_preview(
            experiment=experiment,
            history_df=history_df,
            forecast_horizon=forecast_horizon,
            chart_placeholder=forecast_chart_placeholder,
            pso_model=final_result['model'],
            pso_params=optimization_result.params,
            pso_test_df=build_full_history_prediction_df(
                experiment,
                final_result['model'],
                optimization_result.params,
            ),
        )
        status_box.info("Đã cập nhật đường dự báo PSO-LSTM. Tiếp tục huấn luyện LSTM Baseline...")
        experiment.save_artifacts(final_result['model'], optimization_result, final_result['metrics'])
        baseline_result = train_baseline_optional()
        baseline_params = get_baseline_params(experiment) if baseline_result is not None else None
        if baseline_result is not None:
            preview_payload = render_future_forecast_preview(
                experiment=experiment,
                history_df=history_df,
                forecast_horizon=forecast_horizon,
                chart_placeholder=forecast_chart_placeholder,
                pso_model=final_result['model'],
                pso_params=optimization_result.params,
                pso_test_df=build_full_history_prediction_df(
                    experiment,
                    final_result['model'],
                    optimization_result.params,
                ),
                baseline_model=baseline_result['model'],
                baseline_params=baseline_params,
                baseline_test_df=build_full_history_prediction_df(
                    experiment,
                    baseline_result['model'],
                    baseline_params,
                ),
            )
        return {
            'model': final_result['model'],
            'params': optimization_result.params,
            'optimization': optimization_result,
            'final': final_result,
            'baseline': baseline_result,
            'baseline_params': baseline_params,
            'pso_future': preview_payload['pso_future'],
            'baseline_future': preview_payload['baseline_future'],
            'pso_in_sample': preview_payload['pso_in_sample'],
            'baseline_in_sample': preview_payload['baseline_in_sample'],
            'source': 'trained',
        }

    loaded = load_saved_model()
    if loaded is None:
        optimization_result = experiment.optimize_hyperparameters(progress_callback=pso_progress)
        final_result = experiment.train_final_model(
            optimization_result.params,
            callbacks=[
                EpochProgressCallback(
                    progress_bar,
                    status_box,
                    'Hu?n luy?n model m?i',
                    experiment.config.final_training_epochs,
                )
            ],
        )
        preview_payload = render_future_forecast_preview(
            experiment=experiment,
            history_df=history_df,
            forecast_horizon=forecast_horizon,
            chart_placeholder=forecast_chart_placeholder,
            pso_model=final_result['model'],
            pso_params=optimization_result.params,
            pso_test_df=build_full_history_prediction_df(
                experiment,
                final_result['model'],
                optimization_result.params,
            ),
        )
        status_box.info("Đã cập nhật đường dự báo PSO-LSTM. Tiếp tục huấn luyện LSTM Baseline...")
        experiment.save_artifacts(final_result['model'], optimization_result, final_result['metrics'])
        baseline_result = train_baseline_optional()
        baseline_params = get_baseline_params(experiment) if baseline_result is not None else None
        if baseline_result is not None:
            preview_payload = render_future_forecast_preview(
                experiment=experiment,
                history_df=history_df,
                forecast_horizon=forecast_horizon,
                chart_placeholder=forecast_chart_placeholder,
                pso_model=final_result['model'],
                pso_params=optimization_result.params,
                pso_test_df=build_full_history_prediction_df(
                    experiment,
                    final_result['model'],
                    optimization_result.params,
                ),
                baseline_model=baseline_result['model'],
                baseline_params=baseline_params,
                baseline_test_df=build_full_history_prediction_df(
                    experiment,
                    baseline_result['model'],
                    baseline_params,
                ),
            )
        return {
            'model': final_result['model'],
            'params': optimization_result.params,
            'optimization': optimization_result,
            'final': final_result,
            'baseline': baseline_result,
            'baseline_params': baseline_params,
            'pso_future': preview_payload['pso_future'],
            'baseline_future': preview_payload['baseline_future'],
            'pso_in_sample': preview_payload['pso_in_sample'],
            'baseline_in_sample': preview_payload['baseline_in_sample'],
            'source': 'trained',
        }

    model, params = loaded
    if fine_tune_from_saved:
        try:
            final_result = experiment.fine_tune_model(
                params,
                model,
                callbacks=[
                    EpochProgressCallback(
                        progress_bar,
                        status_box,
                        "Tinh chỉnh model cũ",
                        experiment.config.fine_tune.epochs,
                    )
                ],
            )
            experiment.save_artifacts(final_result['model'], None, final_result['metrics'])
            return {
                'model': final_result['model'],
                'params': params,
                'optimization': None,
                'final': final_result,
                'baseline': None,
                'baseline_params': None,
                'pso_future': None,
                'baseline_future': None,
                'pso_in_sample': None,
                'baseline_in_sample': None,
                'source': 'fine_tuned',
            }
        except Exception as fine_tune_error:
            status_box.warning(
                f"Tinh chỉnh model cũ thất bại ({fine_tune_error}). Sử dụng model đã lưu để tiếp tục dự báo."
            )

    return {
        'model': model,
        'params': params,
        'optimization': None,
        'final': None,
        'baseline': None,
        'baseline_params': None,
        'pso_future': None,
        'baseline_future': None,
        'pso_in_sample': None,
        'baseline_in_sample': None,
        'source': 'loaded',
    }


def collect_preflight_training_messages(
    experiment: PSOLSTMExperiment,
    data_df: pd.DataFrame,
    run_pso: bool,
    fine_tune_from_saved: bool,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    data_end = pd.to_datetime(data_df["Date"]).max()
    split_timestamp = pd.to_datetime(experiment.config.data.split_date)
    if pd.notna(data_end) and split_timestamp >= data_end:
        errors.append("split_date đang vượt quá ngày cuối của dữ liệu đã lọc. Hãy chọn ngày chia sớm hơn.")

    pso_min_look_back = int(experiment.config.pso.bounds[0][0])
    if (run_pso or not MODEL_PATH.exists()) and not errors:
        is_valid, message = experiment.validate_split_and_look_back(pso_min_look_back)
        stats = experiment.get_window_stats(pso_min_look_back)
        if not is_valid:
            errors.append(f"PSO (look_back tối thiểu {pso_min_look_back}): {message}")
        elif int(stats["train_windows"]) < 8:
            warnings.append(
                (
                    f"PSO (look_back tối thiểu {pso_min_look_back}) chỉ có {stats['train_windows']} mẫu train. "
                    "Nên giảm split_date hoặc chọn ngày bắt đầu sớm hơn để model ổn định hơn."
                )
            )

    saved_params = load_saved_params()
    if fine_tune_from_saved and saved_params is not None and not errors:
        is_valid, message = experiment.validate_split_and_look_back(saved_params.look_back)
        if not is_valid:
            errors.append(f"Fine-tune model đã lưu (look_back={saved_params.look_back}): {message}")

    if fine_tune_from_saved and MODEL_PATH.exists() and saved_params is None:
        warnings.append("Không tìm thấy best_params.json nên không thể tiền kiểm look_back của model đã lưu.")

    return errors, warnings


st.sidebar.title("Gold Forecast Studio")
st.sidebar.caption("Bảng điều khiển mô hình")
refresh_data = st.sidebar.checkbox("Làm mới dữ liệu từ yfinance", value=False)
run_pso = st.sidebar.checkbox("Tối ưu lại PSO khi chạy", value=False)
fine_tune_from_saved = st.sidebar.checkbox("Tinh chỉnh từ model đã lưu", value=True)
forecast_horizon = st.sidebar.slider("Số phiên dự báo", min_value=20, max_value=60, value=42, step=1)
start_date = st.sidebar.text_input("Ngày bắt đầu dữ liệu", value="2005-01-01")
split_date = st.sidebar.text_input("Ngày chia train/test", value="2026-02-01")

experiment = get_experiment()
experiment.config = type(experiment.config)(
    data=type(experiment.config.data)(start_date=start_date, split_date=split_date),
    pso=experiment.config.pso,
    baseline=experiment.config.baseline,
    final_training_epochs=experiment.config.final_training_epochs,
    artifact_dir=experiment.config.artifact_dir,
)

if refresh_data:
    clear_cached_data()
    st.session_state.pop("model_bundle", None)

data_df = load_market_data(start_date=start_date, split_date=split_date, refresh=refresh_data)
correlation_df = data_df.drop(columns=["Date"]).corr(numeric_only=True)

preflight_errors, preflight_warnings = collect_preflight_training_messages(
    experiment,
    data_df,
    run_pso=run_pso,
    fine_tune_from_saved=fine_tune_from_saved,
)

if preflight_errors:
    st.sidebar.error("Không thể huấn luyện với cấu hình hiện tại:\n- " + "\n- ".join(preflight_errors))
if preflight_warnings:
    st.sidebar.warning("Cảnh báo trước huấn luyện:\n- " + "\n- ".join(preflight_warnings))

st.markdown('<div class="hero-card">', unsafe_allow_html=True)
st.title("DỰ ĐOÁN GIÁ VÀNG")
st.write(
    "Website này hiển thị dữ liệu benchmark từ yfinance, ma trận tương quan, dự báo PSO-LSTM và dự báo tiếp 2 tháng tới."
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Tổng quan dữ liệu</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Số dòng", f"{len(data_df):,}")
col2.metric("Ngày đầu", str(pd.to_datetime(data_df['Date']).min().date()))
col3.metric("Ngày cuối", str(pd.to_datetime(data_df['Date']).max().date()))
col4.metric("Biến đầu vào", f"{len(data_df.columns) - 1}")

st.dataframe(data_df.tail(10), use_container_width=True, height=260)

model_bundle = st.session_state.get("model_bundle")
if model_bundle is None:
    saved_model = load_saved_model()
    if saved_model is not None:
        model_bundle = {
            "model": saved_model[0],
            "params": saved_model[1],
            "optimization": None,
            "final": None,
            "baseline": None,
            "baseline_params": None,
            "pso_future": None,
            "baseline_future": None,
            "pso_in_sample": None,
            "baseline_in_sample": None,
            "source": "loaded",
        }
        st.session_state["model_bundle"] = model_bundle

st.markdown('<div class="section-title">Biểu đồ ma trận tương quan</div>', unsafe_allow_html=True)
st.plotly_chart(
    build_correlation_chart(correlation_df),
    use_container_width=True,
    key="correlation_heatmap",
)

forecast_preview_box = st.empty()

train_disabled = len(preflight_errors) > 0
if st.sidebar.button("Huấn luyện / tải mô hình", use_container_width=True, disabled=train_disabled):
    with st.spinner("Đang chuẩn bị mô hình... quá trình này có thể mất vài phút"):
        try:
            progress_bar = st.progress(0, text="Đang chờ huấn luyện...")
            status_box = st.empty()
            session_result = train_or_load_model(
                experiment,
                run_pso=run_pso,
                fine_tune_from_saved=fine_tune_from_saved,
                progress_bar=progress_bar,
                status_box=status_box,
                history_df=data_df,
                forecast_horizon=forecast_horizon,
                forecast_chart_placeholder=forecast_preview_box,
            )
            st.session_state["model_bundle"] = session_result
            model_bundle = session_result
            if session_result.get("source") == "fine_tuned":
                st.success("Đã tinh chỉnh model cũ và sẵn sàng dự báo.")
            else:
                st.success("Đã sẵn sàng mô hình để dự báo.")
        except Exception as error:
            st.session_state.pop("model_bundle", None)
            model_bundle = None
            st.error(f"Không thể huấn luyện/tải mô hình: {error}")

left_tab, right_tab = st.tabs(["Dự báo 2 tháng tới", "Chi tiết mô hình"])

with left_tab:
    st.subheader("Đường dự báo tương lai")
    if model_bundle is None:
        st.info("Nhấn nút 'Huấn luyện / tải mô hình' ở thanh bên để tạo model và dự báo 2 tháng tới.")
    else:
        with st.spinner("Đang sinh dự báo tương lai..."):
            try:
                bundle_experiment = experiment
                bundle_experiment.load_data_for_app(refresh=refresh_data)
                future_df = bundle_experiment.forecast_next_days(
                    model_bundle["model"],
                    model_bundle["params"],
                    forecast_horizon,
                )
                baseline_future_df = None
                baseline_result = model_bundle.get("baseline")
                baseline_params = model_bundle.get("baseline_params")
                if baseline_result is not None and baseline_params is not None:
                    baseline_future_df = bundle_experiment.forecast_next_days(
                        baseline_result["model"],
                        baseline_params,
                        forecast_horizon,
                    )
                pso_in_sample_df = model_bundle.get("pso_in_sample")
                if pso_in_sample_df is None and model_bundle.get("final") is not None:
                    pso_in_sample_df = build_full_history_prediction_df(
                        bundle_experiment,
                        model_bundle["model"],
                        model_bundle["params"],
                    )

                baseline_in_sample_df = model_bundle.get("baseline_in_sample")
                if baseline_in_sample_df is None and baseline_result is not None:
                    baseline_in_sample_df = build_full_history_prediction_df(
                        bundle_experiment,
                        baseline_result["model"],
                        baseline_params,
                    )
                chart_df = build_prediction_chart(
                    data_df,
                    future_df,
                    baseline_future_df,
                    pso_in_sample_df,
                    baseline_in_sample_df,
                    split_date=bundle_experiment.config.data.split_date,
                )
                st.plotly_chart(chart_df, use_container_width=True, key="forecast_main_chart")

                summary_col1, summary_col2, summary_col3 = st.columns(3)
                summary_col1.metric("Giá cuối lịch sử", f"{data_df['Gold_Close'].iloc[-1]:,.2f} USD")
                summary_col2.metric("Giá dự báo đầu kỳ", f"{future_df['Predicted_Gold_Close'].iloc[0]:,.2f} USD")
                summary_col3.metric("Giá dự báo cuối kỳ", f"{future_df['Predicted_Gold_Close'].iloc[-1]:,.2f} USD")

                st.dataframe(future_df, use_container_width=True, height=260)
                st.caption(
                    "Lưu ý: dự báo tương lai dùng chiến lược recursive và giữ các biến ngoại sinh ở mức gần nhất quan sát được."
                )
            except Exception as forecast_error:
                st.error(f"Không thể tạo biểu đồ dự báo: {forecast_error}")

with right_tab:
    st.subheader("Thông tin mô hình và đánh giá")
    if model_bundle is None:
        st.info("Chưa có mô hình trong phiên làm việc này.")
    else:
        params = model_bundle["params"]
        st.write("**Siêu tham số sử dụng**")
        st.json(
            {
                "look_back": params.look_back,
                "hidden_units": params.hidden_units,
                "learning_rate": params.learning_rate,
                "batch_size": params.batch_size,
                "num_layers": params.num_layers,
                "dropout_rate": params.dropout_rate,
            }
        )

        if model_bundle.get("final") is not None:
            final_result = model_bundle["final"]
            baseline_result = model_bundle.get("baseline")
            st.metric("MAE", f"{final_result['metrics'].mae:.2f} USD")
            st.metric("RMSE", f"{final_result['metrics'].rmse:.2f} USD")
            st.metric("MAPE", f"{final_result['metrics'].mape:.2f}%")
            st.metric("R²", f"{final_result['metrics'].r2:.4f}")
            st.metric("Directional Accuracy", f"{final_result['metrics'].directional_accuracy:.2f}%")

            if baseline_result is not None:
                st.markdown("**So sánh với baseline**")
                compare_df = pd.DataFrame(
                    {
                        "Metric": ["MAE", "RMSE", "MAPE", "R2", "Directional Accuracy"],
                        "PSO-LSTM": [
                            final_result["metrics"].mae,
                            final_result["metrics"].rmse,
                            final_result["metrics"].mape,
                            final_result["metrics"].r2,
                            final_result["metrics"].directional_accuracy,
                        ],
                        "Baseline": [
                            baseline_result["metrics"].mae,
                            baseline_result["metrics"].rmse,
                            baseline_result["metrics"].mape,
                            baseline_result["metrics"].r2,
                            baseline_result["metrics"].directional_accuracy,
                        ],
                    }
                )
                st.dataframe(compare_df, use_container_width=True, hide_index=True)
                st.markdown("**Biểu đồ so sánh trên tập test**")
                st.plotly_chart(
                    build_test_comparison_chart(final_result, baseline_result),
                    use_container_width=True,
                    key="test_comparison_chart",
                )
            else:
                st.info("Model này được tinh chỉnh từ bản đã lưu nên chỉ hiển thị kết quả fine-tune.")
        else:
            st.write("Model đã tải từ artifact đã lưu.")

    st.markdown("**Đường dẫn lưu trữ**")
    st.code(
        f"{MODEL_PATH}\n{BEST_PARAMS_PATH}\n{METRICS_PATH}",
        language="text",
    )

st.markdown('<div class="section-title">Ghi chú triển khai</div>', unsafe_allow_html=True)
st.info(
    "Biến ngoại sinh trong phần dự báo tương lai chưa có giá trị thực tế nên app dùng giá trị gần nhất đã quan sát để mô phỏng kịch bản. "
    "Nếu muốn, mình có thể nâng cấp thêm phần nhập kịch bản DXY/Oil/SP500 cho 2 tháng tới."
)


