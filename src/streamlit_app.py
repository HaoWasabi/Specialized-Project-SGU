from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow import keras

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
def load_market_data(refresh: bool = False) -> pd.DataFrame:
    experiment = get_experiment()
    return experiment.load_data_for_app(refresh=refresh)


def clear_cached_data() -> None:
    load_market_data.clear()


def load_saved_model() -> tuple[keras.Model, LSTMHyperParameters] | None:
    if not MODEL_PATH.exists() or not BEST_PARAMS_PATH.exists():
        return None

    with BEST_PARAMS_PATH.open("r", encoding="utf-8") as file_handle:
        raw_params = json.load(file_handle)

    params = LSTMHyperParameters(
        look_back=int(raw_params["look_back"]),
        hidden_units=int(raw_params["hidden_units"]),
        learning_rate=float(raw_params["learning_rate"]),
        batch_size=int(raw_params["batch_size"]),
        num_layers=int(raw_params["num_layers"]),
        dropout_rate=float(raw_params["dropout_rate"]),
    )
    model = keras.models.load_model(MODEL_PATH)
    return model, params


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


def build_prediction_chart(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["Date"],
            y=history_df["Gold_Close"],
            name="Lịch sử giá vàng",
            line=dict(color="#1f556e", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Predicted_Gold_Close"],
            name="Dự báo 2 tháng tới",
            line=dict(color="#d39f3f", width=3, dash="dash"),
        )
    )
    forecast_start = forecast_df["Date"].iloc[0]
    fig.add_vline(x=forecast_start, line_width=2, line_dash="dot", line_color="#6d7a84")
    fig.update_layout(
        height=580,
        template="plotly_white",
        title="Giá vàng thực tế và dự báo tương lai",
        xaxis_title="Ngày",
        yaxis_title="Giá vàng (USD)",
        legend_title_text="Chú thích",
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def train_or_load_model(experiment: PSOLSTMExperiment, run_pso: bool) -> dict[str, object]:
    if run_pso or not MODEL_PATH.exists():
        optimization_result: OptimizationResult = experiment.optimize_hyperparameters()
        final_result = experiment.train_final_model(optimization_result.params)
        baseline_result = experiment.train_baseline_model()
        experiment.save_artifacts(final_result["model"], optimization_result, final_result["metrics"])
        return {
            "model": final_result["model"],
            "params": optimization_result.params,
            "optimization": optimization_result,
            "final": final_result,
            "baseline": baseline_result,
            "source": "trained",
        }

    loaded = load_saved_model()
    if loaded is None:
        optimization_result = experiment.optimize_hyperparameters()
        final_result = experiment.train_final_model(optimization_result.params)
        baseline_result = experiment.train_baseline_model()
        experiment.save_artifacts(final_result["model"], optimization_result, final_result["metrics"])
        return {
            "model": final_result["model"],
            "params": optimization_result.params,
            "optimization": optimization_result,
            "final": final_result,
            "baseline": baseline_result,
            "source": "trained",
        }

    model, params = loaded
    return {
        "model": model,
        "params": params,
        "optimization": None,
        "final": None,
        "baseline": None,
        "source": "loaded",
    }


st.sidebar.title("Gold Forecast Studio")
st.sidebar.caption("Bảng điều khiển mô hình")
refresh_data = st.sidebar.checkbox("Làm mới dữ liệu từ yfinance", value=False)
run_pso = st.sidebar.checkbox("Tối ưu lại PSO khi chạy", value=False)
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

data_df = load_market_data(refresh=refresh_data)
correlation_df = data_df.drop(columns=["Date"]).corr(numeric_only=True)

st.markdown('<div class="hero-card">', unsafe_allow_html=True)
st.title("Dự đoán giá vàng bằng Streamlit")
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

if st.sidebar.button("Huấn luyện / tải mô hình", use_container_width=True):
    with st.spinner("Đang chuẩn bị mô hình... quá trình này có thể mất vài phút"):
        try:
            session_result = train_or_load_model(experiment, run_pso=run_pso)
            st.session_state["model_bundle"] = session_result
            st.success("Đã sẵn sàng mô hình để dự báo.")
        except Exception as error:
            st.session_state.pop("model_bundle", None)
            st.error(f"Không thể huấn luyện/tải mô hình: {error}")

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
            "source": "loaded",
        }
        st.session_state["model_bundle"] = model_bundle

st.markdown('<div class="section-title">Biểu đồ ma trận tương quan</div>', unsafe_allow_html=True)
st.plotly_chart(build_correlation_chart(correlation_df), use_container_width=True)

left_tab, right_tab = st.tabs(["Dự báo 2 tháng tới", "Chi tiết mô hình"])

with left_tab:
    st.subheader("Đường dự báo tương lai")
    if model_bundle is None:
        st.info("Nhấn nút 'Huấn luyện / tải mô hình' ở thanh bên để tạo model và dự báo 2 tháng tới.")
    else:
        with st.spinner("Đang sinh dự báo tương lai..."):
            bundle_experiment = experiment
            bundle_experiment.load_data_for_app(refresh=refresh_data)
            future_df = bundle_experiment.forecast_next_days(
                model_bundle["model"],
                model_bundle["params"],
                forecast_horizon,
            )
            chart_df = build_prediction_chart(data_df, future_df)
            st.plotly_chart(chart_df, use_container_width=True)

            summary_col1, summary_col2, summary_col3 = st.columns(3)
            summary_col1.metric("Giá cuối lịch sử", f"{data_df['Gold_Close'].iloc[-1]:,.2f} USD")
            summary_col2.metric("Giá dự báo đầu kỳ", f"{future_df['Predicted_Gold_Close'].iloc[0]:,.2f} USD")
            summary_col3.metric("Giá dự báo cuối kỳ", f"{future_df['Predicted_Gold_Close'].iloc[-1]:,.2f} USD")

            st.dataframe(future_df, use_container_width=True, height=260)
            st.caption(
                "Lưu ý: dự báo tương lai dùng chiến lược recursive và giữ các biến ngoại sinh ở mức gần nhất quan sát được."
            )

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
            baseline_result = model_bundle["baseline"]
            st.metric("MAE", f"{final_result['metrics'].mae:.2f} USD")
            st.metric("RMSE", f"{final_result['metrics'].rmse:.2f} USD")
            st.metric("MAPE", f"{final_result['metrics'].mape:.2f}%")
            st.metric("R²", f"{final_result['metrics'].r2:.4f}")
            st.metric("Directional Accuracy", f"{final_result['metrics'].directional_accuracy:.2f}%")

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
