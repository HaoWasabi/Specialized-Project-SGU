from __future__ import annotations

import logging
import warnings

from src.bll.forecasting_service import PSOLSTMExperiment


def main() -> None:
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    experiment = PSOLSTMExperiment()
    results = experiment.run()

    optimization_result = results["optimization"]
    final_result = results["final"]
    baseline_result = results["baseline"]

    print("\n" + "=" * 80)
    print("PSO TỐI ƯU SIÊU THAM SỐ")
    print("=" * 80)
    print(f"Best cost     : {optimization_result.cost:.4f}")
    print(f"Look-back     : {optimization_result.params.look_back}")
    print(f"Hidden units  : {optimization_result.params.hidden_units}")
    print(f"Learning rate : {optimization_result.params.learning_rate:.5f}")
    print(f"Batch size    : {optimization_result.params.batch_size}")
    print(f"LSTM layers   : {optimization_result.params.num_layers}")
    print(f"Dropout rate  : {optimization_result.params.dropout_rate:.2f}")

    print("\n" + "=" * 80)
    print("KẾT QUẢ PSO-LSTM")
    print("=" * 80)
    print(f"MAE                  : {final_result['metrics'].mae:.2f} USD")
    print(f"RMSE                 : {final_result['metrics'].rmse:.2f} USD")
    print(f"MAPE                 : {final_result['metrics'].mape:.2f}%")
    print(f"R²                   : {final_result['metrics'].r2:.4f}")
    print(f"Directional Accuracy : {final_result['metrics'].directional_accuracy:.2f}%")

    print("\n" + "=" * 80)
    print("KẾT QUẢ LSTM BASELINE")
    print("=" * 80)
    print(f"MAE                  : {baseline_result['metrics'].mae:.2f} USD")
    print(f"RMSE                 : {baseline_result['metrics'].rmse:.2f} USD")
    print(f"MAPE                 : {baseline_result['metrics'].mape:.2f}%")
    print(f"R²                   : {baseline_result['metrics'].r2:.4f}")
    print(f"Directional Accuracy : {baseline_result['metrics'].directional_accuracy:.2f}%")

    print("\nLƯU TRỮ HOÀN TẤT!")
    print("Các file đã được lưu trong thư mục artifacts/")
    print("   • model_PSO_LSTM_final.h5")
    print("   • scaler_gold.pkl")
    print("   • best_params.json")
    print("   • model_metrics.json")


if __name__ == "__main__":
    main()