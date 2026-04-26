from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float
    r2: float
    directional_accuracy: float

    def to_dict(self) -> dict[str, float]:
        return {
            "MAE": self.mae,
            "RMSE": self.rmse,
            "MAPE": self.mape,
            "R2": self.r2,
            "Directional_Accuracy": self.directional_accuracy,
        }


@dataclass(frozen=True, slots=True)
class LSTMHyperParameters:
    look_back: int
    hidden_units: int
    learning_rate: float
    batch_size: int
    num_layers: int
    dropout_rate: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    metrics: RegressionMetrics
    y_true: list[float]
    predictions: list[float]