from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DataConfig:
    start_date: str = "2005-01-01"
    split_date: str = "2026-02-01"


@dataclass(frozen=True, slots=True)
class PSOConfig:
    n_particles: int = 12
    dimensions: int = 6
    iterations: int = 10
    bounds: tuple[tuple[float, ...], tuple[float, ...]] = (
        (30, 80, 0.0005, 16, 1, 0.05),
        (120, 280, 0.008, 256, 3, 0.45),
    )
    options: dict[str, float] = field(
        default_factory=lambda: {"c1": 2.0, "c2": 2.0, "w": 0.9}
    )
    train_eval_ratio: float = 0.80
    objective_epochs: int = 6
    objective_patience: int = 5


@dataclass(frozen=True, slots=True)
class LSTMConfig:
    look_back: int = 60
    hidden_units: int = 128
    learning_rate: float = 0.001
    batch_size: int = 32
    num_layers: int = 2
    dropout_rate: float = 0.2
    epochs: int = 60
    validation_split: float = 0.10
    patience: int = 8


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    pso: PSOConfig = field(default_factory=PSOConfig)
    baseline: LSTMConfig = field(default_factory=LSTMConfig)
    final_training_epochs: int = 100
    artifact_dir: Path = field(default_factory=lambda: Path("artifacts"))