"""Configuration for ML models and training."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the text completion model."""

    model_name: str = "gpt2"
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1

    # Training parameters
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 500

    # Batch processing parameters
    max_batch_size: int = 16
    batch_timeout_ms: int = 100
    enable_dynamic_batching: bool = True

    # Paths
    model_path: Optional[str] = "models/kraken-text-completion"
    checkpoint_dir: Optional[str] = "models/checkpoints"


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50