from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MolEncoderConfig:
    pretrained_model_name: str = "fabikru/MolEncoder"
    max_sequence_length: int = 256
    num_labels: int = 2
    dropout: float = 0.1
    model_cache_dir: Path | None = None
