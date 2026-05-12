from typing import Dict, Any
import torch

from commons.config import Config


class DeviceProcessor:
    def __init__(self, config: Config):
        self._config = config

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self._config.device)
        return batch
