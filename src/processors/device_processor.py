import torch

from typing import Dict, Any

from commons.config import Config
from commons.processor import Processor


class DeviceProcessor(Processor):
    def __init__(self, config: Config):
        self._config = config

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self._config.device)
        return batch
