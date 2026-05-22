import torch

from typing import Dict, Any

from commons.config import Config
from commons.processor import Processor
from commons.datatype import Databatch


class DeviceProcessor(Processor):
    def __init__(self, device: str = "cpu"):
        self._device = device

    def __call__(self, batch: Databatch) -> Databatch:
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self._device)
        return batch
