from typing import List
from abc import abstractmethod, ABC

from commons.datatype import Databatch
from commons.config import Config


class Processor(ABC):
    @abstractmethod
    def __call__(self, batch: Databatch) -> Databatch:
        pass

class PipelineProcessor(Processor):
    def __init__(self, processor_lst: List[Processor]):
        super().__init__()
        self._processor_lst = processor_lst

    def __call__(self, batch: Databatch) -> Databatch:
        for processor in self._processor_lst:
            batch = processor(batch)
        return batch
