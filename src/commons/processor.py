from typing import Any, Dict
from abc import abstractmethod, ABC


class Processor(ABC):
    @abstractmethod
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass
