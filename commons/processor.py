from typing import Any, Dict
from abc import abstractmethod, ABC

class Processor(ABC):
    @abstractmethod
    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def postprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass
