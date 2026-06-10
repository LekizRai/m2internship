from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from commons.processor import Processor, PipelineProcessor
from processors.graph_processor import GraphBuildingProcessor
from processors.device_processor import DeviceProcessor


def make_tacgraspnet_processors(config: TacGraspNetConfig) -> tuple[Processor, Processor]:
    preprocessor = PipelineProcessor([
        DeviceProcessor(config.device),
        GraphBuildingProcessor(config),
        # DeviceProcessor(config.device),
    ])
    postprocessor = PipelineProcessor([])
    return preprocessor, postprocessor