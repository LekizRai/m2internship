from models.tacgraspnet.tacgraspnet_config import TacGraspNetConfig
from commons.processor import Processor, PipelineProcessor
from processors.graph_processor import GraphBuildingProcessor
from processors.device_processor import DeviceProcessor


def make_tacgraspnet_processors(config: TacGraspNetConfig) -> tuple[Processor, Processor]:
    preprocessor = PipelineProcessor([
        DeviceProcessor(config.device), # Put every tensor to GPU for accelerating graph building process
        GraphBuildingProcessor(config), # Build graph
        DeviceProcessor(config.device), # Put every tensor to GPU for model forwarding
    ])
    postprocessor = PipelineProcessor([]) # Do nothing here
    return preprocessor, postprocessor