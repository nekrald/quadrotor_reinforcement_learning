from typing import List

from request import RequestType, IRequest, ImageRequest, \
        VelocityRequest, PositionRequest
from network import NetworkConfig
from transform import ITransform
from enum import Enum


class StateReturnType(Enum):
    SHAPED_NP_ARRAY       = 1
    SHAPED_TENSOR         = 2
    FLAT_NP_ARRAY         = 3
    FLAT_NP_DESCRIBED     = 4
    FLAT_TENSOR           = 5
    FLAT_TENSOR_DESCRIBED = 6
    OBJECT_NP_LIST        = 7
    OBJECT_TENSOR_LIST    = 8


class DataProviderConfig(object):

    def __init__(self,
            request_array: List[IRequest],
            transform_array: List[ITransform],
            frame_stacking_factor: int,
            state_return_type: StateReturnType):

        self.request_array = request_array
        self.transform_array = transform_array
        self.frame_stacking_factor = frame_stacking_factor
        self.state_return_type = state_return_type


class IDataProvider(object):

    def __init__(self,
            config: DataProviderConfig):
        self.config = config
        self.frames = []

    def make_ready_state(self, client):
        raise NotImplementedError


    def make_network_config(self) -> NetworkConfig:
        raise NotImplementedError


class TorchDataProvider(IDataProvider):
    def __init__(self):
        raise NotImplementedError


