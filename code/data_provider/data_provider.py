from typing import List
from enum import Enum

from request import RequestType, IRequest, ImageRequest, \
        VelocityRequest, PositionRequest
from network import NetworkConfig
from transform import ITransform


def make_data_provider_config(
        json_provider_config,
        reward_requirement,
        debug_requirement):
    state_requirement = _make_state_requirement_config(
            json_provider_config)
    raise NotImplementedError


def _make_state_requirement_config(json_provider_config):
    raise NotImplementedError


class StateReturnType(Enum):
    SHAPED_NP_ARRAY       = 1
    SHAPED_TENSOR         = 2
    FLAT_NP_ARRAY         = 3
    FLAT_NP_DESCRIBED     = 4
    FLAT_TENSOR           = 5
    FLAT_TENSOR_DESCRIBED = 6
    OBJECT_NP_LIST        = 7
    OBJECT_TENSOR_LIST    = 8


class QueryConfig(object):

    def __init__(self,
            request_array: List[IRequest],
            transform_array: List[ITransform],
            frame_stacking_factor: int,
            state_return_type: StateReturnType):

        self.request_array = request_array
        self.transform_array = transform_array
        self.frame_stacking_factor = frame_stacking_factor
        self.state_return_type = state_return_type


class DataProviderConfig(object):

    def __init__(self,
            state_requirement: QueryConfig,
            reward_requirement: QueryConfig,
            debug_requirement: QueryConfig):
        self.state_requirement = state_requirement
        self.reward_requirement = reward_requirement
        self.debug_requirement = debug_requirement


class IDataProvider(object):

    def __init__(self,
            config: DataProviderConfig):
        self.config = config
        self.last_frames = []
        self._configure_data_distribution()

    def _configure_data_distribution(self):
        raise NotImplementedError

    def make_observation(self, client, override_last=False):
        raise NotImplementedError

    def submit_reward_requests(self, client, override_last=True):
        raise NotImplementedError

    def provide_debug_data(self, client, override_last=False):
        raise NotImplementedError

    def make_network_config(self) -> NetworkConfig:
        raise NotImplementedError


class TorchDataProvider(IDataProvider):
    def __init__(self):
        raise NotImplementedError
    def _get_sensor_data(self):
        sensed = SensorData()
        requests = []
        transforms = []
        for id_camera, scan_type, transform in \
                self.request_config.request_data:
            requests.append(ImageRequest(id_camera, scan_type,
                    True, False))
            transforms.append(transform)
        sensed.response_frames = client.simGetImages(requests)
        for answer, transform in zip(
                sensed.response_frames, transforms):
            sensed.transformed_frames.append(
                    transform_response(answer,
                        transform[0], transform[1]))
        sensed.make_np_array()
        return sensed

    def _get_state_data(self):
        state = StateData()
        state.quad_position = self.client.getPosition()
        state.quad_velocity = self.client.getVelocity()
        if self.request_config.provide_state:
            state.make_np_array()
        return state

    def _get_observation(self):
        observation = Observation()
        if self.request_config.provide_sensor:
            observation.sensor_data = self._get_sensor_data()
        state_data =  self._get_state_data()
        if self.request_config.provide_state:
            observation.state_data = state_data
        self.last_observation = observation
        self.last_state = state_data
        if self.request_config.raw_result:
            return observation.to_flat_array()
        return observation


