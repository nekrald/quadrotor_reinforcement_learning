import logging

import math
import numpy as np

from enum import Enum
from client.AirSimClient import AirSimImageType, ImageRequest
from custom.constants import RootConfigKeys, RewardConfigKeys

from scipy.spatial.distance import cityblock, euclidean


class RewardConfig(self):
    def __init__(self):
        reward_type = None


class RewardRequirements(self):
    def __init__(self):
        self.need_quad_vel = None
        self.need_quad_position = None
        self.desired_requests = None
        self.need_last_reward = None
        self.need_epoch_reward_sum = None
        self.need_collision_info = None


class RewardInfo(self):
    def __init__(self):
        self.quad_vel = None
        self.quad_position = None
        self.desired_requests = None
        self.desired_transforms = None
        self.last_reward = None
        self.epoch_reward_sum = None
        self.collision_info = None


class AbstractReward(object):
    def __init__(self, supported=[]):
        self.supported = supported

    def check_if_outdoors_supported(self, outdoor_item):
        return outdoor_item in self.supported

    def get_requirements(self) -> RewardRequirements:
        raise NotImplementedError

    def submit_info(self, info: RewardInfo):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    # Reasonable logic.
    def compute_reward() -> float:
        raise NotImplementedError("Abstract method!")

    def is_done(self) -> bool:
        raise NotImplementedError("Abstract method!")


class AggregatedWeightedReward(AbstractReward):

    def __init__(self, base_rewards: List[AbstractReward],
            reward_weights: List[int]):
        raise NotImplementedError


class AggregatedMinimalReward(AbstractReward):

    def __init__(self):
        raise NotImplementedError


class AggregatedMaximalReward(AbstractReward):

    def __init__(self):
        raise NotImplementedError


class TravelDistanceReward(AbstractReward):

    def __init__(self):
        raise NotImplementedError


class ExploredAreaReward(AbstractReward):
    def __init__(self):
        raise NotImplementedError


class UniformAreaTimingReward(AbstractReward):
    def __init__(self):
        raise NotImplementedError


class FindObjectReward(AbstractReward):
    def __init__(self):
        raise NotImplementedError


class NegCloseToObstacleReward(AbstractReward):
    def __init__(self):
        raise NotImplementedError

class NegTooHighReward(AbstractReward):
    def __init__(self):
        raise NotImplementedError

class NegAwayFromAreaReward(AbstractReward):
    def __init__(self):
        raise NotImplementedError


