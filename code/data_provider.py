import numpy as np
from environment import Observation


class AbstractTransform(object):
    def __init__(self):
        pass

    def transform(self, key, array_values):
        raise NotImplementedError


class ScalingTransform(AbstractTransform):
    def __init__(self):
        pass

    def transform(self, key, array_values):
        raise NotImplementedError


class StackingTransform(AbstractTransform):
    def __init__(self):
        pass

    def transform(self, key, array_values):
        raise NotImplementedError


class SelectKthTransform(AbstractTransform):
    def __init__(self, idx=0):
        self.idx = idx

    def transform(self, key, array_values):
        return [array_values[self.idx]]


class CompositeTransform(AbstractTransform):
    def __init__(self, transforms):
        self.transforms = transforms

    def transform(self, key, array_values):
        result_now = array_values
        for operation in transforms:
            result_now = operation(key, result_now)
        return result_now


class PackDescription(object):
    def __init__(self):
        raise NotImplementedError


class DataProvider(object):

    def observation_to_storage(cls, observation: Observation):
        raise NotImplementedError

    def transform_sequence(cls, array_storage, array_transform):
        raise NotImplementedError

    def np_to_tensor(cls, storage):
        raise NotImplementedError

    def pack_to_np_array(cls, storage):
        raise NotImplementedError

    def unpack_np_array(cls, array, description) -> dict:
        raise NotImplementedError

    def pack_to_tensor(cls, storage):
        raise NotImplementedError

    def unpack_tensor(cls, tensor, description) -> dict:
        raise NotImplementedError


