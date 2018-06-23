import numpy as np
from typing import List

from handle import ObjectWrapper, ObjectType


class ITransform(object):

    def __init__(self):
        pass

    def apply(object_array: List[ObjectWrapper]) -> List[
            ObjectWrapper]:
        raise NotImplementedError


class ResizeTransform(ITransform):

    def __init__(self, target_shape):
        raise NotImplementedError

    def apply(object_array: List[ObjectWrapper]) -> List[
            ObjectWrapper]:
        raise NotImplementedError


class StackTransform(ITransform):

    def __init__(self):
        raise NotImplementedError

    def apply(object_array: List[ObjectWrapper]) -> List[
            ObjectWrapper]:
        raise NotImplementedError


class NumpyToTensorTransform(ITransform):

    def __init__(self):
        raise NotImplementedError

    def apply(object_array: List[ObjectWrapper]) -> List[
            ObjectWrapper]:
        raise NotImplementedError


class CompositeTransform(ITransform):

    def __init__(self, transforms: List[ITransform]):
        raise NotImplementedError

    def apply(object_array: List[ObjectWrapper]): -> List[
            ObjectWrapper]:
        raise NotImplementedError


