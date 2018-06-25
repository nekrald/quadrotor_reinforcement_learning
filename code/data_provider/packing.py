from typing import List, Tuple
from enum import Enum

from handle import ObjectWrapper, ObjectType
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PackType(Enum):
    NUMPY = 1
    TORCH = 2


class PackDescription(object):

    def __init__(self,
            chunk_sizes: List[int],
            chunk_shapes: List[Tuple],
            chunk_names: List[str],
            pack_type: PackType,
            object_type=None: ObjectType,
            chunk_keys=None: List[str]):

        self.chunk_sizes = chunk_sizes
        self.chunk_shapes = chunk_shapes
        self.chunk_names = chunk_names
        self.pack_type = pack_type
        self.chunk_keys = chunk_keys
        self.object_types = object_types


def pack_to_np_array(object_array: List[ObjectWrapper]) -> (
        np.array, PackDescription):
    raise NotImplementedError


def pack_to_torch_tensor(object_array: List[ObjectWrapper]) -> (
        torch.Tensor, PackDescription)
    raise NotImplementedError


def unpack_np_to_object(
            data: np.array,
            description: PackDescription) -> List[
                    ObjectWrapper]:
    raise NotImplementedError


def unpack_torch_to_object(
            data: torch.Tensor,
            description: PackDescription) -> List[
                    ObjectWrapper]:
    raise NotImplementedError


