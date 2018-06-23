from enum import Enum


class ObjectType(Enum):
    NP_ARRAY     = 1
    TORCH_TENSOR = 2


class ObjectWrapper(object):
    def __init__(self, obj_type, shape, name, base_request) :
        self.obj_type = obj_type
        self.name = name
        self.base_request = base_request
        self.shape = shape


