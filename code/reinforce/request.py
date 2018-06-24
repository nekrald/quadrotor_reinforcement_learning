from enum import Enum
from client import AirSimImageType


def make_request_array_from_json(json_request_config):
    raise NotImplementedError


def make_request_from_json(json_request_config):
    raise NotImplementedError


class RequestType(Enum):
    VELOCITY = 1
    POSITION = 2
    IMAGE    = 3


class IRequest(object):
    def __init__(self, request_type):
        self.request_type = request_type


class VelocityRequest(IRequest):
    def __init__(self):
        super(VelocityRequest, self).__init__(RequestType.VELOCITY)


class PositionRequest(IRequest):
    def __init__(self):
        super(PositionRequest, self).__init__(RequestType.POSITION)


class ImageRequest(IRequest):
    def __init__(self, image_type: AirSimImageType):
        super(ImageRequest, self).__init__(RequestType.IMAGE)
        self.image_type = image_type

