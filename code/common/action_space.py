from enum import Enum

from custom.constants import RootConfigKeys, ActionConfigKeys


class AbstractActionSpace(object):
    def __init__(self):
        pass

    def interpret_action(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_num_actions(self):
        raise NotImplementedError


class ActionSpaceType(object):
    DEFAULT_SPACE = 'default'
    GRID_SPACE = 'grid'
    FLAT_SPACE = 'flat'
    CORRIDOR_SPACE = 'corridor'


class DefaultActionSpace(object):
    def __init__(self, scaling_factor=0.25):
        self.scaling_factor = scaling_factor
        self.num_actions = 7

    def interpret_action(self, action):
        scaling_factor = self.scaling_factor
        if action == 0:
            quad_offset = (0, 0, 0)
        elif action == 1:
            quad_offset = (scaling_factor, 0, 0)
        elif action == 2:
            quad_offset = (0, scaling_factor, 0)
        elif action == 3:
            quad_offset = (0, 0, scaling_factor)
        elif action == 4:
            quad_offset = (-scaling_factor, 0, 0)
        elif action == 5:
            quad_offset = (0, -scaling_factor, 0)
        elif action == 6:
            quad_offset = (0, 0, -scaling_factor)
        return quad_offset

    def get_num_actions(self):
        return self.num_actions


class FlatActionSpace(object):
    def __init__(self, scaling_factor=0.25, backward_coef=0.25, sideways_coef=0.5, yaw_degree=20):
        self.scaling_factor = scaling_factor
        self.backward_coef = backward_coef
        self.sideways_coef = sideways_coef
        self.yaw_degree = yaw_degree
        self.num_actions = 7

    def interpret_action(self, action):
        scaling_factor = self.scaling_factor
        backward_coef = self.backward_coef
        sideways_coef = self.sideways_coef
        yaw_degree = self.yaw_degree

        if action == 0:
            quad_offset = (scaling_factor, 0, 0)
        elif action == 1:
            quad_offset = (-backward_coef*scaling_factor, 0, 0)
        elif action == 2:
            quad_offset = (0, sideways_coef*scaling_factor, 0)
        elif action == 3:
            quad_offset = (0, -sideways_coef*scaling_factor, 0)
        elif action == 4:
            quad_offset = (0, 0, scaling_factor)
        elif action == 5:
            quad_offset = (yaw_degree,)
        elif action == 6:
            quad_offset = (-yaw_degree,)
        return quad_offset

    def get_num_actions(self):
        return self.num_actions


class GridActionSpace(object):

    def __init__(self, scaling_factor=0.25, grid_size=0):
        self.grid_size = grid_size
        self.scaling_factor = scaling_factor

    def interpret_action(self, action):
        assert action < self.grid_size * self.grid_size
        scaling_factor = self.scaling_factor
        dx = scaling_factor
        dy = scaling_factor * action / self.grid_size
        dz = scaling_factor * action % self.grid_size
        return dx, dy, dz

    def get_num_actions(self):
        return self.grid_size * self.grid_size


class CorridorActionSpace(object):
    def __init__(self, scaling_factor=0.5, backward_coef=0.25, sideways_coef=0.5, yaw_degree=20):
        self.scaling_factor = scaling_factor
        self.backward_coef = backward_coef
        self.sideways_coef = sideways_coef
        self.yaw_degree = yaw_degree
        self.num_actions = 8

    def interpret_action(self, action):
        scaling_factor = self.scaling_factor
        backward_coef = self.backward_coef
        sideways_coef = self.sideways_coef
        yaw_degree = self.yaw_degree

        if action == 0:
            quad_offset = (scaling_factor, 0, 0)
        elif action == 1:
            quad_offset = (-backward_coef*scaling_factor, 0, 0)
        elif action == 2:
            quad_offset = (0, sideways_coef*scaling_factor, 0)
        elif action == 3:
            quad_offset = (0, -sideways_coef*scaling_factor, 0)
        elif action == 4:
            quad_offset = (0, 0, scaling_factor)
        elif action == 5:
            quad_offset = (0, 0, -scaling_factor)
        elif action == 6:
            quad_offset = (yaw_degree,)
        elif action == 7:
            quad_offset = (-yaw_degree,)
        return quad_offset

    def get_num_actions(self):
        return self.num_actions


def make_action(config):
    action_config = config[RootConfigKeys.ACTION_CONFIG]
    scale_factor = action_config[ActionConfigKeys.SCALING_FACTOR]
    if action_config[ActionConfigKeys.ACTION_SPACE_TYPE] == ActionSpaceType.DEFAULT_SPACE:
        print("Selected default action space")
        action = DefaultActionSpace(scale_factor)
    elif action_config[ActionConfigKeys.ACTION_SPACE_TYPE] == ActionSpaceType.GRID_SPACE:
        action = GridActionSpace(scale_factor, action_config[ActionConfigKeys.GRID_SIZE])
    elif action_config[ActionConfigKeys.ACTION_SPACE_TYPE] == ActionSpaceType.FLAT_SPACE:
        action = FlatActionSpace(scale_factor)
    elif action_config[ActionConfigKeys.ACTION_SPACE_TYPE] == ActionSpaceType.CORRIDOR_SPACE:
        action = CorridorActionSpace(scale_factor)
    else:
        raise ValueError("Unexpected ActionSpaceType.")
    return action

