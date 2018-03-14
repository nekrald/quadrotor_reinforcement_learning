class DefaultActionSpace(object):
    def __init__(self):
        pass

    def interpret_action(self, action):
        scaling_factor = 0.25
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
        return 7

class GridActionSpace(object):

    def __init__(self, grid_size=0):
        self.grid_size  = grid_size

    def interpret_action(self, action):
        scaling_factor = 0.25
        assert action < grid_size * grid_size
        dx = scaling_factor
        dy = scaling_factor * action / grid_size
        dz = scaling_factor * action % grid_size

    def get_num_actions(self):
        return self.grid_size * self.grid_size


