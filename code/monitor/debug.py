
def make_debug_config(config_json_dict: dict):
    raise NotImplementedError


class DebugConfig(object):
    def __init__(self):
        self.path_to_folder = path_to_folder
        self.step_write_frequency = step_write_frequency
        self.write_from_cameras = write_from_cameras
        self.write_position = write_position
        self.write_action = write_action
        self.write_action_percentage = write_action_percentage


def make_debug_requirements(object) -> (
        List[IRequest], List[ITransform], int):
    return NotImplementedError

