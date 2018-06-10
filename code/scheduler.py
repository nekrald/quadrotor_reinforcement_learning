
from client.AirSimClient import MultirotorClient, Pose, Vector3r, \
    AirSimClientBase
from environment import EnvironmentConfig, QuadrotorEnvironment

class ISchedulerConfig(object):
    def __init__(self, env_config: EnvironmentConfig, args):
        self.env_config = env_config
        self.args = args



class IScheduler(object):

    def __init__(self, config: ISchedulerConfig):

        # Copy initialization to fields.
        self.config = config
        self.args = config.args

        self.env_config = config.env_config
        self.env = QuadrotorEnvironment(self.env_config)


    def process_problem(self):
        raise NotImplementedError("Abstract class!")
