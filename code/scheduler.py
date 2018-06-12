from client.AirSimClient import MultirotorClient, Pose, Vector3r, \
    AirSimClientBase
from environment import EnvironmentConfig, QuadrotorEnvironment


class ISchedulerConfig(object):
    def __init__(self, env_config: EnvironmentConfig):
        self.env_config = env_config


class IScheduler(object):

    def __init__(self, config: ISchedulerConfig):

        # Copy initialization to fields.
        self.config = config

        self.env_config = config.env_config
        self.request_config = self.env_config.request_config
        self.env = QuadrotorEnvironment(self.env_config)

    def process_problem(self):
        raise NotImplementedError("Abstract class!")

