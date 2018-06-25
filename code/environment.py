from gym import Env
import numpy as np
from collections import defaultdict

from client.AirSimClient import MultirotorClient, Pose, Vector3r, \
    AirSimClientBase, ImageRequest, AirSimImageType, DrivetrainType

from debug import DebugConfig


def make_environment_config(env_json_dict: dict):
    raise NotImplementedError


def make_environment(environment_config: EnvironmentConfig):
    raise NotImplementedError


class EnvironmentConfig(object):
    def __init__(self,
            action_space_config,
            reward_provider_config,
            data_provider_config,
            forward_only=False,
            debug_config=None):
        self.action_space_config = action_space_config
        self.reward_provider_config = reward_provider_config
        self.data_provider_config = data_provider_config
        self.forward_only = forward_only
        self.debug_config = debug_config


class QuadrotorEnvironment(Env):

    def _init_action_space(self, config):
        self.action_space = make_action_space(
                config.action_space_config)

    def _init_reward_provider(self, config):
        self.reward_provider = make_reward_provider(
                config.reward_provider_config)
        self.reward_requierements = \
                self.reward_provider.get_requirements()

    def _init_data_provider(self, config):
        self.data_provider = make_data_provider(
                config.data_provider_config)

    def _init_debug_policy(self, config):
        raise NotImplementedError

    def __init__(self, config: EnvironmentConfig):
        self.config = config
        self.forward_only = config.forward_only

        self._init_action_space(config)
        self._init_reward_provider(config)
        self._init_debug_policy(config)
        self._init_data_provider(config)

        # Activate client.
        self.client = MultirotorClient()
        self.client.confirmConnection()
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Initial position moves.
        initial_position = self.client.getPosition()
        if config[RootConfigKeys.USE_FLAG_POS]:
            self.initX = initial_position.x_val
            self.initY = initial_position.y_val
            self.initZ = initial_position.z_val
            logging.info("Staying at initial coordinates:"
                + "({}, {}, {})".format(
                    self.initX, self.initY, self.initZ))
        else:
            self.initX = config[RootConfigKeys.INIT_X]
            self.initY = config[RootConfigKeys.INIT_Y]
            self.initZ = config[RootConfigKeys.INIT_Z]
            logging.info(
                ("Ignoring flag. Using coordinates (X, Y, Z):{}"
                + ", Rotation:{}").format(
                    (self.initX, self.initY, self.initZ), (0, 0, 0)))
            self.client.simSetPose(Pose(Vector3r(
                self.initX, self.initY, self.initZ),
                AirSimClientBase.toQuaternion(0, 0, 0)),
                ignore_collison=True)

    def reset(self):
        """Resets the state of the environment and
        returns an initial observation.

        Returns: observation (object): the initial
            observation of the space.
        """
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.simSetPose(Pose(Vector3r(
                self.initX, self.initY, self.initZ),
                AirSimClientBase.toQuaternion(0, 0, 0)),
                ignore_collison=True)
        self.action_space.reset()
        self.reward_provider.reset()
        self.data_provider.reset()
        self.last_observation = None
        self.last_state = None
        return self._get_observation()

    def step(self, action):
        """Run one timestep of the environment's dynamics.
        When end of episode is reached, you are responsible
        for calling `reset()` to reset this environment's state.
        Accepts an action and returns a tuple
        (observation, reward, done, info).
        Args:
            action (object): an action provided by the environment
        Returns:
            observation (object): agent's observation of
                the current environment
            reward (float) : amount of reward returned after
                previous action
            done (boolean): whether the episode has ended, in
                which case further step() calls will return undefined
                results
            info (dict): contains auxiliary diagnostic information
                (helpful for debugging, and sometimes learning)
        """
        info = {}
        quad_offset = self.action_space.interpret_action(action)
        client = self.client
        if self.forward_only:
            if len(quad_offset) == 1:
                client.rotateByYawRate(quad_offset[0],
                        move_duration)
            else:
                client.moveByVelocity(
                    quad_offset[0], quad_offset[1],
                    quad_offset[2], move_duration,
                    DrivetrainType.ForwardOnly)
        else:
            client.moveByVelocity(
                quad_offset[0], quad_offset[1],
                quad_offset[2], move_duration,
                DrivetrainType.MaxDegreeOfFreedom)
        self._update_reward_provider()
        observation = self.data_provider.make_observation(self.client)
        reward = self.reward_provider.compute_reward()
        done = self.reward_provider.is_done()
        self._write_debug_info()
        return observation, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        return

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def _update_reward_provider(self):
        raise NotImplementedError

    def _write_debug_info(self):
        raise NotImplementedError

