from gym import Env
import numpy as np

from client.AirSimClient import MultirotorClient, Pose, Vector3r, \
    AirSimClientBase

class ObservationConfig(object):
    def __init__(self):
        self.track_cameras = [3]
        self.image_stacking = 4
        self.append_coordinates = False
        self.image_types = []


class Observation(object):
    def __init__(self):
        pass


class EnvironmentConfig(object):
    def __init__(self):
        self.action_space = None
        self.reward_estimator = None
        self.state_config = None
        self.forward_only = None

    def __init__(self, action_space, reward_estimator, state_config,
            forward_only=False):
        self.action_space = action_space
        self.reward_estimator = reward_estimator
        self.state_config = state_config
        self.forward_only = forward_only


class QuadrotorEnvironment(Env):
    def __init__(self, config):

        self.action_space = config.action_space
        self.state_config = config.state_config
        self.reward_estimator = config.reward_estimator
        self.config = config

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
        self.reward_estimator.reset()
        return self._get_current_observation()

    def _get_observation(self):
        raise NotImplementedError

    def _get_explicit_observation(self):
        raise NotImplementedError

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
        quad_offset = self.action_space.interpret_action(action)
        quad_before_state = self.client.getPosition()
        if args.forward_only:
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
        quad_position = client.getPosition()
        quad_velocity = client.getVelocity()
        collision_info = client.getCollisionInfo()


        pass

    def render(self, mode='human'):
        pass

    def close(self):
        return

    def seed(self, seed=None):
        logger.warn("Could not seed environment %s", self)
        return

