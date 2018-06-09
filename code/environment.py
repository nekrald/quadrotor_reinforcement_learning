from gym import Env
import numpy as np
from collections import defaultdict

from client.AirSimClient import MultirotorClient, Pose, Vector3r, \
    AirSimClientBase, ImageRequest, AirSimImageType


class ObservationConfig(object):
    def __init__(self):
        self.camera_ids = [0, 1, 2, 3]
        self.image_types = [
                AirSimImageType.DepthPerspective,
                AirSimImageType.Scene,
                AirSimImageType.Segmentation
            ]
        # Expected to depend on image type,
        # but not on camera.
        self.transformation_law = [(84, 84), (84, 84), (20, 20)]
        self.image_history = 4

        self.track_position = False
        self.image_types = []
        self.make_flat_observation = False


class Observation(object):
    def __init__(self):
        # Camera Inputs.
        self.responses = []
        self.transformed_frames = []

        # Simulator (oracle) inputs.
        self.positions = []
        self.velocities = []

        self.np_conv_input = None
        self.np_position_input = None

        self.flat_observation = None


class EnvironmentConfig(object):

    def __init__(self):
        self.action_space = None
        self.reward_provider = None
        self.observation_config = None
        self.forward_only = None

    def __init__(self, action_space, reward_provider, state_config,
            forward_only=False):
        self.action_space = action_space
        self.reward_provider = reward_provider
        self.state_config = state_config
        self.forward_only = forward_only


class QuadrotorEnvironment(Env):
    def __init__(self, config):

        self.action_space = config.action_space
        self.observation_config = config.observation_config
        self.reward_provider = config.reward_provider
        self.last_observation = None
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
        self.reward_provider.reset()
        self.last_observation = None
        return self._get_observation()

    def _configure_action_space(self):
        raise NotImplementedError

    def _configure_reward_provider(self):
        raise NotImplementedError

    def _get_observation(self):
        requests = []
        for id_camera in self.observation_config.camera_ids:
            for scan_type in self.observation_config.image_types:
                requests.append(ImageRequest(id_camera, scan_type,
                    True, False))
        responses = client.simGetImages(requests)

        self.last_observation = self._prepare_observation(responses)

        if self.observation_config.make_flat_observation:
            raise NotImplementedError
        else:
            return self.last_observation

    def _prepare_observation(self, responses):
        current_observation = copy.deepcopy(self.last_observation)
        transformed = self._transform_reposes(responses)
        quad_position = client.getPosition()
        quad_velocity = client.getVelocity()
        collision_info = client.getCollisionInfo()
        if current_observation is not None:
            del current_observation.responses[0]
            del current_observation.transformed_frames[0]
            del current_observation.positions[0]
            current_observation.np_conv_input = None
            current_observation.np_position_input = None
            current_observation.flat_observation = None
        else:
            current_observation = Observation()
        while len(current_observation.responses) < \
                self.observation_config.image_history:
            current_observation.responses.append(responses)
            current_observation.transformed_frames.append(responses)

        raise NotImplementedError

    def _transform_responses(self, responses):
        raise NotImplementedError


    def _get_complete_observation(self):
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

