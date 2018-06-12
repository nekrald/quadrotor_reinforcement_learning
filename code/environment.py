from gym import Env
import numpy as np
from collections import defaultdict

from client.AirSimClient import MultirotorClient, Pose, Vector3r, \
    AirSimClientBase, ImageRequest, AirSimImageType, DrivetrainType


class StateData(object):
    def __init__(self):
        self.position = None
        self.velocity = None
        self.np_array = None

    def make_np_array(self):
        raise NotImplementedError

    def to_flat_array(self):
        raise NotImplementedError


class SensorData(object):
    def __init__(self):
        self.response_frames = []
        self.transformed_frames = []

        # If all shapes of transformed frames are equal,
        # it is an np array of transformed frames.
        # Else it is a join of flattened arrays.
        self.np_array = None

    def make_np_array(self):
        raise NotImplementedError

    def to_flat_array(self):
        raise NotImplementedError


class Observation(object):
    def __init__(self):
        self.state_data = None
        self.sensor_data = None

    def to_flat_array(self):
        raise NotImplementedError


class RequestConfig(object):
    def __init__(self, request_data=None, provide_sensor=True,
            provide_state=False, raw_result=False):
        request_data = request_data or [
                (3, AirSimImageType.DepthPerspective, (84, 84))]
        self.request_data = request_data
        self.provide_sensor = provide_sensor
        self.provide_state = provide_state
        self.raw_result = raw_result

    def set_request_data(camera_ids, image_types,
            transforms=None, dimensions=(84, 84)):
        self.request_data = []
        for ind_cam, camera in camera_ids:
            for ind_type, image_type in image_types:
                reshape = dimensions
                if transforms is not None:
                    reshape = transforms[ind_type]
                self.request_data.append(
                        (camera, image_type, reshape) )


class EnvironmentConfig(object):

    def __init__(self):
        self.action_space = None
        self.reward_provider = None
        self.request_config = None
        self.forward_only = None

    def __init__(self, action_space, reward_provider,
            state_config, forward_only=False):
        self.action_space = action_space
        self.reward_provider = reward_provider
        self.state_config = state_config
        self.forward_only = forward_only


class QuadrotorEnvironment(Env):
    def __init__(self, config: EnvironmentConfig):
        self.action_space = config.action_space
        self.request_config = config.request_config
        self.reward_provider = config.reward_provider
        self.last_observation = None
        self.last_state = None
        self.forward_only = config.forward_only
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
        observation = self._get_observation()
        self._update_reward_provider()
        reward = self.reward_provider.compute_reward()
        done = self.reward_provider.is_done()
        return observation, reward, done, info

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        return

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    def _configure_action_space(self):
        raise NotImplementedError

    def _configure_reward_provider(self):
        raise NotImplementedError

    def _update_reward_provider(self):
        collision_info = self.client.getCollisionInfo()
        raise NotImplementedError

    def _get_sensor_data(self):
        sensed = SensorData()
        requests = []
        transforms = []
        for id_camera, scan_type, transform in \
                self.request_config.request_data:
            requests.append(ImageRequest(id_camera, scan_type,
                    True, False))
            transforms.append(transform)
        sensed.response_frames = client.simGetImages(requests)
        for answer, transform in zip(
                sensed.response_frames, transforms):
            sensed.transformed_frames.append(
                    transform_response(answer,
                        transform[0], transform[1]))
        sensed.make_np_array()
        return sensed

    def _get_state_data(self):
        state = StateData()
        state.quad_position = self.client.getPosition()
        state.quad_velocity = self.client.getVelocity()
        if self.request_config.provide_state:
            state.make_np_array()
        return state

    def _get_observation(self):
        observation = Observation()
        if self.request_config.provide_sensor:
            observation.sensor_data = self._get_sensor_data()
        state_data =  self._get_state_data()
        if self.request_config.provide_state:
            observation.state_data = state_data
        self.last_observation = observation
        self.last_state = state_data
        if self.request_config.raw_result:
            return observation.to_flat_array()
        return observation


