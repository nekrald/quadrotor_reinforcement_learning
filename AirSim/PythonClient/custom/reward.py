import logging

import math
import numpy as np

from enum import Enum
from AirSimClient import AirSimImageType, ImageRequest
from custom.constants import RootConfigKeys, RewardConfigKeys


class RewardType(object):
    EXPLORATION_REWARD = "exploration"
    PATH_REWARD = "path"


class ExplorationReward(object):
    def __init__(self, client,
            collision_penalty=-200, height_penalty=-100,
            used_cams=[3], vehicle_rad=0.5, thresh_dist=3,
            goal_id=0, max_height = 40):
        self.collision_penalty = collision_penalty
        self.client = client
        self.used_cams = used_cams
        self.vehicle_rad = vehicle_rad
        self.tau_d = thresh_dist
        self.goal_id = goal_id
        self.max_height = max_height
        self.height_penalty = height_penalty

    def isDone(self, reward):
        done = 0
        if reward <= self.collision_penalty:
            done = 1
        return done

    def compute_reward(self, quad_state, quad_vel, collision_info):
        if collision_info.has_collided:
            reward = self.collision_penalty
        elif quad_state.z_val > self.max_height:
            reward = self.height_penalty
        else:
            client = self.client
            INF = 1e100
            min_depth_perspective = INF
            min_depth_vis = INF
            min_depth_planner = INF
            for camera_id in self.used_cams:
                requests = [
                    ImageRequest(camera_id, query, True, False)
                    for query in [
                        AirSimImageType.DepthPerspective,
                        AirSimImageType.DepthVis,
                        AirSimImageType.DepthPlanner,
                    ]]
                responses = client.simGetImages(requests)
                min_depth_perspective = min(min_depth_perspective,
                    np.min(np.array(responses[0].image_data_float)))
                min_depth_vis = min(min_depth_vis,
                    np.min(np.array(responses[1].image_data_float)))
                min_depth_planner = min(min_depth_planner,
                    np.min(np.array(responses[2].image_data_float)))
            goals = [min_depth_perspective, min_depth_vis,
                        min_depth_planner]
            logging.debug(
                "ExplorationReward: these are the goals = {}".format(
                    goals))
            dist = goals[self.goal_id]
            reward = (dist - self.vehicle_rad) / (
                    self.tau_d - self.vehicle_rad)
            logging.debug("ExplorationReward: before truncating" + \
                    " we have = {}".format(reward))
            reward = min(reward, 1)
            logging.debug("ExplorationReward: after truncation" + \
                    " we obtained {}".format(reward))
        return reward


class PathReward(object):

    def __init__(self, points=None,
            thresh_dist=7, beta=1,
            collision_penalty=-100,
            large_dist_penalty=-10,
            client=None):
        if points is None:
            points = [
               [-0.55265, -31.9786, -19.0225],
               [48.59735, -63.3286, -60.07256],
               [193.5974, -55.0786, -46.32256],
               [369.2474, 35.32137, -62.5725],
               [541.3474, 143.6714, -32.07256]
            ]
        self.points = list()
        for item in points:
            self.points.append(np.array(item))
        self.thresh_dist = thresh_dist
        self.beta = beta
        self.collision_penalty = collision_penalty
        self.large_dist_penalty = large_dist_penalty
        self.client = client

    def isDone(self, reward):
        done = 0
        if reward <= self.collision_penalty \
                or reward <= 3 * self.large_dist_penalty:
            done = 1
        return done

    def compute_reward(self, quad_state, quad_vel, collision_info):
        thresh_dist = self.thresh_dist
        beta = self.beta
        pts = self.points
        quad_pt = np.array(list((quad_state.x_val, quad_state.y_val,
            quad_state.z_val)))
        if collision_info.has_collided:
            reward = self.collision_penalty
        else:
            dist = 10000000
            for i in range(0, len(pts)-1):
                dist = min(dist, np.linalg.norm(
                    np.cross((quad_pt - pts[i]),
                    (quad_pt - pts[i+1]))) / np.linalg.norm(
                        pts[i]-pts[i+1]))
            if dist > thresh_dist:
                reward = self.large_dist_penalty
            else:
                reward_dist = (math.exp(-beta*dist) - 0.5)
                reward_speed = (np.linalg.norm([quad_vel.x_val,
                    quad_vel.y_val, quad_vel.z_val]) - 0.5)
                reward = reward_dist + reward_speed
        return reward


def make_reward(config, client):
    reward_config = config[RootConfigKeys.REWARD_CONFIG]
    reward_type = reward_config[RewardConfigKeys.REWARD_TYPE]
    collision_penalty = reward_config[
            RewardConfigKeys.COLLISION_PENALTY]
    thresh_dist = reward_config[
            RewardConfigKeys.THRESH_DIST]
    reward_type = reward_config[
            RewardConfigKeys.REWARD_TYPE]

    reward = None
    if reward_type == RewardType.EXPLORATION_REWARD:
        used_cams = reward_config[
                RewardConfigKeys.EXPLORE_USED_CAMS_LIST]
        vehicle_rad = reward_config[
                RewardConfigKeys.EXPLORE_VEHICLE_RAD]
        goal_id = reward_config[
                RewardConfigKeys.EXPLORE_GOAL_ID]
        max_height = reward_config[
                RewardConfigKeys.EXPLORE_MAX_HEIGHT]
        height_penalty = reward_config[
                RewardConfigKeys.EXPLORE_HEIGHT_PENALTY]
        reward = ExplorationReward(client,
            collision_penalty, height_penalty,
            used_cams, vehicle_rad, thresh_dist,
            goal_id, max_height)
    elif reward_type == RewardType.PATH_REWARD:
        points = reward_config[
                RewardConfigKeys.PATH_POINTS_LIST]
        beta = reward_config[
                RewardConfigKeys.PATH_BETA]
        dist_penalty = reward_config[
                RewardConfigKeys.PATH_LARGE_DIST_PENALTY]
        reward = PathReward(points, thresh_dist, beta,
            collision_penalty, dist_penalty, client)
    else:
        raise ValueError("Unknown reward type!")
    return reward

