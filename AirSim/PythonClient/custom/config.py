import json

from custom.action_space import DefaultActionSpace, \
    GridActionSpace, ActionSpaceType, make_action
from custom.reward import PathReward, RewardType, make_reward
from custom.constants import RootConfigKeys, ActionConfigKeys, \
        RewardConfigKeys, RewardConstants, ActionConstants


def make_default_root_config():
    root_config = dict()
    root_config[RootConfigKeys.TRAIN_AFTER] = 1000
    root_config[RootConfigKeys.SLEEP_TIME] = 0.1
    root_config[RootConfigKeys.INIT_X] = -0.55265
    root_config[RootConfigKeys.INIT_Y] = -31.9786
    root_config[RootConfigKeys.INIT_Z] = -19.0225
    root_config[RootConfigKeys.MOVE_DURATION] = 4
    root_config[RootConfigKeys.USE_FLAG_POS] = True
    root_config[RootConfigKeys.EPOCH_COUNT] = 100
    root_config[RootConfigKeys.MAX_STEPS_MUL] = 10000
    root_config[RootConfigKeys.MEMORY_SIZE] = 5000
    root_config[RootConfigKeys.TARGET_UPDATE_INTERVAL] = 50000
    root_config[RootConfigKeys.TRAIN_INTERVAL] = 4
    root_config[RootConfigKeys.ACTION_CONFIG] = {}
    root_config[RootConfigKeys.REWARD_CONFIG] = {}
    return root_config


def make_default_action_config(action_type):
    action_config = dict()
    action_config[ActionConfigKeys.ACTION_SPACE_TYPE] = action_type
    action_config[ActionConfigKeys.SCALING_FACTOR] = 0.25
    if action_type  == ActionSpaceType.GRID_SPACE:
        pass
    elif action_type == ActionSpaceType.DEFAULT_SPACE:
        action_config[ActionConfigKeys.GRID_SIZE] = 4
    else:
        raise ValueError("unknown action type")
    return action_config


def make_default_reward_config(reward_type):
    reward_config = dict()
    reward_config[RewardConfigKeys.COLLISION_PENALTY] = -1000
    reward_config[RewardConfigKeys.THRESH_DIST] = 7
    reward_config[RewardConfigKeys.REWARD_TYPE] = reward_type
    if reward_type == RewardType.EXPLORATION_REWARD:
        reward_config[RewardConfigKeys.EXPLORE_USED_CAMS_LIST] = [3]
        reward_config[RewardConfigKeys.EXPLORE_VEHICLE_RAD] = 7
        reward_config[RewardConfigKeys.EXPLORE_GOAL_ID] = 0
        reward_config[RewardConfigKeys.EXPLORE_MAX_HEIGHT] = 100
        reward_config[RewardConfigKeys.EXPLORE_HEIGHT_PENALTY] = -200
    elif reward_type == RewardType.PATH_REWARD:
        reward_config[RewardConfigKeys.PATH_BETA] = 1.0
        reward_config[RewardConfigKeys.PATH_POINTS_LIST] = None
        reward_config[RewardConfigKeys.PATH_LARGE_DIST_PENALTY] = -10
    else:
        raise ValueError("unknown reward type")
    return reward_config


