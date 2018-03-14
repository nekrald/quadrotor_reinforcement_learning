import os

class RootConfigKeys(object):
    TRAIN_AFTER = "train_after"
    SLEEP_TIME = "sleep_time"
    INIT_X = "initX"
    INIT_Y = "initY"
    INIT_Z = "initZ"
    USE_FLAG_POS = "use_flag_position"
    ACTION_SPACE_TYPE = "action_space_type"
    REWARD_TYPE = "reward_type"


class ActionConfigKeys(object):
    SCALING_FACTOR = "scaling_factor"
    GRID_SIZE = "grid_size"


class RewardConfigKeys(object):
    COLLISION_PENALTY = "collision_penalty"
    THRESH_DIST = "thresh_dist"

    EXPLORE_USED_CAMS_LIST = "used_cams"
    EXPLORE_ VEHICLE_RAD = "vehicle_radius"
    EXPLORE_GOAL_ID = "goal_id"
    EXPLORE_MAX_HEIGHT = "max_height"
    EXPLORE_HEIGHT_PENALTY = "height_penalty"

    PATH_POINTS_LIST = "points_list"
    PATH_BETA = "path_beta"
    PATH_LARGE_DIST_PENALTY = "large_dist_penalty"


class RewardConstants:
    PATH_REWARD = "path_reward"
    EXPLORE_REWARD = "expore_reward"


class ActionConstants:
    DEFAULT_ACTION_SPACE = "default_action_space"
    GRID_ACTION_SPACE = "grid_action_space"

