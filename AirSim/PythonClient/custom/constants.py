import os


class RootConfigKeys(object):
    TRAIN_AFTER = "train_after"
    SLEEP_TIME = "sleep_time"
    INIT_X = "initX"
    INIT_Y = "initY"
    INIT_Z = "initZ"
    MOVE_DURATION = "move_duration"
    USE_FLAG_POS = "use_flag_position"
    ACTION_CONFIG = "action_config"
    REWARD_CONFIG = "reward_config"
    EPOCH_COUNT = "epoch_number"
    MAX_STEPS_MUL = "max_steps_mul"
    MEMORY_SIZE = "memory_size"
    TARGET_UPDATE_INTERVAL = "target_update_interval"
    TRAIN_INTERVAL = "train_interval"


class ActionConfigKeys(object):
    ACTION_SPACE_TYPE = "action_space_type"
    SCALING_FACTOR = "scaling_factor"
    GRID_SIZE = "grid_size"


class RewardConfigKeys(object):
    COLLISION_PENALTY = "collision_penalty"
    THRESH_DIST = "thresh_dist"
    REWARD_TYPE = "reward_type"

    EXPLORE_USED_CAMS_LIST = "used_cams"
    EXPLORE_VEHICLE_RAD = "vehicle_radius"
    EXPLORE_GOAL_ID = "goal_id"
    EXPLORE_MAX_HEIGHT = "max_height"
    EXPLORE_HEIGHT_PENALTY = "height_penalty"

    PATH_POINTS_LIST = "points_list"
    PATH_BETA = "path_beta"
    PATH_LARGE_DIST_PENALTY = "large_dist_penalty"

    LANDSCAPE_GOAL_POINT = "goal_point"


class RewardConstants:
    PATH_REWARD = "path_reward"
    EXPLORE_REWARD = "expore_reward"
    LANDSCAPE_REWARD = "landscape_reward"


class ActionConstants:
    DEFAULT_ACTION_SPACE = "default_action_space"
    GRID_ACTION_SPACE = "grid_action_space"

