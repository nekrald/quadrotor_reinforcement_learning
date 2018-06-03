from dqn.train import DQNTrainScheduler
from reinforce.train import REINFORCETrainScheduler
from constants import Approaches, RootConfigKeys


def make_scheduler(config, args):
    approach = config[RootConfigKeys.APPROACH]
    if approach == Approaches.DQN_CNTK:
        return DQNTrainScheduler(config, args)
    elif approach == Approaches.REINFORCE_PYTORCH:
        return REINFORCETrainScheduler(config, args)
    else:
        raise ValueError("Unknown approach!")



