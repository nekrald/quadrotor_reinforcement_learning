import sys
import os
import logging

file_dir = os.path.dirname(__file__)
upper_dir = os.path.realpath(os.path.join(file_dir, ".."))
sys.path.append(upper_dir)
del file_dir
del upper_dir

from AirSimClient import *

from action_space import DefaultActionSpace, GridActionSpace
from reward import PathReward
from replay_memory import ReplayMemory
from history import History
from agent import DeepQAgent, huber_loss, transform_input
from exploration import LinearEpsilonAnnealingExplorer
from constants import RootConfigKeys, ActionConfigKeys, \
        RewardConfigKeys, RewardConstants, ActionConstants
from dqn_log import configure_logging

from argparse import ArgumentParser

import numpy as np
from cntk.core import Value
from cntk.initializer import he_uniform
from cntk.layers import Sequential, Convolution2D, Dense, default_options
from cntk.layers.typing import Signature, Tensor
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType
from cntk.logging import TensorBoardProgressWriter
from cntk.ops import abs, argmax, element_select, less, relu, reduce_max, reduce_sum, square
from cntk.ops.functions import CloneMethod, Function
from cntk.train import Trainer

import csv
import json


def main(config, args):
    initX = config[RootConfigKeys.INIT_X]
    initY = config[RootConfigKeys.INIT_Y]
    initZ = config[RootConfigKeys.INIT_Z]

    initial_position = client.getPosition()
    if config[RootConfigKeys.USE_FLAG_POS]:
        initX = initial_position.x_val
        initY = initial_position.y_val
        initZ = initial_position.z_val

    # Connect to the AirSim simulator.
    client = MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    client.goHome()
    client.takeoff()
    client.moveToPosition(initX, initY, initZ, 5)
    time.sleep(config[RootConfigKeys.SLEEP_TIME])

    # Train
    epoch = config[RootConfigKeys.EPOCH_COUNT]
    max_steps = epoch * config[RootConfigKeys.MAX_STEPS_MUL]
    current_step = 0

    responses = client.simGetImages(
        [ImageRequest(3, AirSimImageType.DepthPerspective,
        True, False)])
    current_state = transform_input(responses)

    reward_processor = make_reward(config, client)
    action_processor = make_action(config)

    # Make RL agent
    NumBufferFrames = 4
    SizeRows = 84
    SizeCols = 84
    NumActions = action_processor.get_num_actions()
    train_after = config[RootConfigKeys.TRAIN_AFTER]
    memory_size = config[RootConfigKeys.MEMORY_SIZE]
    update_interval = config[
            RootConfigKeys.TARGET_UPDATE_INTERVAL]
    train_interval = config[
            RootConfigKeys.TRAIN_INTERVAL]
    agent = DeepQAgent((NumBufferFrames, SizeRows, SizeCols),
        NumActions, monitor=True, train_after=train_after,
        memory_size=memory_size, train_interval=train_interval,
        target_update_interval=update_interval)
    move_duration = config[RootConfigKeys.MOVE_DURATION]

    while current_step < max_steps:
        action = agent.act(current_state)
        quad_offset = action_processor.interpret_action(action)
        quad_vel = client.getVelocity()
        client.moveByVelocity(
            quad_vel.x_val+quad_offset[0],
            quad_vel.y_val+quad_offset[1],
            quad_vel.z_val+quad_offset[2], move_duration)
        time.sleep(config[RootConfigKeys.SLEEP_TIME])

        quad_state = client.getPosition()
        quad_vel = client.getVelocity()
        collision_info = client.getCollisionInfo()

        reward = reward_processor.compute_reward(
                quad_state, quad_vel, collision_info)
        done = reward_processor.isDone(reward)
        logging.info('Action, Reward, Done: {} {} {}'.format(
            action, reward, done))

        agent.observe(current_state, action, reward, done)
        agent.train()

        if done:
            client.goHome()
            client.takeoff()
            client.moveToPosition(initX, initY, initZ, 5)
            time.sleep(config[RootConfigKeys.SLEEP_TIME])
        current_step +=1

        responses = client.simGetImages([ImageRequest(3,
            AirSimImageType.DepthPerspective, True, False)])
        current_state = transform_input(responses)


if __name__ == "__main__":
    default_config = {
                "train_after" : 1000,
                "sleep_time"  : 0.1,
                "initX" : -.55265,
                "initY" : -31.9786,
                "initZ" : -19.0225,
                "use_flag_position": False,
                "action_space_type": "w"
    }
    with open("drone_config.json", "w") as f:
        json.dump(default_config, f)
    config = default_config
#    with open("drone_config.json", "r") as f:
#        config = json.load(f)
    print(config)
    quit()
    main()
