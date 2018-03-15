#!/usr/bin/env python3
import sys
import os
import logging
import csv
import json
import copy
import shutil
import argparse

import numpy as np


file_dir = os.path.dirname(__file__)
upper_dir = os.path.realpath(os.path.join(file_dir, ".."))
sys.path.append(upper_dir)
del file_dir
del upper_dir

from AirSimClient import *

from action_space import DefaultActionSpace, \
    GridActionSpace, make_action, ActionSpaceType
from reward import PathReward, make_reward, RewardType
from replay_memory import ReplayMemory
from history import History
from deep_agent import DeepQAgent, huber_loss, transform_input
from exploration import LinearEpsilonAnnealingExplorer
from constants import RootConfigKeys, ActionConfigKeys, \
        RewardConfigKeys, RewardConstants, ActionConstants
from dqn_log import configure_logging
from config import make_default_root_config,\
        make_default_action_config, make_default_reward_config


def main(config, args):
    initX = config[RootConfigKeys.INIT_X]
    initY = config[RootConfigKeys.INIT_Y]
    initZ = config[RootConfigKeys.INIT_Z]

    # Connect to the AirSim simulator.
    client = MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    initial_position = client.getPosition()
    if config[RootConfigKeys.USE_FLAG_POS]:
        initX = initial_position.x_val
        initY = initial_position.y_val
        initZ = initial_position.z_val



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
        target_update_interval=update_interval, traindir_path=args.traindir,
        checkpoint_path=args.checkpoint)
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


def init_and_dump_configs():
    config_grid_path = make_default_root_config()
    config_grid_explore = make_default_root_config()

    config_default_path = make_default_root_config()
    config_default_explore = make_default_root_config()

    reward_explore = make_default_reward_config(
            RewardType.EXPLORATION_REWARD)
    reward_path = make_default_reward_config(
            RewardType.PATH_REWARD)

    action_grid = make_default_action_config(
            ActionSpaceType.GRID_SPACE)
    action_default = make_default_action_config(
            ActionSpaceType.DEFAULT_SPACE)

    for item in [config_grid_path, config_default_path]:
        item[RootConfigKeys.REWARD_CONFIG] = reward_path
    for item in [config_grid_explore, config_default_explore]:
        item[RootConfigKeys.REWARD_CONFIG] = reward_explore
    for item in [config_grid_path, config_grid_explore]:
        item[RootConfigKeys.ACTION_CONFIG] = action_grid
    for item in [config_default_path, config_default_explore]:
        item[RootConfigKeys.ACTION_CONFIG] = action_default

    names = ["grid_path", "default_path", \
            "grid_explore", "default_explore"]
    items = [config_grid_path, config_default_path, \
            config_grid_explore, config_default_explore]

    if not os.path.exists("config-example"):
        os.makedirs("config-example")
    for config, name in zip(items, names):
        with open("config-example/" + name + ".json", "w") as f:
            json.dump(config, f, indent=4)

def parse_arguments():
    parser = argparse.ArgumentParser(description='DQNdrone for AirSim')
    parser.add_argument('config', metavar='CONFIG', type=str, help='path-to-file-with-config')
    parser.add_argument('--traindir', default='traindir', type=str, metavar='DIR', help='path-to-traindir')
    parser.add_argument('--checkpoint', default=None, type=str, metavar='DNN', help='path-to-checkpoint')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists(args.traindir):
        os.makedirs(args.traindir)
    shutil.copy(args.config, os.path.join(os.path.realpath(args.traindir), "config.json"))
    with open(args.config, "r") as fconf:
        config = json.load(fconf)
    configure_logging(os.path.join(args.traindir, "main.log"))
    main(config, args)

