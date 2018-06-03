#!/usr/bin/env python3
import argparse

from common.log import configure_logging
from scheduler import make_scheduler


def parse_arguments():
    parser = argparse.ArgumentParser(
            description='Reinforcement learning for AirSim')
    parser.add_argument('config', metavar='CONFIG', type=str,
            help='path-to-file-with-config')
    parser.add_argument('--traindir', default='traindir',
            type=str, metavar='DIR', help='path-to-traindir')
    parser.add_argument('--checkpoint', default=None, type=str,
            metavar='NN', help='path-to-checkpoint')
    parser.add_argument('--max-flight-steps', default=2500,
            metavar='DURATION', type=int)
    args = parser.parse_args()
    return args


def main(config, args):
    scheduler = make_scheduler(config, args)
    scheduler.process_problem()


if __name__ == "__main__":
    args = parse_arguments()
    if not os.path.exists(args.traindir):
        os.makedirs(args.traindir)
    configure_logging(args.traindir)
    shutil.copy(args.config,
            os.path.join(os.path.realpath(args.traindir),
                "config.json"))
    with open(args.config, "r") as fconf:
        config = json.load(fconf)
    main(config, args)

