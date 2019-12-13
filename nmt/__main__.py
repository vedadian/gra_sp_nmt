# coding: utf-8
"""
Entrypoint of command line utility
"""

import json
import argparse

from nmt.common import configuration, make_logger, logger
from nmt.dataset import get_train_dataset, prepare_data
from nmt.train import train
from nmt.predict import predict, evaluate

def main():
    parser = argparse.ArgumentParser('nmt')
    parser.add_argument(
        '-m',
        '--mode',
        choices=['prepare_data', 'train', 'test', 'translate'],
        help='Prepare corpora to be used in training.',
        required=True
    )
    parser.add_argument(
        '-c', '--config_path',
        type=str,
        help='Path to the JSON formatted config file.',
        required=True
    )

    args = parser.parse_args()

    with open(args.config_path) as f:
        configuration.load(json.load(f))
    make_logger()

    if args.mode == 'prepare_data':
        prepare_data()
    elif args.mode == 'train':
        train()
    elif args.mode == 'test':
        evaluate()
    elif args.mode == 'translate':
        predict()
    else:
        raise ValueError('Invalid execution mode.')

if __name__ == "__main__":
    main()