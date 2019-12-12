# coding: utf-8
"""
Entrypoint of command line utility
"""

import argparse

from nmt.common import configuration
from nmt.dataset import get_train_dataset

def main():
    parser = argparse.ArgumentParser('nmt')
    parser.add_argument(
        'mode',
        choices=['prepare_data'],
        help='Prepare corpora to be used in training.'
    )
    parser.add_argument(
        '-c', '--config_path',
        type=str,
        help='Path to the JSON formatted config file.'
    )

    args = parser.parse_args()

    print(args)

    if args.mode == 'prepare_data':
        print(configuration)
    else:
        raise ValueError('Invalid execution mode.')

if __name__ == "__main__":
    main()