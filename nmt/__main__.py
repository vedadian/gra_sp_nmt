# coding: utf-8
"""
Entrypoint of command line utility
"""

import json
import argparse

import nmt.model
import nmt.optimization

from nmt.common import configuration, make_logger, logger
from nmt.dataset import prepare_data, get_validation_dataset, get_test_dataset
from nmt.train import train
from nmt.predict import predict, evaluate
from nmt.sanity import sanity_check

def run_evaluate(get_dataset, log_prefix):
    dataset = get_dataset()
    evaluate(dataset, log_prefix)

def main():

    parser = argparse.ArgumentParser('nmt')
    parser.add_argument(
        '-m',
        '--mode',
        choices=[
            'save_config', 'prepare_data', 'train', 'test', 'translate',
            'sanity_check'
        ],
        help='Prepare corpora to be used in training.',
        required=True
    )
    parser.add_argument(
        '-c',
        '--config_path',
        type=str,
        help='Path to the JSON formatted config file.',
        required=True
    )
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help='Path to input text file.',
        required=False
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        help='Path to output hypotheses file.',
        required=False
    )

    args = parser.parse_args()

    if args.mode != 'save_config':
        with open(args.config_path) as f:
            configuration.load(json.load(f))
        make_logger()

    if args.mode == 'save_config':
        with open(args.config_path, 'w') as f:
            json.dump(configuration.get_as_dict(), f, indent=4)
    elif args.mode == 'prepare_data':
        prepare_data()
    elif args.mode == 'train':
        train()
    elif args.mode == 'test':
        run_evaluate(get_validation_dataset, 'validation')
        run_evaluate(get_test_dataset, 'test')
    elif args.mode == 'translate':
        if not args.input or not args.output:
            raise ValueError('In order to use translate command, both input and output path must be specified.')
        predict(args.input, args.output, 'translate')
    elif args.mode == 'sanity_check':
        sanity_check()
    else:
        raise ValueError('Invalid execution mode.')


if __name__ == "__main__":
    main()