# coding: utf-8
"""
Entrypoint of command line utility
"""

import os
import sys
import json
import shutil
import argparse
from typing import Sequence

import nmt.common

from nmt.common import configuration, set_mode, make_logger, logger
from nmt.dataset import prepare_data, get_validation_dataset, get_test_dataset
from nmt.model import get_model_short_description, get_model_source_code_path
from nmt.train import train
from nmt.predict import predict, evaluate, get_vocabularies
from nmt.sanity import sanity_check
from nmt.visualization import visualization

def run_evaluate(get_dataset, log_prefix):
    dataset = get_dataset()
    evaluate(dataset, log_prefix)

def update_and_ensure_model_output_path(mode, index):
    model_short_description = get_model_short_description()
    model_source_code_path = get_model_source_code_path()
    model_configuration = configuration.ensure_submodule('model')

    @nmt.common.configured('data')
    def get_train_dataset_title(train_root_path: str = './data/train'):
        if os.path.isdir(train_root_path):
            result = train_root_path
        else:
            result = os.path.dirname(train_root_path)
        if result[-1] == '/':
            result = result[:-1]
        result = os.path.basename(result)
        if result == '.' or result == '':
            result = 'UNKNOWN'
        return result

    base_output_path = os.path.join(model_configuration.output_path, get_train_dataset_title())
    if model_short_description is not None:
        base_output_path = os.path.join(base_output_path, f'{model_short_description}')

    if mode != 'train':
        if not os.path.exists(base_output_path):
            raise IOError(f'Path `{base_output_path}` does not exist!')
        if index is None:
            index = 1
            while not os.path.exists(f'{base_output_path}/{index:03}'):
                index += 1
        model_configuration.output_path = f'{base_output_path}/{index:03}'
    else:
        index = 1
        while os.path.exists(f'{base_output_path}/{index:03}'):
            index += 1
        model_configuration.output_path = f'{base_output_path}/{index:03}'
        os.makedirs(model_configuration.output_path, exist_ok=True)
        shutil.copyfile(
            model_source_code_path,
            os.path.join(model_configuration.output_path, 'model.py')
        )
        shutil.copyfile(
            nmt.common.args.config_path,
            os.path.join(model_configuration.output_path, 'config.json')
        )

def main(cli_args: Sequence[str]):

    parser = argparse.ArgumentParser('nmt')
    parser.add_argument(
        '-m',
        '--mode',
        choices=[
            'save_config', 'prepare_data', 'train', 'test', 'translate',
            'sanity_check'
        ],
        help='Mode of execution.',
        required=True
    )
    parser.add_argument(
        '-x',
        '--index',
        type=int,
        help='Model index to be used in case of "test/translate" modes.',
        required=False
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
    parser.add_argument(
        '-t',
        '--type',
        type=str,
        help='Model type. Overrides configuration file.',
        required=False
    )
    parser.add_argument(
        '-v',
        '--visualize',
        action='store_true',
        help='Enable/disables visualization of model artifcats.',
        required=False
    )
    parser.add_argument(
        '-b',
        '--base_output_path',
        type=str,
        help='Forces base output path.',
        required=False
    )
    parser.add_argument(
        '-n',
        '--init_file_path',
        type=str,
        help='Pickle file to load initial values from',
        required=False
    )

    args = parser.parse_args(cli_args)
    nmt.common.args = args

    set_mode(args.mode)

    if args.mode != 'save_config':
        with open(args.config_path) as f:
            configuration.load(json.load(f))
        if args.type is not None:
            configuration.ensure_submodule('model').type = args.type
        if args.mode != 'translate' or not args.base_output_path:
            update_and_ensure_model_output_path(args.mode, args.index)
        else:
            configuration.ensure_submodule('model').output_path = args.base_output_path

        make_logger()

    if args.visualize:
        visualization.enabled = True

    if args.mode == 'save_config':
        with open(args.config_path, 'w') as f:
            json.dump(configuration.get_as_dict(), f, indent=4)
    elif args.mode == 'prepare_data':
        prepare_data()
    elif args.mode == 'train':
        train(args.init_file_path)
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