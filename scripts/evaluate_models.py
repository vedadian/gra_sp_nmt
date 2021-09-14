# coding: utf-8
"""
Script to evaluate all models trained on a given dataset
"""

import os
import sys
import json
import glob
import argparse
from importlib import util as iu

try:
    nmt_module_path = '.'
    import nmt
except:
    sys.path.insert(0, '..')
    nmt_module_path = '..'
    import nmt

def main():

    parser = argparse.ArgumentParser('nmt')
    parser.add_argument(
        '-b',
        '--base_path',
        help='Base path to trained models',
        required=True
    )
    parser.add_argument(
        '-c',
        '--config_path',
        help='Path to the base config file. Model type will be overrided.',
        required=False
    )

    args = parser.parse_args()

    MODEL_KEY = 'nmt.mtmodel'
    MODEL_DICT = {}
    def load_model_module(file_path: str, short_description: str = None, store_in_model_dict: bool = True):
        if (file_path not in MODEL_DICT) or not store_in_model_dict:
            module_spec = iu.spec_from_file_location(MODEL_KEY, file_path)
            if module_spec is None:
                raise Exception(
                    f'Error loading `{file_path}` file.'
                )
            model_module = iu.module_from_spec(module_spec)
            module_spec.loader.exec_module(model_module)
            if hasattr(model_module.Model, 'short_description'):
                short_description = model_module.Model.short_description()
            else:
                print(f'Warning! Model in {file_path} has no `short_description`.')
            if short_description is None:
                print(f'Warning! Populated `short_description` is `None`.')
                short_description, _ = os.path.splitext(os.path.basename(file_path))
            print(f'Registering {short_description} <= {file_path}')
            if not store_in_model_dict:
                return model_module
            MODEL_DICT[file_path] = (short_description, model_module)

    def gather_available_models():
        for file_path in glob.glob(f'{nmt_module_path}/nmt/models/**/*.py'):
            load_model_module(file_path)
    gather_available_models()

    for model_path in glob.glob(f'{args.base_path}/**/*'):
        print(f'Model in {model_path}')
        print('***********************************************************')

        model_file_path = None
        config_file_path = None

        for file_path in glob.glob(f'{model_path}/*.py'):
            model_file_path = file_path
            break
        for file_path in glob.glob(f'{model_path}/*.json'):
            config_file_path = file_path
            break

        if config_file_path is None:
            config_file_path = args.config_path

        if not config_file_path:
            print(f'Warning! No configuration could be inferred.')
            continue

        if model_file_path is None:
            path_parts = model_path.split('/')
            index = None
            try:
                index = int(path_parts[-1])
                model_type = path_parts[-2]
            except:
                model_type = path_parts[-1]
            for file_path, (short_description, _) in MODEL_DICT.items():
                if short_description == model_type:
                    model_file_path = file_path
                    break
            if model_file_path is None:
                for file_path, (short_description, _) in MODEL_DICT.items():
                    if model_type in file_path:
                        model_file_path = file_path
                        break
        if model_file_path is None:
            print(f'Skipping `{model_type}`, model implementation not found.')
            continue

        if model_file_path in MODEL_DICT:
            sys.modules[MODEL_KEY] = MODEL_DICT[model_file_path][-1]
        else:
            sys.modules[MODEL_KEY] = load_model_module(model_file_path, None, False)

        with open(config_file_path) as f:
            configuration = json.load(f)

        input_file_path = f"{configuration['data']['test_root_path']}.{configuration['data']['src_lang_code']}"
        output_file_path = os.path.join(model_path, os.path.basename(configuration['data']['test_root_path']))
        output_file_path = f"{output_file_path}.hypo.{configuration['data']['tgt_lang_code']}"

        print(f'BASE CONFIG. FILE: {config_file_path}')
        print(f'INPUT FILE: {input_file_path}')
        print(f'OUTPUT FILE: {output_file_path}')

        nmt.cli_interface.main([
            '-m',
            'translate',
            '-c',
            config_file_path,
            '-i',
            input_file_path,
            '-o',
            output_file_path,
            '-t',
            'mtmodel',
            '-b',
            model_path,
        ])
if __name__ == '__main__':
    main()