# coding: utf-8
"""
Script to evaluate all models trained on a given dataset
"""

import os
import math
import json
import re
import glob
import argparse

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

    labels = set()
    loss = []
    validation_loss = []
    validation_bleu = []

    for model_path in glob.glob(f'{args.base_path}/**/*'):
        if not os.path.isfile(f'{model_path}/train.log'):
            print(f'Warning! no `train.log` for {model_path}')
        loss_values = []
        loss_steps = []
        validation_loss_values = []
        validation_loss_steps = []
        validation_bleu_values = []
        validation_bleu_steps = []
        with open(f'{model_path}/train.log') as log_file:
            for l in log_file:
                m = re.match('.*Epoch_\d+\s+Step_(\d+):\s+loss=(.*?)\(.*$', l)
                if m:
                    loss_steps.append(float(m.group(1)))
                    loss_values.append(float(m.group(2)))
                else:
                    m = re.match('.*Epoch_\d+\s+Step_(\d+):\s+evaluation_loss=(.*?),.*$', l)
                    if m:
                        validation_loss_steps.append(float(m.group(1)))
                        validation_loss_values.append(float(m.group(2)))
                    else:
                        m = re.match('.*Epoch_\d+\s+Step_(\d+):\s+evaluation bleu=(.*?)\(.*$', l)
                        if m:
                            validation_bleu_steps.append(float(m.group(1)))
                            validation_bleu_values.append(float(m.group(2)))

        def normalize(x):
            return [None if math.isnan(e) else e for e in x]

        label = os.path.basename(os.path.dirname(model_path))
        if label in labels:
            index = 2
            while f'{label}_{index}' in labels:
                index += 1
        labels.add(label)
        if len(loss_values) > 100:
            loss.append({
                'label': label,
                'data': [{'x': x, 'y': y} for x, y in zip(normalize(loss_steps), normalize(loss_values))],
                'tension': 0.1
            })
        if len(validation_loss_values) > 2:
            validation_loss.append({
                'label': label,
                'data': [{'x': x, 'y': y} for x, y in zip(normalize(validation_loss_steps), normalize(validation_loss_values))],
                'tension': 0.1
            })
        if len(validation_bleu_values) > 2:
            validation_bleu.append({
                'label': label,
                'data': [{'x': x, 'y': y} for x, y in zip(normalize(validation_bleu_steps), normalize(validation_bleu_values))],
                'tension': 0.1
            })

    data_object_file_path = os.path.dirname(__file__)
    with open(f'{data_object_file_path}/data.json', 'w') as data_file:
        json.dump({'loss': loss, 'validation_loss': validation_loss, 'validation_bleu': validation_bleu}, data_file)

    print(f'Gathering data is done. Use `python3 -m http.server -d{data_object_file_path}` command to start serving GUI.')

if __name__ == '__main__':
    main()