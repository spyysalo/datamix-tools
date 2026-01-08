#!/usr/bin/env python3

import sys
import json
import logging

from string import Template
from argparse import ArgumentParser


# JSON keys
COMMENT = 'comment'
VARIABLES = 'variables'
PROPORTION = 'proportion'
DATA = 'data'
MIXTURE = 'mixture'


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('mixture')
    ap.add_argument('paths')
    ap.add_argument('--output', help='Output filename')
    return ap.parse_args()


def interpolate_variables(value, variables):
    # Recursively apply Template.substitute(variables) to all string
    # values in JSON-like structure
    if isinstance(value, str):
        return Template(value).substitute(variables)
    elif isinstance(value, list):
        return [interpolate_variables(i, variables) for i in value]
    elif isinstance(value, dict):
        return {
            k: interpolate_variables(v, variables) for k, v in value.items()
        }
    else:
        return value


def remove_comments(value):
    if not isinstance(value, dict):
        return value
    else:
        return {
            k: remove_comments(v) for k, v in value.items() if k != COMMENT
        }


def load_json_with_variables(fn):
    with open(fn) as f:
        data = json.load(f)
    variables = data.pop(VARIABLES, {})
    return interpolate_variables(remove_comments(data), variables)


def validate_paths(paths):
    seen = set()
    for k, v in paths.items():
        if not isinstance(v, str):
            raise ValueError(f'expected string value for "{k}"')
        if v in seen:
            raise ValueError(f'duplicate value "{v}"')
        seen.add(v)


def validate_mixture(mixture, paths, label=None, names=None, ids=None):
    if label is None:
        label = 'top level mixture'
    if names is None:
        names = set()
    if ids is None:
        ids = set()

    proportions = []
    for k, v in mixture.items():
        if k in names:
            raise ValueError(f'duplicate name "{k}" in "{v}". names: {names}')
        names.add(k)
    
        if not isinstance(v, dict):
            raise ValueError(f'expected dict for "{k}"')

        if PROPORTION not in v:
            raise ValueError(f'missing "{PROPORTION}" for "{k}"')
        if not isinstance(v[PROPORTION], float):
            raise ValueError(f'expected float "{PROPORTION}" value for "{k}"')
        if not 0 < v[PROPORTION] <= 1.0:
            raise ValueError(f'expected 0 < "{PROPORTION}" <= 1 for "{k}"')
        proportions.append(v[PROPORTION])

        if MIXTURE in v:
            if not isinstance(v[MIXTURE], dict):
                raise ValueError(f'expected dict "{DATA}" value for "{k}"')
            if DATA in v:
                raise ValueError(f'both "{DATA}" and "{MIXTURE}" for "{k}"')
            validate_mixture(v[MIXTURE], paths, k, names, ids)
        elif DATA in v:
            if not isinstance(v[DATA], str):
                raise ValueError(f'expected string "{DATA}" value for "{k}"')
            if v[DATA] not in paths:
                raise ValueError(f'unknown data ID "{v[DATA]}" for "{k}"')
            if v[DATA] in ids:
                raise ValueError(f'duplicate reference to "{v[DATA]}"')
            ids.add(v[DATA])
        else:
            raise ValueError(f'neither "{DATA}" not "{MIXTURE}" for "{k}"')

    if round(sum(proportions), 10) != 1: # fixed to avoid floating point issues
        print(sum(proportions))
        raise ValueError(f'"{PROPORTION}" values do not add to 1 for {label}')


def flatten_mixture(mixture, parent_weight=1.0, flattened=None):
    if flattened is None:
        flattened = {}

    for k, v in mixture.items():
        weight = parent_weight * v[PROPORTION]
        if MIXTURE in v:
            flatten_mixture(v[MIXTURE], weight, flattened)
        else:
            assert DATA in v, f'missing {DATA}'
            flattened[k] = {
                PROPORTION: weight,
                DATA: v[DATA]
            }

    return flattened

def output_megatron_data_path(mixture, paths):
    flattened = flatten_mixture(mixture)
    items = list(flattened.items())
    
    precision = 6
    multiplier = 10**precision # scale weights to integers based on precision (6 decimal places)
    
    
    scaled_values = []
    for k, v in items:
        exact_val = v[PROPORTION] * multiplier
        floor_val = int(exact_val)
        remainder = exact_val - floor_val # calculate floor values and keep track of remainders
        scaled_values.append({
            'key': k,
            'path': paths[v[DATA]],
            'floor': floor_val,
            'remainder': remainder
        })
    
    # calculate how many 0.000001 are missing to reach 1.0
    total_floor_sum = sum(x['floor'] for x in scaled_values)
    diff = multiplier - total_floor_sum  # the number of 1e-6 units to distribute
    
    # Sort by remainder descending
    scaled_values.sort(key=lambda x: x['remainder'], reverse=True)
    
    for i in range(int(diff)):
        scaled_values[i]['floor'] += 1
        
    for item in scaled_values:
        final_proportion = item['floor'] / multiplier
        print(f"{final_proportion:.{precision}f} {item['path']}")

def save_megatron_data_path(mixture, paths, output_file):
    flattened = flatten_mixture(mixture)
    items = list(flattened.items())
    
    precision = 6
    multiplier = 10**precision
    
    # scale and calculate floor values
    processed_items = []
    for k, v in items:
        exact_val = v[PROPORTION] * multiplier
        floor_val = int(exact_val)
        remainder = exact_val - floor_val
        processed_items.append({
            'path': paths[v[DATA]],
            'floor': floor_val,
            'remainder': remainder
        })
    
    # fix of rounding errors (Largest Remainder Method)
    total_floor_sum = sum(x['floor'] for x in processed_items)
    diff = multiplier - total_floor_sum
    
    # sort by remainder descending to give more proportions to those closest to rounding up
    processed_items.sort(key=lambda x: x['remainder'], reverse=True)
    for i in range(int(diff)):
        processed_items[i]['floor'] += 1
        
    # all proportion-path pairs are joined by a space
    output_parts = []
    for item in processed_items:
        final_proportion = item['floor'] / multiplier
        output_parts.append(f"{final_proportion:.{precision}f} {item['path']}")
    
    single_line_output = " ".join(output_parts)

    with open(output_file, 'w') as f:
        f.write(single_line_output)
    
    print(f"The datapath: {output_file}")


def main():
    args = parse_args()

    logging.basicConfig(format='%(message)s')
    
    try:
        mixture = load_json_with_variables(args.mixture)
    except Exception as e:
        logging.error(f'error loading {args.mixture}: {e}')
        return 1

    try:
        paths = load_json_with_variables(args.paths)
    except Exception as e:
        logging.error(f'error loading {args.paths}: {e}')
        return 1

    try:
        validate_paths(paths)
    except Exception as e:
        logging.error(f'error validating {args.paths}: {e}')
        return 1

    try:
        validate_mixture(mixture, paths)
    except Exception as e:
        logging.error(f'error validating {args.mixture}: {e}')
        return 1

    try:
        if args.output:
            # If the user provided --output, save to file
            save_megatron_data_path(mixture, paths, args.output)
        else:
            # If --output is None, print/output to console
            output_megatron_data_path(mixture, paths)
    except Exception as e:
        logging.error(f'error processing output: {e}')
        return 1

if __name__ == '__main__':
    sys.exit(main())
