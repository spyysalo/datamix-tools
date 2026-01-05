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
    ap.add_argument('--output', default='datapath.txt', help='Output filename')
    return ap.parse_args()

def interpolate_variables(value, variables):
    if isinstance(value, str):
        return Template(value).substitute(variables)
    elif isinstance(value, list):
        return [interpolate_variables(i, variables) for i in value]
    elif isinstance(value, dict):
        return {k: interpolate_variables(v, variables) for k, v in value.items()}
    else:
        return value

def remove_comments(value):
    if not isinstance(value, dict):
        return value
    return {k: remove_comments(v) for k, v in value.items() if k != COMMENT}

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
    if label is None: label = 'top level mixture'
    if names is None: names = set()
    if ids is None: ids = set()

    proportions = []
    for k, v in mixture.items():
        if k in names:
            raise ValueError(f'duplicate name "{k}" in "{v}"')
        names.add(k)
    
        if not isinstance(v, dict):
            raise ValueError(f'expected dict for "{k}"')

        if PROPORTION not in v:
            raise ValueError(f'missing "{PROPORTION}" for "{k}"')
        if not isinstance(v[PROPORTION], (float, int)):
            raise ValueError(f'expected numeric "{PROPORTION}" for "{k}"')
        proportions.append(v[PROPORTION])

        if MIXTURE in v:
            validate_mixture(v[MIXTURE], paths, k, names, ids)
        elif DATA in v:
            if v[DATA] not in paths:
                raise ValueError(f'unknown data ID "{v[DATA]}" for "{k}"')
            ids.add(v[DATA])
        else:
            raise ValueError(f'neither "{DATA}" nor "{MIXTURE}" for "{k}"')

    if round(sum(proportions), 10) != 1:
        raise ValueError(f'"{PROPORTION}" values do not add to 1 for {label}')

def flatten_mixture(mixture, parent_weight=1.0, flattened=None):
    if flattened is None:
        flattened = {}
    for k, v in mixture.items():
        weight = parent_weight * v[PROPORTION]
        if MIXTURE in v:
            flatten_mixture(v[MIXTURE], weight, flattened)
        else:
            flattened[k] = {PROPORTION: weight, DATA: v[DATA]}
    return flattened

def save_megatron_data_path(mixture, paths, output_file):
    """
    Largest Remainder Method is applied to ensure 6-decimal precision
    and an exact sum of 1.0, then saves as a SINGLE LINE to a file.
    """
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
        paths = load_json_with_variables(args.paths)
        validate_paths(paths)
        validate_mixture(mixture, paths)
        
        save_megatron_data_path(mixture, paths, args.output)
        
    except Exception as e:
        logging.error(f'Error: {e}')
        return 1

if __name__ == '__main__':
    sys.exit(main())