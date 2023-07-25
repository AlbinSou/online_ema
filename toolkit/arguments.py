#!/usr/bin/env python3
import argparse
import yaml

""" Functions to manage config files """

def save_config(config: argparse.Namespace, filename: str):
   with open(filename,'w') as f:
       f.write('!!python/object:argparse.Namespace\n')
       yaml.dump(vars(config), f)

def set_config_args(parser: argparse.ArgumentParser):
    """ 
    From an existing parser, parses the config file given as --config argument, 
    returns the modified parser with overriden default values for the arguments,
    the new defaults are the ones given in the config file
    """
    # Parses config file to set default arguments
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', '-c', help='config_file')
    args, remaining_argv = config_parser.parse_known_args()
    defaults = {}
    if args.config:
        with open(args.config) as f:
            file_args = yaml.load(f, Loader=yaml.FullLoader)
        defaults.update(vars(file_args))
    parser.set_defaults(**defaults)
    return parser
