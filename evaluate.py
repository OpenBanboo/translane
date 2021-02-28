#!/usr/bin/env python
# Modified based on DETR (https://github.com/facebookresearch/detr)
# and LSTR (https://github.com/liuruijin17/LSTR)
# by Fang Lin (flin4@stanford.edu)

import os
import json
import torch
import pprint
import argparse
import importlib
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")

from configuration import setup_configurations
from factory.network_builder import NetworkFactory
from database.datasets import datasets
from database.utils.evaluator import Evaluator
from util.general_utils import create_directories
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained TransLane Network")
    parser.add_argument("configuration", help="Configuration File", type=str)
    parser.add_argument("-b", dest="begin_iteration",
                        help="Test using saved weights at begging of iteration",
                        default=None, type=int)
    parser.add_argument("-s", dest="split",
                        help="Database split for training, validation, and test",
                        default="validation", type=str)
    parser.add_argument("-c", dest="customized_image_path",
                        default=None, type=str)
    parser.add_argument("-m", dest="mode",
                        default=None, type=str)
    parser.add_argument("--suffix", dest="suffix", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch", dest='batch',
                        help="Used to tune the performance on device",
                        default=1, type=int)
    parser.add_argument("--debugEnc", action="store_true")
    parser.add_argument("--debugDec", action="store_true")
    args = parser.parse_args()
    return args

def evaluate(database, split, begin_iteration,
         debug=False, suffix=None, mode=None, customized_image_path=None, batch=1,
         debugEnc=False, debugDec=False):

    print(database, split, begin_iteration, debug, suffix, mode, customized_image_path, batch,
          debugEnc, debugDec)

    result_dir = setup_configurations.result_dir
    result_dir = os.path.join(result_dir, str(begin_iteration), split)

    # Append suffix to the result directory
    if suffix is not None:
        result_dir = os.path.join(result_dir, suffix)
    # Create the result directory if it does not existed.
    create_directories([result_dir])
    # Use the max iteration defined in the configuration file if -b is not assigned
    saved_weights_iteration = setup_configurations.max_iter if begin_iteration is None else begin_iteration
    print("Using the parameters saved at iteration: {}".format(saved_weights_iteration))

    print("Start building the model...")
    model = NetworkFactory()

    print("Loading parameters to the built model...")
    model.load_params(saved_weights_iteration)
    # Put it on cuda if existed.
    model.cuda()
    model.eval_mode()

    print("Initializing the evaluator...")
    evaluator = Evaluator(database, result_dir)

    if mode == 'eval':
        print('Testing the statistics of the saved model...')
        validation_file = "test.tusimple"
        testing = importlib.import_module(validation_file).testing
        testing(database, model, result_dir,
                debug=debug, evaluator=evaluator, repeat=batch,
                debugEnc=debugEnc, debugDec=debugDec)

    elif mode == 'customized':
        if customized_image_path == None:
            raise ValueError('-c customized_image_path is not defined!')
        print("Processing customized [images]...")
        validation_file = "test.images"
        image_testing = importlib.import_module(validation_file).testing
        image_testing(database, model, customized_image_path,
                      debug=debug, evaluator=None)

    else:
        raise ValueError('-m must be chosen from eval/customized')

if __name__ == "__main__":
    args = parse_args()

    if args.suffix is None:
        configuration = os.path.join(setup_configurations.config_dir, args.configuration + ".json")
    else:
        configuration = os.path.join(setup_configurations.config_dir, args.configuration + "-{}.json".format(args.suffix))
    print("configuration: {}".format(configuration))

    with open(configuration, "r") as f:
        configs = json.load(f)
            
    configs["system"]["snapshot_name"] = args.configuration
    setup_configurations.update_config(configs["system"])

    train_data = setup_configurations.train_split
    val_data   = setup_configurations.val_split
    test_data  = setup_configurations.test_split

    # Data split dictionary
    data_split = {
        "training": train_data,
        "validation": val_data,
        "testing": test_data
    }[args.split]

    print("loading all datasets...")
    dataset = setup_configurations.dataset
    print("=========={}".format(dataset))
    print("Dataset Split: {}".format(data_split))  # test

    testing_database = datasets[dataset](configs["db"], data_split)

    evaluate(testing_database,
         args.split,
         args.begin_iteration,
         args.debug,
         args.suffix,
         args.mode,
         args.customized_image_path,
         args.batch,
         args.debugEnc,
         args.debugDec,)
