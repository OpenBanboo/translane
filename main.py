#!/usr/bin/env python
# Modified based on DETR (https://github.com/facebookresearch/detr)
# and LSTR (https://github.com/liuruijin17/LSTR)
# by Fang Lin (flin4@stanford.edu)

import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import queue
import numpy as np
import torch
import importlib
import threading
import util.misc as utils
from tqdm import tqdm
from util import stdout_to_tqdm
from configuration import setup_configurations
from factory.network_builder import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
from database.datasets import datasets
import models.py_utils.misc as utils
from util.general_utils import create_directories, pin_memory, start_multi_tasks, prefetch_data

def parse_args():
    parser = argparse.ArgumentParser(description="Train TransLane Network")
    # Model parameters
    parser.add_argument("configuration", help="Configuration File", type=str)
    # The beginning iteration for training
    parser.add_argument("-b", dest="begin_iteration",
                        help="Begin to train at iteration b using pretrained model",
                        default=0, type=int)
    # The number of cpu threads used for loading data in order to achieve minimal
    parser.add_argument("-t", dest="num_threads", default=6, type=int)
    parser.add_argument("--freeze", action="store_true")

    args = parser.parse_args()
    return args

def train(train_databases, validation_database, begin_iteration=0, freeze=False):
    lr    = setup_configurations.lr
    end_iter    = setup_configurations.end_iter
    pretrained_model = setup_configurations.pretrain
    snapshot         = setup_configurations.snapshot
    val_iter         = setup_configurations.val_iter
    display          = setup_configurations.display
    decay_rate       = setup_configurations.decay_rate
    stepsize         = setup_configurations.stepsize
    batch_size       = setup_configurations.batch_size

    # fetch the size of train and validation
    training_size   = len(train_databases[0].db_inds)
    validation_size = len(validation_database.db_inds)

    # Create queues prefecthing data for training
    training_queue   = Queue(setup_configurations.prefetch_size)
    validation_queue = Queue(5)

    # Create queues saving fixed data for training
    fixed_training_queue   = queue.Queue(setup_configurations.prefetch_size)
    fixed_validation_queue = queue.Queue(5)

    # load data sampling function
    data_file   = "sample.{}".format(train_databases[0].data) # "sample.coco"
    sample_data = importlib.import_module(data_file).sample_data

    # Start threads for parallel data processing
    process_train = start_multi_tasks(train_databases, training_queue, sample_data)
    if val_iter:
        process_validation = start_multi_tasks([validation_database],
                                               validation_queue,
                                               sample_data)

    # Using multi-threading to accelerating the training speed
    fixed_train_semaphore   = threading.Semaphore()
    fixed_validation_semaphore = threading.Semaphore()
    fixed_train_semaphore.acquire()
    fixed_validation_semaphore.acquire()

    fixed_train_args   = (training_queue, fixed_training_queue, fixed_train_semaphore)
    fixed_train_thread = threading.Thread(target=pin_memory,
                                          args=fixed_train_args)
    fixed_train_thread.daemon = True
    fixed_train_thread.start()

    fixed_validation_args   = (validation_queue, fixed_validation_queue,
                               fixed_validation_semaphore)
    fixed_validation_thread = threading.Thread(target=pin_memory,
                                               args=fixed_validation_args)
    fixed_validation_thread.daemon = True
    fixed_validation_thread.start()

    # Build the network from factory
    model = NetworkFactory(flag=True)

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("Could not find the pre-trained model!")
        model.load_pretrained_params(pretrained_model)

    if begin_iteration:
        # Using learning rate decay
        lr /= (decay_rate ** (begin_iteration // stepsize))
        model.load_params(begin_iteration)
        model.set_lr(lr)
        print("Begins training from iteration {} with lr {}".format(begin_iteration + 1, decay_rate))
    else:
        model.set_lr(lr)

    # Put model on GPU for acceleration
    model.cuda()
    model.train_mode()
    header = None
    # Use MetricLogger from DETR
    stdout_writer = utils.MetricLogger(delimiter="  ")
    stdout_writer.add_meter('lr', utils.SmoothedValue(window_size=1,
                                                      fmt='{value:.6f}'))
    stdout_writer.add_meter('class_error', utils.SmoothedValue(window_size=1,
                                                               fmt='{value:.2f}'))

    with stdout_to_tqdm() as output_file:
        for iteration in stdout_writer.log_every(tqdm(range(begin_iteration + 1,
                                                            end_iter + 1),
                                                      file=output_file,
                                                      ncols=67),
                                                 print_freq=25,
                                                 header=header):

            training = fixed_training_queue.get(block=True)
            viz_split = 'train'
            save = True if (display and iteration % display == 0) else False
            # Start training the model
            (set_loss, loss_dict) \
                = model.train(iteration, save, viz_split, **training)
            (loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, loss_value) = loss_dict
            stdout_writer.update(loss=loss_value,
                                 **loss_dict_reduced_scaled,
                                 **loss_dict_reduced_unscaled)
            stdout_writer.update(class_error=loss_dict_reduced['class_error'])
            stdout_writer.update(lr=lr)

            # Remove the set_loss
            del set_loss

            # Validation Process
            if val_iter and validation_database.db_inds.size and iteration % val_iter == 0:
                model.eval_mode()
                viz_split = 'val'
                save = True # Default Ture
                validation = fixed_validation_queue.get(block=True)
                # Start validating
                (val_set_loss, val_loss_dict) \
                    = model.validate(iteration, save, viz_split, **validation)

                (loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, loss_value) \
                    = val_loss_dict
                print('[Validation: ]\t[Storing images of train and validation]')
                stdout_writer.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
                stdout_writer.update(class_error=loss_dict_reduced['class_error'])
                stdout_writer.update(lr=lr)
                model.train_mode()

            if iteration % snapshot == 0:
                model.save_params(iteration)

            if iteration % stepsize == 0:
                lr /= decay_rate
                model.set_lr(lr)

            if iteration % (training_size // batch_size) == 0:
                stdout_writer.synchronize_between_processes()
                print("Averaged stats:", stdout_writer)


    # End the threads
    fixed_train_semaphore.release()
    fixed_validation_semaphore.release()

    # End the data fetching tasks
    for t in process_train:
        t.terminate()
    for t in process_validation:
        t.terminate()

if __name__ == "__main__":
    '''
    Arguments parsing
    '''
    args = parse_args()

    # Fetch the json configuration file
    configuration = os.path.join(setup_configurations.config_dir, args.configuration + ".json")
    with open(configuration, "r") as f:
        configs_dict = json.load(f)

    configs_dict["system"]["snapshot_name"] = args.configuration
    setup_configurations.update_config(configs_dict["system"])

    # Fetch the inputs of traning and validation
    inputs_train = setup_configurations.train_split
    inputs_validation   = setup_configurations.val_split

    dataset = setup_configurations.dataset  # MSCOCO | FVV
    num_threads = args.num_threads  # Default is 6: every 6 epoch shuffle the indices

    # Load databases
    train_databases  = [datasets[dataset](configs_dict["db"], inputs_train) for _ in range(num_threads)]
    validation_database = datasets[dataset](configs_dict["db"], inputs_validation)

    # Start training
    train(train_databases, validation_database, args.begin_iteration, args.freeze) # 0
