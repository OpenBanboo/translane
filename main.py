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
import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import importlib
import datasets
import threading
import traceback
import util.misc as utils
from tqdm import tqdm
from util import stdout_to_tqdm
from configuration import setup_configurations
from factory.network_factory import NetworkFactory
from torch.multiprocessing import Process, Queue, Pool
from database.datasets import datasets
import models.py_utils.misc as utils
from util.general_utils import create_directories, pin_memory, init_parallel_jobs, prefetch_data
#from datasets import build_dataset, get_coco_api_from_dataset
#from engine import evaluate, train_one_epoch
#from models import build_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Network")
    # Model parameters
    parser.add_argument("configuration", help="Configuration File", type=str)
    # Number o
    parser.add_argument("-b", dest="begin_iteration",
                        help="Begin to train at iteration b using pretrained model",
                        default=0, type=int)
    parser.add_argument("-t", dest="num_threads", default=8, type=int)
    parser.add_argument("--freeze", action="store_true")

    args = parser.parse_args()
    return args

def train(training_dbs, validation_db, begin_iteration=0, freeze=False):
    lr    = setup_configurations.lr
    end_iter    = setup_configurations.end_iter
    pretrained_model = setup_configurations.pretrain
    snapshot         = setup_configurations.snapshot
    val_iter         = setup_configurations.val_iter
    display          = setup_configurations.display
    decay_rate       = setup_configurations.decay_rate
    stepsize         = setup_configurations.stepsize
    batch_size       = setup_configurations.batch_size

    # getting the size of each database
    training_size   = len(training_dbs[0].db_inds)
    validation_size = len(validation_db.db_inds)

    # queues storing data for training
    training_queue   = Queue(setup_configurations.prefetch_size) # 5
    validation_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(setup_configurations.prefetch_size) # 5
    pinned_validation_queue = queue.Queue(5)

    # load data sampling function
    data_file   = "sample.{}".format(training_dbs[0].data) # "sample.coco"
    sample_data = importlib.import_module(data_file).sample_data
    # print(type(sample_data)) # function

    # allocating resources for parallel reading
    training_tasks   = init_parallel_jobs(training_dbs, training_queue, sample_data)
    if val_iter:
        validation_tasks = init_parallel_jobs([validation_db], validation_queue, sample_data)

    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()

    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

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

    # Put it on GPU
    model.cuda()
    model.train_mode()
    header = None
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    with stdout_to_tqdm() as save_stdout:
        for iteration in metric_logger.log_every(tqdm(range(begin_iteration + 1, end_iter + 1),
                                                      file=save_stdout, ncols=67),
                                                 print_freq=10, header=header):

            training = pinned_training_queue.get(block=True)
            viz_split = 'train'
            save = True if (display and iteration % display == 0) else False
            (set_loss, loss_dict) \
                = model.train(iteration, save, viz_split, **training)
            (loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, loss_value) = loss_dict
            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=lr)

            del set_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                model.eval_mode()
                viz_split = 'val'
                save = True
                validation = pinned_validation_queue.get(block=True)
                (val_set_loss, val_loss_dict) \
                    = model.validate(iteration, save, viz_split, **validation)
                (loss_dict_reduced, loss_dict_reduced_unscaled, loss_dict_reduced_scaled, loss_value) = val_loss_dict
                print('[VAL LOG]\t[Saving training and evaluating images...]')
                metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
                metric_logger.update(class_error=loss_dict_reduced['class_error'])
                metric_logger.update(lr=lr)
                model.train_mode()

            if iteration % snapshot == 0:
                model.save_params(iteration)

            if iteration % stepsize == 0:
                lr /= decay_rate
                model.set_lr(lr)

            if iteration % (training_size // batch_size) == 0:
                metric_logger.synchronize_between_processes()
                print("Averaged stats:", metric_logger)


    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()
    for validation_task in validation_tasks:
        validation_task.terminate()

if __name__ == "__main__":
    args = parse_args()

    # Fetch the json configuration file
    configuration = os.path.join(setup_configurations.config_dir, args.configuration + ".json")

    with open(configuration, "r") as f:
        configs_dict = json.load(f)

    configs_dict["system"]["snapshot_name"] = args.configuration
    setup_configurations.update_config(configs_dict["system"])

    train_split = setup_configurations.train_split
    val_split   = setup_configurations.val_split

    dataset = setup_configurations.dataset  # MSCOCO | FVV

    num_threads = args.num_threads  # 4 every 4 epoch shuffle the indices
    training_dbs  = [datasets[dataset](configs_dict["db"], train_split) for _ in range(num_threads)]
    validation_db = datasets[dataset](configs_dict["db"], val_split)

    train(training_dbs, validation_db, args.begin_iteration, args.freeze) # 0
