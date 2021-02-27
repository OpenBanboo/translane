# Use most of the settings the same as LSTR: https://github.com/liuruijin17/LSTR

import numpy as np
import os

class Configuration:
    def __init__(self):
        # Settings dictionary
        self._settings = {}
        self._settings["dataset"] = None
        self._settings["sampling_function"] = "kp_detection"

        # Random generator settings
        self._settings["data_rng"] = np.random.RandomState(123)
        self._settings["nnet_rng"] = np.random.RandomState(317)

        # Dir settings
        self._settings["data_dir"]   = "./data"
        self._settings["saved_dir"] = "./saved"
        self._settings["config_dir"] = "./configurations"
        self._settings["result_dir"] = "./results"

        # Training Config
        self._settings["display"]           = 5
        self._settings["snapshot"]          = 5000
        self._settings["stepsize"]          = 450000
        self._settings["lr"]                = 0.00025
        self._settings["decay_rate"]        = 10
        self._settings["end_iter"]          = 500000
        self._settings["val_iter"]          = 100
        self._settings["batch_size"]        = 1
        self._settings["snapshot_name"]     = None
        self._settings["prefetch_size"]     = 100
        self._settings["weight_decay"]      = False
        self._settings["weight_decay_rate"] = 1e-5
        self._settings["weight_decay_type"] = "l2"
        self._settings["pretrain"]          = None
        self._settings["opt_algo"]          = "adam"
        self._settings["chunk_sizes"]       = None
        self._settings["use_crop"]          = False

        # Training/Validation/Test Dataset Split settings
        self._settings["train_split"] = "trainval"
        self._settings["val_split"]   = "minival"
        self._settings["test_split"]  = "testdev"

        # Settings for MSDETR Model
        self._settings["res_layers"] = [2, 2, 2, 2]
        self._settings["res_dims"] = [64, 128, 256, 512]
        self._settings["res_strides"] = [1, 2, 2, 2]
        self._settings["attn_dim"] = 64
        self._settings["dim_feedforward"] = 4 * 64

        self._settings["drop_out"] = 0.1
        self._settings["num_heads"] = 8
        self._settings["enc_layers"] = 6
        self._settings["dec_layers"] = 6
        self._settings["lsp_dim"] = 8
        self._settings["mlp_layers"] = 3

        self._settings["aux_loss"] = True
        self._settings["pos_type"] = 'sine'
        self._settings["pre_norm"] = False
        self._settings["return_intermediate"] = True

        self._settings["block"] = "BasicBlock"
        self._settings["lane_categories"] = 2

        self._settings["num_queries"] = 100

        # LaneDetection Setting
        self._settings["max_lanes"] = 6

    # class properties
    @property
    def max_lanes(self):
        return self._settings["max_lanes"]

    @property
    def num_queries(self):
        return self._settings["num_queries"]

    @property
    def lane_categories(self):
        return self._settings["lane_categories"]

    @property
    def block(self):
        return self._settings["block"]

    @property
    def return_intermediate(self):
        return self._settings["return_intermediate"]

    @property
    def pre_norm(self):
        return self._settings["pre_norm"]

    @property
    def pos_type(self):
        return self._settings["pos_type"]

    @property
    def aux_loss(self):
        return self._settings["aux_loss"]

    @property
    def mlp_layers(self):
        return self._settings["mlp_layers"]

    @property
    def lsp_dim(self):
        return self._settings["lsp_dim"]

    @property
    def dec_layers(self):
        return self._settings["dec_layers"]

    @property
    def enc_layers(self):
        return self._settings["enc_layers"]

    @property
    def num_heads(self):
        return self._settings["num_heads"]

    @property
    def drop_out(self):
        return self._settings["drop_out"]

    @property
    def dim_feedforward(self):
        return self._settings["dim_feedforward"]

    @property
    def attn_dim(self):
        return self._settings["attn_dim"]

    @property
    def res_strides(self):
        return self._settings["res_strides"]

    @property
    def res_dims(self):
        return self._settings["res_dims"]

    @property
    def res_layers(self):
        return self._settings["res_layers"]

    @property
    def chunk_sizes(self):
        return self._settings["chunk_sizes"]

    @property
    def use_crop(self):
        return self._settings["use_crop"]

    @property
    def train_split(self):
        return self._settings["train_split"]

    @property
    def val_split(self):
        return self._settings["val_split"]

    @property
    def test_split(self):
        return self._settings["test_split"]

    @property
    def full(self):
        return self._settings

    @property
    def sampling_function(self):
        return self._settings["sampling_function"]

    @property
    def data_rng(self):
        return self._settings["data_rng"]

    @property
    def nnet_rng(self):
        return self._settings["nnet_rng"]

    @property
    def opt_algo(self):
        return self._settings["opt_algo"]

    @property
    def weight_decay_type(self):
        return self._settings["weight_decay_type"]

    @property
    def prefetch_size(self):
        return self._settings["prefetch_size"]

    @property
    def pretrain(self):
        return self._settings["pretrain"]

    @property
    def weight_decay_rate(self):
        return self._settings["weight_decay_rate"]

    @property
    def weight_decay(self):
        return self._settings["weight_decay"]

    @property
    def result_dir(self):
        result_dir = os.path.join(self._settings["result_dir"], self.snapshot_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        return result_dir

    @property
    def dataset(self):
        return self._settings["dataset"]

    @property
    def snapshot_name(self):
        return self._settings["snapshot_name"]

    @property
    def snapshot_dir(self):
        snapshot_dir = os.path.join(self.saved_dir, "model", self.snapshot_name)

        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)

        return snapshot_dir

    @property
    def snapshot_file(self):
        snapshot_file = os.path.join(self.snapshot_dir, self.snapshot_name + "_{}.pkl")
        return snapshot_file

    @property
    def box_snapshot_dir(self):
        box_snaptshot_dir = os.path.join(self.box_saved_dir, 'model', self.snapshot_name)
        return box_snaptshot_dir

    @property
    def box_snapshot_file(self):
        box_snapshot_file = os.path.join(self.box_snapshot_dir, self.snapshot_name + "_{}.pkl")
        return box_snapshot_file

    @property
    def config_dir(self):
        return self._settings["config_dir"]

    @property
    def batch_size(self):
        return self._settings["batch_size"]

    @property
    def end_iter(self):
        return self._settings["end_iter"]

    @property
    def lr(self):
        return self._settings["lr"]

    @property
    def decay_rate(self):
        return self._settings["decay_rate"]

    @property
    def stepsize(self):
        return self._settings["stepsize"]

    @property
    def snapshot(self):
        return self._settings["snapshot"]

    @property
    def display(self):
        return self._settings["display"]

    @property
    def val_iter(self):
        return self._settings["val_iter"]

    @property
    def data_dir(self):
        return self._settings["data_dir"]

    @property
    def saved_dir(self):
        if not os.path.exists(self._settings["saved_dir"]):
            os.makedirs(self._settings["saved_dir"])
        return self._settings["saved_dir"]

    @property
    def box_saved_dir(self):
        return self._settings['box_saved_dir']

    def update_config(self, new):
        for key in new:
            if key in self._settings:
                self._settings[key] = new[key]

setup_configurations = Configuration()
