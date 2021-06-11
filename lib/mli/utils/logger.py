import errno
import json
import os
import sys
import torch

from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """
    Base Logger object

    Initializes the log directory and creates log files given by name in arguments.
    Can be used to append future log values to each file.
    """
    def __init__(self, log_dir, *args):
        self.log_dir = log_dir

        try:
            os.makedirs(log_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        with open(os.path.join(self.log_dir, "cmd.txt"), "w") as f:
            f.write(" ".join(sys.argv))

        self.log_names = [a for a in args]
        for arg in self.log_names:
            setattr(self, "log_{}".format(arg), lambda epoch, value, name=arg: self.log(name, epoch, value))
            self.init_logfile(arg)

    def log_config(self, config):
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f)

    def init_logfile(self, name, xlabel="epoch"):
        fname = self.get_log_fname(name)

        with open(fname, "w") as log_file:
            log_file.write("{},{}\n".format(xlabel, name))

    def get_log_fname(self, name):
        return os.path.join(self.log_dir, "{}.log".format(name))

    def log(self, name, epoch, value):
        if name not in self.log_names:
            try:  # initialize only if not done so already
                self.get_log_fname(name)
            except FileNotFoundError:
                self.init_logfile(name)
            self.log_names.append(name)
        fname = self.get_log_fname(name)

        with open(fname, "a") as log_file:
            log_file.write("{},{}\n".format(epoch, value))

    def log_test_value(self, name, value):
        test_name = "test_" + name
        self.init_logfile(test_name)
        self.log(test_name, 0, value)


class ExperimentLogger(object):
    """Wraps Tensorboard logger and lightweight logger.
    TensorBoard logger can be useful for looking at all experiments

    Performs checkpointing and resumes experiments that are not completed.
    """
    def __init__(self, log_dir, checkpoint_name="latest.tar"):
        self.log_dir = log_dir
        self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir)
        self.logger = Logger(log_dir=log_dir)
        self.checkpoint_name = os.path.join(self.log_dir, checkpoint_name)

    def update_tensorboard_writer(self, epoch):
        self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir, purge_step=epoch)

    def checkpoint_exists(self):
        return os.path.isfile(self.checkpoint_name)

    def load_checkpoint(self):
        return torch.load(self.checkpoint_name)

    def log_hyperparams(self, hyp_dict, file_name="hyperparams.json"):
        with open(os.path.join(self.log_dir, file_name), "w") as fp:
            json.dump(hyp_dict, fp)

    def add_scalar(self, name, iter, val):
        self.logger.log(name, iter, val)
        self.tensorboard_writer.add_scalar(name, val, iter)

    def log_epoch(self, epoch, epoch_stats, state=None):
        # saves different metrics and optionally save checkpoint itself (via state)
        if state is not None:
            assert "epoch" in state
            assert "model" in state
            assert "optimizer" in state

            torch.save(state, self.checkpoint_name)
        for k, v in epoch_stats.items():
            self.add_scalar(k, epoch, v)


def join_exp_dict(exp_dict):
    return ",".join(["{}={}".format(k, v) for k, v in exp_dict.items()])


def create_exp_name(args, hyperparameters):
    from collections import OrderedDict

    ordered_params = OrderedDict()
    for h in hyperparameters:
        # getting as dictionary should work
        try:
            v = getattr(args, h)
            ordered_params[h] = v
        except AttributeError:
            raise Exception("hyperparameter not found")
    exp_name = join_exp_dict(ordered_params)
    return ordered_params, exp_name
