#!/usr/bin/env python3
# Load dataset into trainer, train model and test accordingly

import torch
import data
import models
import trainer
import argparse
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--train', action = 'store_true', help = 'run training with given config file')
parser.add_argument('--cont', action = 'store_true', help = 'continue from previous checkpoint')
parser.add_argument('--infer', action = 'store_true', help = 'run inference with trained model')
parser.add_argument('--config_path', type = str, help = 'path to configuration file')
args = parser.parse_args()
train = args.train
cont = args.cont
infer = args.infer
config_path = args.config_path

if config_path is None:
    sys.exit("No configuration file given, ending program")

if train and infer:
    sys.exit("Both training and inferrence flags set. Only one can be run at a time")

# Parse given configuration files
config = data.parse_config(config_path)

# Get datasets as pandas dataframes
train_df, valid_df, test_df = data.get_dataset(config)

if train:
    # TODO train model

if infer:
    # TODO add inference ability on random.wav files
