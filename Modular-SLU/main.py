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
parser.add_argument('--cont', action = 'store_true', help = 'continue from previous point')
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

# Manually set seed (for reproducability)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

if train:
    # Get datasets as pandas dataframes
    train_data, valid_data, test_data = data.get_dataset(config)
    embedding_size = len(train_data[0][0][0])
    # Initialize the model
    model = models.Model(config, embedding_size)
    # Initialie the training class
    trainer = trainer.Trainer(model, config)
    # If continuing, load previous checkpoint
    if cont:
        trainer.load()
    # Train the model
    for epoch in range(config.num_epochs):
        print("----------------Epoch #%d of %d" % (epoch+1, config.num_epochs))
        # Train the model on the training dataset
        train_accuracy, train_loss = trainer.train(train_data)
        valid_accuracy, valid_loss = trainer.test(valid_data)
        # Print results of epoch of training
        print("-------Results: training accuracy: %.2f, training loss: %.2f, \
                valid accuracy: %.2f, valid loss %.2f" % (train_accuracy, train_loss, \
                valid_accuracy, valid_loss))
        # Save model at end of each epoch
        trainer.save()
    # Get final test set results
    test_accuracy, test_loss = trainer.test(test_data)
    print("----------------Final Results: test accuracy: %.2f, test loss: %.2f" % (test_accuracy, test_loss))

if infer:
    # TODO add inference ability on random.wav files
    a = 0
