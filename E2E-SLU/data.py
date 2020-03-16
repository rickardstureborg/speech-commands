# Handle data ingest and configuration file ingest

# NOTE: Inspiration for the ingest of this dataset comes from original
#       creators of this dataset. The paper that introduces this dataset
#       can be found at https://arxiv.org/abs/1904.03670

import os
import pandas as pd
from collections import Counter
import torch
import torch.utils.data
import torchaudio
import numpy
import configparser
import sys

def parse_config(config_file):
    """ Set all config options form given config file, return new Config file class """
    config = Config()
    parser = configparser.ConfigParser()
    parser.read(config_file)
    # Set every config option from the configuration file
    # CNN configurations
    config.cnn_filter_input_sizes = [int(x) for x in parser.get("cnn", "cnn_filter_input_sizes").split(",")]
    config.cnn_filter_output_sizes = [int(x) for x in parser.get("cnn", "cnn_filter_output_sizes").split(",")]
    config.cnn_kernel_sizes = [int(x) for x in parser.get("cnn", "cnn_kernel_sizes").split(",")]
    config.cnn_strides = [int(x) for x in parser.get("cnn", "cnn_strides").split(",")]
    config.cnn_pool_kernel_sizes = [int(x) for x in parser.get("cnn", "cnn_pool_kernel_sizes").split(",")]

    # RNN Configurations
    config.rnn_hidden_num = [int(x) for x in parser.get("rnn", "rnn_hidden_num").split(",")]
    config.rnn_bidirectional = (parser.get("rnn", "rnn_bidirectional") == "True")
    config.rnn_dropout = [float(x) for x in parser.get("rnn", "rnn_dropout").split(",")]

    # Training configurations
    config.data_path = parser.get("training", "data_path")
    config.seed = int(parser.get("training", "seed"))
    config.training_batch_size = int(parser.get("training", "training_batch_size"))
    config.num_epochs = int(parser.get("training", "num_epochs"))
    config.learning_rate = float(parser.get("training", "learning_rate"))
    config.save_path = parser.get("training", "save_path")

    return config

""" Class for holding configuration file information """
class Config:
    def __init__(self):
        """ Initialize cofiguration file class """

def get_dataset(config):
    """ Returns a class containing the SLU dataset and relevant functions """
    # Set path to dataset from configuartion file
    path = config.data_path
    # Create training, validation and test dataframes
    training_path = os.path.join(path, "data", "train_data.csv")
    training_dataframe = pd.read_csv(training_path)
    valid_path = os.path.join(path, "data", "valid_data.csv")
    valid_dataframe = pd.read_csv(valid_path)
    test_path = os.path.join(path, "data", "test_data.csv")
    test_dataframe = pd.read_csv(test_path)
    # Get all possible output values and assign values to them
    possible_intents = {"action": {}, "object": {}, "location": {}}
    values_per_col = []
    output_columns = ["action", "object", "location"]
    for col in output_columns:
        possible_values = Counter(training_dataframe[col])
        for idx, value in enumerate(possible_values):
            possible_intents[col][value] = idx
        values_per_col.append(len(possible_values))
    config.possible_intents = possible_intents
    config.values_per_col = values_per_col
    # Create and return dataset classes
    train_data = SLUData(training_dataframe, possible_intents, path, config)
    valid_data = SLUData(valid_dataframe, possible_intents, path, config)
    test_data = SLUData(test_dataframe, possible_intents, path, config)

    return train_data, valid_data, test_data

""" Class that holds data and functions relating to the SLU Dataset """
class SLUData(torch.utils.data.Dataset):
    def __init__(self, dataframe, possible_intents, path, config):
        """ Initialize the dataset """
        self.df = dataframe
        self.possible_intents = possible_intents
        self.path = path
        self.config = config
        # Use pytorch's dataloader for ease of use down the road
        self.loader = torch.utils.data.DataLoader(self, batch_size=config.training_batch_size, \
                shuffle = True, collate_fn = CollateSLU(self.possible_intents))

    def __len__(self):
        """ Length of dataset """
        return len(self.df)

    def __getitem__(self, num):
        """ Return an input value and output value for a given dataset location """
        if num >= len(self.df):
            sys.exit("Cannot retrieve data, out of bounds")
        num = num % len(self.df)
        # Path to wav file for this dataset row
        wav_path = os.path.join(self.path, self.df.loc[num].path)
        # Create data representation of wav file
        # NOTE: This representation is the one used by the original creators of this dataset.
        #       You can find the relevant paper linked at the top of this file. All credit for
        #       this method of ingest of .wav audio files goes to them
        effect = torchaudio.sox_effects.SoxEffectsChain()
        effect.set_input_file(wav_path)
        wav, fs = effect.sox_build_flow_effects()
        x = wav[0].numpy()
        del wav, effect
        # Get all three intent values for this dataset row
        y = []
        intent_columns = ["action", "object", "location"]
        for col in intent_columns:
            val = self.df.loc[num][col]
            y.append(self.possible_intents[col][val])

        return (x, y)

""" Collate function for the SLU dataset, use in the DataLoader in the SLUData class """
class CollateSLU:
    def __init__(self, possible_intents):
        """ Initialize the collation class for this dataset """
        self.possible_intents = possible_intents
        self.num_labels = len(self.possible_intents)

    def __call__(self, batch):
        """ Return a minibatch of wavs and labels as Tensors """
        xs = []
        ys = []
        batch_size = len(batch)
        # Convert all values in batch to tensors
        for i in range(batch_size):
            x, y = batch[i]
            xs.append(torch.tensor(x).float())
            ys.append(torch.tensor(y).long())
        # All sequences are now padded to have the same length
        max_size = 0
        for x in xs:
            size = len(x)
            if size > max_size:
                max_size = size
        for i in range(batch_size):
            pad_len = max_size - len(xs[i])
            xs[i] = torch.nn.functional.pad(xs[i], (0, pad_len))
        # Stack data to convert from list of tensors to one tensor w/ extra dimension
        xs = torch.stack(xs)
        ys = torch.stack(ys)
        return (xs, ys)
