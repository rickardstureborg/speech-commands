# Handle data ingest and configuration file ingest

import os
import pandas as pd
from collections import Counter
import torch
import torch.utils.data
import numpy as np
import configparser
import sys
import speech_recognition as sr
import gensim.models

def parse_config(config_file):
    """ Set all config options form given config file, return new Config file class """
    config = Config()
    parser = configparser.ConfigParser()
    parser.read(config_file)
    # Set every config option from the configuration file
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
        """ Initialize configuration file class """

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
    # Preload word2vec model
    print("Loading word2vec model...")
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(\
            './GoogleNews-vectors-negative300.bin.gz', binary = True)
    # Create and return dataset classes
    train_data = SLUData(training_dataframe, possible_intents, path, config, word2vec)
    valid_data = SLUData(valid_dataframe, possible_intents, path, config, word2vec)
    test_data = SLUData(test_dataframe, possible_intents, path, config, word2vec)

    return train_data, valid_data, test_data

""" Class that holds data and functions relating to the SLU Dataset """
class SLUData(torch.utils.data.Dataset):
    def __init__(self, dataframe, possible_intents, path, config, word2vec):
        """ Initialize the dataset """
        self.df = dataframe
        self.possible_intents = possible_intents
        self.path = path
        self.config = config
        # Use pytorch's dataloader for ease of use down the road
        self.loader = torch.utils.data.DataLoader(self, batch_size=config.training_batch_size, \
                shuffle = True, collate_fn = CollateSLU(self.possible_intents))
        # Use speech_recognitions Recognizer
        self.recognizer = sr.Recognizer()
        # Use the Google News word2vec model
        self.word2vec = word2vec

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
        # Run speech_recognition library on .wav file
        with sr.AudioFile(wav_path) as source:
            audio = self.recognizer.record(source)
        texti = self.recognizer.recognize_google(audio, language = \
                'en-US', show_all = True)
        if not isinstance(texti, dict):
            text = 'words '
        else:
            text = texti['alternative'][0]['transcript']
        # Get word embeddings for this text
        embeddings = []
        for word in text.split(" "):
            if str(word) in self.word2vec.vocab:
                embeddings.append(self.word2vec[str(word)])
        x = embeddings
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
        embedding_len = 300
        zero_tensor = torch.zeros([1, embedding_len], dtype=torch.double).float()
        # Convert all values in batch to tensors
        for i in range(batch_size):
            x, y = batch[i]
            word_embeddings = []
            for j in range(0, len(x)):
                word_embeddings.append(torch.tensor(x[j]).float())
            if len(word_embeddings) == 0:
                word_embeddings.append(torch.squeeze(zero_tensor))
            xs.append(torch.stack(word_embeddings))
            ys.append(torch.tensor(y).long())
        # Pad out all lists of word embeddings to be the same length
        max_size = 0
        for x in xs:
            if len(x) > max_size:
                max_size = len(x)
        embedding_len = len(xs[0][0])
        new_xs = []
        for x in xs:
            if len(x) < max_size:
                pad_size = max_size - len(x)
                for i in range(pad_size):
                    x = torch.cat((x, zero_tensor), dim=0)
            new_xs.append(x)
        # Stack data to convert from list of tensors to one tensor w/ extra dimension
        new_xs = torch.stack(new_xs)
        ys = torch.stack(ys)
        return (new_xs, ys)
