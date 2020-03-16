# Holds all models and model functions
import torch

""" Ensemble model of CNN and deep RNN layers """
class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.possible_intents = config.possibe_intents
        self.cnn_layers = []
        self.rnn_layers = []
        num_cnn_layers = config.num_cnn_layers
        num_rnn_layers = config.num_rnn_layers
        # Initialize all CNN layers
        for num in range(num_cnn_layers):
            # Create and add new convolutional layer
            # Note: All convolutional layers 1D, because wav files are 1D
            new_layer = torch.nn.Conv1d(config.cnn_filter_input_sizes[num], \
                    config.cnn_filter_output_sizes[num], kernel_size = config.cnn_kernel_sizes[num], \
                    stride = config.cnn_strides[num], padding = config.cnn_kernel_sizes[num] // 2)
            new_layer.name = "Convolutional Layer %d" % num
            self.cnn_layers.append(new_layer)
            # Create and add max pooling layer
            new_layer = torch.nn.MaxPool1d(config.cnn_pool_kernel_sizes[num], ceil_mode = True)
            new_layer.name = "Max Pooling Layer %d" % num
            self.cnn_layers.append(new_layer)
            # Create and add new Rectified Linear Unit Layer (LeakyReLU for training time speedup)
            new_layer = torch.nn.LeakyReLU(0.2)
            new_layer.name = "Leaky Rectified Linear Unit Layer %d" % num
            self.cnn_layers.append(new_layer)


