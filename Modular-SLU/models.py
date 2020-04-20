# Holds all models and model functions
import torch

""" Selects hidden states of RNN output """
class RNNOutput(torch.nn.Module):
    def __init__(self):
        """ Initialize class """
        super(RNNOutput, self).__init__()

    def forward(self, inp):
        """ Grabs first of input """
        return inp[0]

""" Final pooling layer for this model """
class FinalPooling(torch.nn.Module):
    def __init__(self):
        """ Initialize Final Pooling class """
        super(FinalPooling, self).__init__()

    def forward(self, inp):
        """ Returns a reshaped matrix tensor """
        return inp.max(dim=1)[0]

""" RNN Model """
class Model(torch.nn.Module):
    def __init__(self, config, embedding_size):
        """ Initialize model """
        super(Model, self).__init__()
        self.is_cuda = torch.cuda.is_available()
        self.possible_intents = config.possible_intents
        self.values_per_col = config.values_per_col
        self.num_total_values = sum(self.values_per_col)
        self.rnn_layers = []
        num_rnn_layers = len(config.rnn_hidden_num)
        # Initialize all RNN layers
        for num in range(num_rnn_layers):
            # Create and add new recurrent layer
            # Note: Using GRU's (Gated Recurrence Units)
            new_layer = torch.nn.GRU(input_size = embedding_size, hidden_size = config.rnn_hidden_num[num], \
                    batch_first = True, bidirectional = config.rnn_bidirectional)
            new_layer.name = "Gated Recurrence Layer %d" % num
            self.rnn_layers.append(new_layer)
            rnn_out_dim = config.rnn_hidden_num[num]
            if config.rnn_bidirectional:
                rnn_out_dim = rnn_out_dim * 2
            # Layer for grabbing output of GRU RNN (GRU returns 2 values, so return the first one)
            new_layer = RNNOutput()
            new_layer.name = "RNN Output Selection Layer %d" % num
            self.rnn_layers.append(new_layer)
            # Create and add layer for dropout
            new_layer = torch.nn.Dropout(p = config.rnn_dropout[num])
            new_layer.name = "RNN Dropout Layer %d" % num
            self.rnn_layers.append(new_layer)
        # Create and add final, linear feedforward layer
        new_layer = torch.nn.Linear(rnn_out_dim, self.num_total_values)
        new_layer.name = "Final Classifier Layer"
        self.rnn_layers.append(new_layer)
        # Create and add final pooling layer
        new_layer = FinalPooling()
        new_layer.name = "Final Pooling Layer"
        self.rnn_layers.append(new_layer)
        # Convert RNN Layers list to list of torch modules
        self.rnn_layers = torch.nn.ModuleList(self.rnn_layers)
        # Set model to be cuda based, if available
        if self.is_cuda:
            self.cuda()

    def forward(self, x, y):
        """ Performs a forward pass on this model """
        # Convert x and y to be cuda based, if available
        if self.is_cuda:
            x = x.cuda()
            y = y.cuda()
        out = x
        # Perform forward pass for RNN layers
        for layer in self.rnn_layers:
            out = layer(out)
        logits = out
        loss = 0
        start_num = 0
        predicted_output = []
        # Convert from vector to predicted output for each column
        for col in range(len(self.values_per_col)):
            end_num = start_num + self.values_per_col[col]
            subset = logits[:, start_num:end_num]
            loss += torch.nn.functional.cross_entropy(subset, y[:, col])
            predicted_output.append(subset.max(1)[1])
            start_num = end_num
        predicted_output = torch.stack(predicted_output, dim = 1)
        # Compute accuracy (all slots must be correct in order to count)
        accuracy = (predicted_output == y).prod(1).float().mean()

        return loss, accuracy


