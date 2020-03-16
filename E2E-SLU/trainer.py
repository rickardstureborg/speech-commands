# Handles trainer class and training related functions
import torch
import data
import models
from tqdm import tqdm

""" Handles training related functions and holds training information """
class Trainer:
    def __init__(self, model, config):
        """ Initialize trainer class """
        self.model = model
        self.config = config
        self.lr = config.learning_rate
        self.save_path = config.save_path
        self.cur_epoch = 0
        # Use Adam optimizer, because it's just the best
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr)

    def load(self):
        """ load previous state_dict to resume training """
        try:
            if self.model.is_cuda:
                self.model.load_state_dict(torch.load(os.path.join(self.save_path, "save_point.pth")))
            else:
                self.model.load_state_dict(torch.load(os.path.join(self.save_path, \
                        "save_point.pth"), map_location="cpu"))
        except:
            sys.exit("Unable to load previous model")

    def save(self):
        """ save state_dict of model """
        try:
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "save_point.pth"))
        except:
            print("Unable to save the model")

    def train(self, dataset):
        """ train the given model on the given dataset """
        print_interval = 100
        train_accuracy = 0
        train_loss = 0
        num_examples_trained = 0
        # Put model into training mode
        self.model.train()
        # Step through dataset using pytorch data loader, and a tqdm progress bar
        for num, batch in enumerate(tqdm(dataset.loader)):
            xs, ys = batch
            batch_size = len(xs)
            num_examples_trained += batch_size
            iloss, iaccuracy = self.model(xs, ys)
            loss = iloss
            # Zero out the gradient
            self.optimizer.zero_grad()
            # Perform backwards pass (backprop)
            loss.backward()
            # Step the optimizer
            self.optimizer.step()
            train_loss += iloss.cpu().data.numpy().item() * batch_size
            train_accuracy += iaccuracy.cpu().data.numpy().item() * batch_size
            # Print results info
            if num % print_interval == 0:
                print("training loss: " + str(iloss.cpu().data.numpy().item()))
                print("training acc: " + str(iaccuracy.cpu().data.numpy().item()))
        train_accuracy = train_accuracy / num_examples_trained
        train_loss = train_loss / num_examples_trained
        # Increment current epoch number
        self.cur_epoch += 1
        # Return accuracy and loss
        return train_accuracy, train_loss

    def test(self, dataset):
        """ test the given model on the given dataset """
        test_accuracy = 0
        test_loss = 0
        num_examples_tested = 0
        # Put model into evaluation mode
        self.model.eval()
        for num, batch in enumerate(dataset.loader):
            xs, ys = batch
            batch_size = len(xs)
            num_examples += batch_size
            iloss, iaccuracy = self.model(xs, ys)
            test_loss += iloss.cpu().data.numpy().item() * batch_size
            test_accuracy += iaccuracy.cpu().data.numpy().item() * batch_size
        test_accuracy = test_accuracy / num_examples
        test_loss = test_loss / num_examples
        # Return accuracy and loss for this model on the test set
        return test_accuracy, test_loss



