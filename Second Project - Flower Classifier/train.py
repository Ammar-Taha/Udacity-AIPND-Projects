from network_routines import DataLoaders, CNNetwork, NetworkTrainer, NetworkTester, SaveCheckpoint
from argparse import ArgumentParser
import torch

parser = ArgumentParser()

## Adding Arguments to the Parser - in order of the project specs
parser.add_argument('data_dir'  , action="store")
parser.add_argument('--save_dir', dest="save_dir", action="store")
parser.add_argument('--arch'    , type=str, dest="arch"    , action="store", choices=['vgg16', 'alexnet', 'densenet121'])
parser.add_argument('--learning_rate', dest="learning_rate", action="store")
parser.add_argument('--hidden_units' , type=int, dest="hidden_units", action="store")
parser.add_argument('--epochs', dest="epochs", type=int, action="store")
parser.add_argument('--gpu'   , dest="gpu", action="store")

"""
Note for the --arch Argument:
choices=['vgg16', 'alexnet', 'densenet121']: Here, 'choices' is a keyword argument for the 'add_argument' 
    method of the 'ArgumentParser' object named 'parser'. 
It is set to a list of strings: 'vgg16,' 'alexnet,' and 'densenet121'. 
This list specifies the valid values that the '--arch' argument can take.
By defining the 'choices' keyword argument in this way, you ensure that when users run your script, 
    they can only provide one of these three CNN architecture names as an argument value for '--arch.' 
If they try to provide any other value, the argparse module will raise an error, ensuring that the input 
    is restricted to the specified choices. This helps in ensuring the correctness and robustness 
    of your script's command-line interface.
"""

## Parsing the Added Arguments
parsed_args = parser.parse_args()

## Propagating Parsed Arguments
data_directory = parsed_args.data_dir
check_path = parsed_args.save_dir
pretrained = parsed_args.arch
learn_rate = parsed_args.learning_rate
hidden1 = parsed_args.hidden_units
epochs = parsed_args.epochs
worker = parsed_args.gpu

if worker == 'gpu' and torch.cuda.is_available():
    device = 'gpu'
else:
    device = 'cpu'

## The main() method
def main():
    trainset, trainloader, validloader, testloader = DataLoaders(data_directory)
    dropout = 0.2
    model, criterion, optimizer = CNNetwork(pretrained, learn_rate, hidden1, dropout, device)
    NetworkTrainer(model, criterion, optimizer, trainloader, validloader, epochs, device)
    NetworkTester(model, criterion, testloader, device)
    n_classes=102
    SaveCheckpoint(model, trainset, check_path, pretrained, epochs, n_classes, hidden1, dropout, learn_rate)

if __name__== "__main__":
    main()
