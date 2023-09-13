from network_routines import LoadCheckpoint, Predictor
from argparse import ArgumentParser
import json
import torch

parser = ArgumentParser()

## Adding Arguments to the Parser - in order of the project specs
parser.add_argument('input'     , type = str, action="store")
parser.add_argument('checkpoint', type = str, action="store")
parser.add_argument('--top_k'   , type = int, action="store", default=3, dest="top_k")
parser.add_argument('--category_names', action="store", default='cat_to_name.json', dest="category_names")
parser.add_argument('--gpu', action="store", dest="gpu")

## Parsing the Added Arguments
parsed_args = parser.parse_args()

## Propagating Parsed Arguments
image_path = parsed_args.input
check_path = parsed_args.checkpoint
topk_probs = parsed_args.top_k
class_dict = parsed_args.category_names
worker = parsed_args.gpu

if worker == 'gpu' and torch.cuda.is_available():
    device = 'gpu'
else:
    device = 'cpu'

## The main() method
def main():
    model = LoadCheckpoint(check_path)
    with open(class_dict, 'r') as f:
        cat_to_name = json.load(f, strict=False)
    class_probs, class_names, pred_class_name = Predictor(image_path, model, cat_to_name, topk_probs, device)
    print(f"The Network Predicts that the Input Image is : {pred_class_name}\n"
          f"The Class Probabilities are : {class_probs}\n"
          f"The topk {topk_probs} Probs are for Class Names of : {class_names}")

if __name__== "__main__":
    main()