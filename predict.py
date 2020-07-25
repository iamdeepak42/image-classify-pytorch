import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from PIL import Image
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import shutil, argparse, json

from train_utils import load_checkpoint,process_image,predict_from_checkpoint


def main():
<<<<<<< HEAD
	'''
	First of all we define get input from the user on all the required parameters for predictions 
	
	'''
    inputs = argparse.Argumentinput(description='Predict flower name from an image along with the probability of that name')
    inputs.add_argument('path', type=str)
    inputs.add_argument('checkpoint', type=str)
    inputs.add_argument('topk', type=int)
    inputs.add_argument('category_names', type=str)
    inputs.add_argument('--gpu', action='store_true')
=======
   
    parser = argparse.ArgumentParser(description='Predict flower name')
    
    parser.add_argument('input', type=str, help='Image path')
    parser.add_argument('checkpoint', type=str, help='Models checkpoint')
    
    parser.add_argument('--top_k', type=int, help='top k most likely classes')
    parser.add_argument('--category_names', type=str, help='Use a mapping of categories to real names from a json file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU ')

    args, _ = parser.parse_known_args()
>>>>>>> af40fd637ed72070c4bccdd57526dee80e78836e

    
    arguments = inputs.parse_args() # storing all the arguments
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'
	
	# Categorical mapping is stored in cat_to_name
	with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = ImageClassifier(device)
    model.load(args.checkoint)
	
    image_path = args.input
    checkpoint = args.checkpoint
	
	
	predict_name_prob = model.predict_from_checkpoint(process_image(args.path), args.top_k, cat_to_name)

<<<<<<< HEAD
    for i,j in predict_name_prob:
		print(i,j)
=======
    top_k = 1
    if args.top_k:
        top_k = args.top_k

    category_names = None
    if args.category_names:
        category_names = args.category_names

    cuda = False
    if args.gpu:
        if torch.cuda.is_available():
            cuda = True
        else:
            print("GPU not available")

    probs, classes, class_names = predict_from_checkpoint(image_path, checkpoint, topk=top_k, category_names=category_names, cuda=cuda)
    print("="*70)
    print(" "*25 + 'FLOWER PREDICTOR')
    print("="*75)
    print("Input label (or labels) = {}".format(classes))
    print("Probability confidence(s) = {}".format(probs))
    print("Class(es) name(s) = {}".format(class_names))
    print("="*75)
>>>>>>> af40fd637ed72070c4bccdd57526dee80e78836e
    



if __name__ == '__main__':
    main()	
