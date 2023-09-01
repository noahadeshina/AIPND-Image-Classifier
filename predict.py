import numpy as np
import torch
from torch import nn
from torchvision import models
from PIL import Image
import json
from collections import OrderedDict
import argparse


def load_checkpoint(file_path):
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(file_path, map_location=device)
    arch = checkpoint['arch']
    # arch = 'densenet121'

    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        print('Invalid Architecture!')

    for param in model.parameters():
            param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state'])
  
    return model

def process_image(image):
    img = Image.open(image)
    width, height = img.size
    
    if width > height:
        img.thumbnail((width, 256))
    else:
        img.thumbnail((256, height))
    
    left = (img.width - 224) / 2
    bottom = (img.height - 224) / 2
    right = left + 224
    top = bottom + 224
    
    img = img.crop((left, bottom, right, top))
    
    np_image = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose([2, 0, 1])
    
    return np_image

def predict(image_path, topk):
    model = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model.to(device)
    model.eval()
    
    img = process_image(image_path)
    img = np.expand_dims(img, axis=0)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    
    with torch.no_grad():
        output = model.forward(img_tensor)

    probs = torch.exp(output)
    probs, idx = probs.topk(topk)
    probs = probs.numpy().squeeze()
    idx = idx.numpy().squeeze()
    
    class_idx = model.class_to_idx
    idx_class = {value: key for key, value in class_idx.items()}
    classes = [idx_class[i] for i in idx]
    
    return probs, classes

def main():
    probs, classes = predict(args.input, args.top_k)
    print("\nPredicted Classes and Probabilty")
    for prob, class_ in zip(probs, classes):
        print(f'Class: {class_}     Probability: {prob:.2f}')
    
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        
        class_names = {class_:cat_to_name[class_] for class_ in classes}
        print(f'\n{class_names}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='path to the image')
    parser.add_argument('checkpoint', type=str, help='model checkpoint')
    parser.add_argument('--top_k', type=int, help='number of top k class', default=3)
    parser.add_argument('--category_names', type=str, help='category of real names', 
                        default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')
    args = parser.parse_args()
    main()