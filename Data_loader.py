
import os
import json
import numpy as np
import torch,torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary
from PIL import Image
import matplotlib.pyplot as plt
from chitra.image import Chitra
from pybboxes import BoundingBox

#annotations file path
anns_file_path = 'data/annotations.json'

# Read annotations
with open(anns_file_path, 'r') as f:
    dataset = json.loads(f.read())

#seperate the annotations
cat = dataset['categories']
anns = dataset['annotations']
imgs = dataset['images']

#intializing paramters
num_classes=28
bs=10

#categories
super_cat_ids = []
super_cat_last_name = ''
for cat_it in categories:
    super_cat_name = cat_it['supercategory']
    if super_cat_name != super_cat_last_name:
        super_cat_ids.append(super_cat_name) 
        super_cat_last_name = super_cat_name

#loading and organizing the data
data_ids={'batch_1':{},'batch_2':{},'batch_3':{},'batch_4':{},'batch_5':{},'batch_6':{},'batch_7':{},'batch_8':{},'batch_9':{},'batch_10':{},'batch_11':{},'batch_12':{},'batch_13':{},'batch_14':{},'batch_15':{}}
for i in imgs:
    r=i['file_name']
    data_ids['batch_'+str(r)[str(r).find('_') + 1:str(r).find('/')]][str(r)[str(r).find('/') + 1:]]=i['id']

data_class={}
for a in anns:
    coco_bbox=a['bbox']
    coco_bbox = BoundingBox.from_coco(*my_coco_box)  
    voc_bbox = coco_bbox.to_voc()  
    voc_bbox_values = coco_bbox.to_voc(return_values=True)
    voc_bbox_values=list(voc_bbox_values)
    data_class[a['image_id']].append([super_cat_ids[cat[a['category_id']]['supercategory']],voc_bbox_values)

#resize image with bboxes


#Applying Transforms to the Data
image_transforms = { 
    'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

#making an iterator to load the data
train_data_loader=[]
test_data_loader=[]
valid_data_loader=[]
                                     
for i in range(1,10,1):
    train_data_loader.append(DataLoader( datasets.ImageFolder(root='data/batch_'+str(i), transform=image_transforms['train']), batch_size=bs, shuffle=True))

train_data_loader.append(DataLoader( datasets.ImageFolder(root='data/batch_12', transform=image_transforms['train']), batch_size=bs, shuffle=True))
valid_data_loader.append(DataLoader( datasets.ImageFolder(root='data/batch_10', transform=image_transforms['valid']), batch_size=bs, shuffle=True))
valid_data_loader.append(DataLoader( datasets.ImageFolder(root='data/batch_11', transform=image_transforms['valid']), batch_size=bs, shuffle=True))
test_data_loader.append(DataLoader( datasets.ImageFolder(root='data/batch_13', transform=image_transforms['test']), batch_size=bs, shuffle=True))
test_data_loader.append(DataLoader( datasets.ImageFolder(root='data/batch_14', transform=image_transforms['test']), batch_size=bs, shuffle=True))
test_data_loader.append(DataLoader( datasets.ImageFolder(root='data/batch_15', transform=image_transforms['test']), batch_size=bs, shuffle=True))

#size of Data
train_data_size=1015
test_data_size=285
valid_data_size=200

