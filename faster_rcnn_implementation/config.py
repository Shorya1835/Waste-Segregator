import torch

#Hyperparameters
BATCH_SIZE=7
RESIZE_TO=(256,224)
NUM_EPOCHS=500
NUM_WORKERS=0

#Choosing between cpu or gpu
DEVICE=torch.device(('cuda') if torch.cuda.is_available() else torch.device('cpu'))

#Training images
TRAIN_DIR='resized_data/train'

#Validation images 
VALID_DIR='resized_data/validation'

#classes: 0 index is reserved for background
CLASSES=[
    '__background__','Aluminium foil', 'Battery', 'Blister pack', 'Bottle', 'Bottle cap', 'Broken glass', 'Can', 'Carton', 'Cup', 'Food waste', 'Glass jar', 'Lid', 'Other plastic', 'Paper', 'Paper bag', 'Plastic bag & wrapper', 'Plastic container', 'Plastic glooves', 'Plastic utensils', 'Pop tab', 'Rope & strings', 'Scrap metal', 'Shoe', 'Squeezable tube', 'Straw', 'Styrofoam piece', 'Unlabeled litter', 'Cigarette']

NUM_CLASSES=len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

# location to save model and plots
OUT_DIR = 'outputs'



    
