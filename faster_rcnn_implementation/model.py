import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes):

    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    #freeze the backbone (it will freeze the body and fpn params)
    for p in model.backbone.parameters():
        p.requires_grad=False

    #freeze the fc6 layer in roi_heads
    for p in model.roi_heads.box_head.fc6.parameters():
        p.requires_grad=False
    
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model
