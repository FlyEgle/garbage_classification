import os
import torch
import shutil
import copy
import torchvision
import torch.nn as nn
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.transforms as transfrom
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet34, resnet50
from torchvision.models import inception_v3, vgg16_bn
from torchvision.models import alexnet
import torch.optim as optim
import torch.optim.lr_scheduler
import numpy as np
from PIL import Image
import math
from collections import OrderedDict
from torchstat import stat
from MobileNetV2 import MobileNetV2


class GenerateFeaturesModel:
    '''
        这个类构建提取特征的cnn模型
    '''
    def __init__(self, model_name):
        self.model_name = model_name
        
    def choice_model_by_name(self):
        if self.model_name == 'resnet34':
            return self.resnet_features()
        elif self.model_name == 'resnet50':
            return self.resnet_features()
        elif self.model_name == 'alexnet':
            return self.alexnet_features()
        elif self.model_name == 'mobilenet':
            return self.mobilenet_features()
        else:
            print('Sorry, we do not train this model')
        
    def alexnet_features(self):
        model = torch.load('model_weights/alexnet.pkl')
        get_features_model = model.features
        return get_features_model
    
    def mobilenet_features(self):
        model = torch.load('model_weights/mobilenet.pkl')
        get_features_model = model.features
        return get_features_model
    
    def resnet_features(self):
        resnet_get_features_model = GetResNetFeatures(self.model_name)
        return resnet_get_features


#输入为model与image
# 输出是概率
def image_outputs(model, image):
    model = model.eval()
    device = torch.device('cpu')
    model.to(device)
    # image is PIL object
    image_transfrom = transfrom.Compose([
        transfrom.Resize((224, 224)),
        transfrom.ToTensor(),
        transfrom.Normalize(mean_for_norm, std_for_norm)
    ])
    image_tensor = image_transfrom(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor)
#     _, predicted = torch.max(outputs.data, 1)
    return outputs

# Ensemble模型
# 使用mode进行选择，bagging与boosting两种模式
class EnsemebleModel:
    def __init__(self, mode, weights=(0.4, 0.5, 0.1)):
        self.resnet34 = torch.load('model_weights/resnet34.pkl')
        self.resnet50 = torch.load('model_weights/resnet50.pkl')
        self.mobilenet = torch.load('model_weights/mobilenet.pkl')
        self.mode = mode
        self.weights = weights
        
    def fit_transfrom(self, images):
        if self.mode == 'bagging':
            self.weights = None
            predicted = self.resnet34_50_mobilenet(images)
        elif self.mode == 'boosting':
            w1, w2, w3 = self.weights
            predicted = self.resnet34_resnet50_mobilenet_boost(images, w1, w2, w3)
        return predicted
            
    def resnet34_50_mobilenet(self, images):
        r34_pre = image_outputs(self.resnet34, images)
        r50_pre = image_outputs(self.resnet50, images)
        mob_pre = image_outputs(self.mobilenet, images)
        sum_pre = r34_pre + r50_pre + mob_pre
        bagging_pre = sum_pre / 3
        _, max_predicted = torch.max(bagging_pre.data, 1)
        return max_predicted
    
    def resnet34_resnet50_mobilenet_boost(self, images, w1, w2, w3):
        r34_pre = image_outputs(self.resnet34, images)
        r50_pre = image_outputs(self.resnet50, images)
        mob_pre = image_outputs(self.mobilenet, images)
        boosting_pre = w1 * r34_pre + w2 * r50_pre + w3 * mob_pre
        _, max_predicted = torch.max(boosting_pre.data, 1)
        return max_predicted


if __name__ == '__main__':
    ensemble_model = EnsemebleModel('boosting')
    ensemble_model.fit_transfrom(image)



    