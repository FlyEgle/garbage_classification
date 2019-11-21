# 这个py文件主要是用来进行集成学习的预测
# 包括，boosting和bagging
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
from sklearn.svm import SVC
from sklearn.externals import joblib


# 重新实现一个nn model来作为特征提取的model Resnet34
class GetFeatures(nn.Module):
    def __init__(self, model):
        super(GetFeatures, self).__init__()
        self.model = model
        self.conv1 = self.model.conv1
        self.bn1 = self.model.bn1
        self.relu = self.model.relu
        self.maxpool = self.model.maxpool
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# 定义一个特征提取器
# inputs: 特征提取模型和PIL图片
# outputs: 提取图片的features
def get_cnn_features_extract(get_features_model, image):
    get_features_model = get_features_model.eval()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    get_features_model.to(device)
    # image is PIL object
    image_transfrom = transfrom.Compose([
        transfrom.Resize((224, 224)),
        transfrom.ToTensor(),
        transfrom.Normalize(mean_for_norm, std_for_norm)
    ])
    with torch.no_grad():
        image_tensor = image_transfrom(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        features = get_features_model(image_tensor)
        features = features.view(1, -1)
        return features

def translate_numpy(list_tensor):
    new_list = []
    for tensor in list_tensor:
        new_list.append(tensor.numpy())
    return new_list

from tqdm import tqdm
def get_tra_images_featuers(data_path, model):
    total_features_ = []
    total_classes_ = []
    classes_ = 0
    for data_classes in os.listdir(data_path):
        for data_ in tqdm(os.listdir(data_path + '/' + data_classes)):
            try:
                image = Image.open(data_path + '/' + data_classes + '/' + data_)
                features = get_cnn_features_extract(model, image)
                total_features_.append(features)
                total_classes_.append(classes_)
            except IOError:
                print(data_path + '/' + data_classes + '/' + data_)
        classes_ += 1
    return total_features_, total_classes_

if __name__ == '__main__':
	data_path = 'garbage_dataset_agument/train'
	total_features, total_classes = get_tra_images_featuers(data_path, get_features)
	features_list = translate_numpy(total_features
	features_arr = np.array(features_list)
	features_arr = features_arr.reshape(features_arr.shape[0], -1)
	labels_arr = np.array(total_classes)
	labels_arr = labels_arr.reshape(-1, )
	classifiy = SVC(gamma='auto')
	classifiy.fit(features_arr, labels_arr)
	joblib.dump(classifiy, 'svm_train_features_traagu.m')
	test_image_pth = 'garbage_dataset/validation/glass/glass456.jpg'
	test_image = Image.open(test_image_pth)
	test_features = get_cnn_features_extract(get_features, test_image)
	test_features = test_features.numpy()
	classifiy.predict(test_features)




