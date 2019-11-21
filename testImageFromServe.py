'''
测试图片的API
输入的是图片的路径，采用Image进行读取。
使用mobilenet进行输出图片类别
'''
import torch
import torchvision
import torchvision.transforms as transfrom
import numpy as np
from PIL import Image


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
    _, predicted = torch.max(outputs.data, 1)
    return predicted.data.item()


def test_image(image_path, model_weights_path):
	'''
	input: image_path for test
		   model_weights_path
	return: image_classes
	'''
	class_dict = {
		0:'cardboard',
		1:'glass',
		2:'metal',
		3:'paper',
		4:'plastic',
		5:'trash'
	}
	model = torch.load(model_weights_path)
	images = Image.open(image_path)
	output = image_outputs(model, images)
	classes = class_dict[output]
	return classes






