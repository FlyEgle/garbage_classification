'''
这个py文件主要是进行训练数据的生成和模型训练，可以选择不同种类的模型进行训练
'''
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


# 进行模型选择，目前训练的resnet34, 50, alexnet, mobilenet, vgg和inception由于
# 模型比较大没有训练
def model_choice(model_name, num_classes=6):
    if model_name == 'resnet34':
        model = resnet34(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif model_name == 'resnet50':
        model = resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif model_name == 'inception_v3':
        model = inception_v3(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif model_name == 'vgg16_bn':
        model = vgg16_bn(pretrained=False)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        return model
    elif model_name == 'alexnet':
        model = alexnet(pretrained=False)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        return model
    elif model_name == 'mobilenet':
        model = MobileNetV2(n_class=6)
        return model

# 计算训练集的mean与std，要先进行/255后在按通道进行计算
def get_image_mean_std(image):
    if type(image) != 'numpy.ndarray':
        image = np.array(image)
    image = image / 255.
    image_mean = []
    image_std = []
    for i in range(3):
        image_mean.append(np.mean(image[:,:,i]))
        image_std.append(np.std(image[:,:,i]))
    return image_mean, image_std

# 获取批量数据的mean与std
def get_training_mean_std(train_image_dataset):
    mean_ = []
    std_ = []
    for image_classes in train_image_dataset:
        for image_ in image_classes:
            image = Image.open(image_)
            mean, std = get_image_mean_std(image)
            mean_.append(mean)
            std_.append(std)
    return mean_, std_

# 计算mean和std
def calculate_mean_std(mean_list, std_list):
    sum_count = len(mean_list)
    sum_mean = np.sum(np.array(mean_list), axis=0)
    sum_std = np.sum(np.array(std_list), axis=0)
    mean = sum_mean / sum_count
    std = sum_std / sum_count
    return mean, std

# data loader 生成器
def generate_dataloader(data_path):
    data_transforms = transfrom.Compose([
        transfrom.Resize((224, 224)),
        transfrom.ToTensor(),
        transfrom.Normalize(mean_for_norm, std_for_norm)
    ])
    dataset = ImageFolder(data_path, transform=data_transforms)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_works)
    return data_loader

# 计算准确率
def calculate_accuracy(dataloader, device, model, state):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    return correct / total

# 训练模型
def training_(model, dataloader, device, optimizer, loss_fn, scheduler, model_save_name):
	tra_loader, val_loader = dataloader['tra'], dataloader['val']
	loss_each_epoch = []
	train_acc = []
	val_acc = []
	static_sum = 20  
	print('======Start Training！=======')
	for epoch in range(epochs):
		loss_each_dataloader = 0.0
		scheduler.step()
		sum_batch_count = 0        
		loss_each_ = []
		train_acc_each_ = []
		val_acc_each_ = []
		for i, data in enumerate(tra_loader, 0):
			images, labels = data
			images, labels = images.to(device), labels.to(device)
			optimizer.zero_grad()
			image_outputs = model(images)
			loss_ = loss_fn(image_outputs, labels)
			loss_.backward()
			optimizer.step()
			loss_each_dataloader += loss_.item()

			if i % static_sum == 19:
				sum_batch_count += 1
				avearge_loss_epoch = loss_each_dataloader / static_sum
				loss_each_.append(avearge_loss_epoch)
				train_correct = calculate_accuracy(tra_loader, device, model, 'train')
				val_correct = calculate_accuracy(val_loader, device, model, 'val')
				train_acc_each_.append(train_correct)
				val_acc_each_.append(val_correct)
				print('epoch{%d/%d}, training_loss=%.5f, train_acc=%.2f %%, val_acc=%.2f %%'%(epoch+1, epochs, avearge_loss_epoch, train_correct*100, val_correct*100))
				loss_each_dataloader = 0.0 
		loss_each_epoch.append(sum(loss_each_)/ 3)
		train_acc.append(sum(train_acc_each_) / 3.)
		val_acc.append(sum(val_acc_each_)/3.)
	print('Finish Training！')
	print('=======save model=======')
	torch.save(model, os.path.join('model_weights', model_save_name))
	loss_arr = np.array(loss_each_epoch)
	print('=======save loss========')
	np.save(os.path.join('loss_acc_', str(model_save_name) + '_loss.npy'), loss_arr)
	np.save(os.path.join('loss_acc_', str(model_save_name) + '_tra_acc.npy'), train_acc)
	np.save(os.path.join('loss_acc_', str(model_save_name) + '_val_acc.npy'), val_acc)

# 模型训练，选择模型
def model_train(model_name, dataloader, device, model_save_name):
    model = model_choice(model_name)
    model.to(device)
    # 优化器与损失函数
    criterion = nn.CrossEntropyLoss()
    adam_optim = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(adam_optim, step_size=20, gamma=0.5)
    training_(model, dataloader, device, adam_optim, criterion, scheduler, model_save_name) 

# 显示损失函数的曲线
def show_loss_cruve(loss_arr):
    epoch = [i+1 for i in range(loss_arr.shape[0])]
    plt.figure(figsize=(16, 8))
    plt.plot(epoch, loss_arr)
    plt.title('train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

# 显示准确率曲线的函数
def show_acc_cruve(tra_acc, val_acc):
    epoch = [i+1 for i in range(tra_acc.shape[0])]
    plt.figure(figsize=(16, 8))
    plt.plot(epoch, tra_acc, width=1.5, color='g', label='tra_acc')
    plt.plot(epoch, val_acc, width=1.5, color='b', label='val_acc')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(loc='best')
    plt.show()

# 测试图片
def test_images(model, image):
    model = model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    return predicted    

if __name__ == '__main__':

	dataset_path = 'dataset_train'
	for image_class in os.listdir(dataset_path):
	    if image_class =='.DS_Store':
	        continue
	    else:
	        train_data_list = []
	        for image in os.listdir(os.path.join(dataset_path, image_class)):
	            if image.split('.')[-1] in ['jpg', 'png', 'jpeg']:
	                train_data_list.append(dataset_path+'/'+image_class+'/'+image)
	        image_all_list.append(train_data_list)

	mean_, std_ = get_training_mean_std(train_data_list)
	mean_for_norm, std_for_norm = calculate_mean_std(mean_, std_)

	# 这里使用的是torchvision的数据增强，并把数据进行resize，统一到tensor中
	validation_transfrom = transfrom.Compose([
	        transfrom.Resize((224, 224)),
	        transfrom.ToTensor(),
	        transfrom.Normalize(mean_for_norm, std_for_norm)
	])

	# 生成数据集, dataset_train_path, dataset_validation_path都是数据存放的位置
	train_dataset = ImageFolder(dataset_train_path, transform=train_transfrom)
	validation_dataset = ImageFolder(dataset_validation_path, transform=validation_transfrom)

	# 这个是我自己训练时候的参数， 根据自己的配置进行更改
	batch_size = 32
	num_works = 4
	epochs = 50

	# 使用cuda， 如果有gpu就用cuda，没有就用cpu
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# 生成数据
	data_loader = generate_dataloader(dataset_agu_path)
	dataloader_dict = {'tra': data_loader,
                   		'val': validation_dataloader}

    # 训练模型实例
    model_train('alexnet', dataloader_dict, device1, 'alexnet.pkl')
    # model_train('mobilenet', dataloader_dict, device1, 'mobilenet.pkl')
    # 测试数据
    image = Image.open('garbage_dataset/train/glass/glass1.jpg')
    model = torch.load('model_weights/resnet50.pkl')
    outputs = test_images(model, image)
    print(outputs.data.item())

    # 计算正确率
    total_list = []
	for data_class in os.listdir('garbage_dataset/validation'):
	    class_list = []
	    for data in os.listdir(os.path.join('garbage_dataset/validation', data_class)):
	        try:
	            image = Image.open('garbage_dataset/validation/'+os.path.join(data_class, data))
	            predicted = test_images(model, image)
	            class_list.append(predicted)
	        except IOError:
	            print('io error')
	    total_list.append(class_list)

	sum_ = 0
	for num in total_list:
	    sum_ += len(num)

	num_list = [0, 1, 2, 3, 4, 5]
	error = 0
	for i in range(len(total_list)):
	    for data in total_list[i]:
	        if data != num_list[i]:
	            error += 1

	print('error: %.2f'%(error / sum_))











