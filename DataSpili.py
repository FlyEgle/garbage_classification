'''
这个py文件主要是对数据集进行划分以及
数据增强
'''
import os 
import shutil
import numpy as np  
import math  
from PIL import Image  
import copy


# 定义函数来划分数据为train与validation，
# ratio 是验证数据的百分比
def get_train_validation_dataset(data_image_list, ratio=0.2):
    validation_classes_list = []
    for image_classes_list in data_image_list:
        length_validation = int(len(image_classes_list) * ratio)
        validation_image_list = []
        i = 0
        for _ in range(length_validation):
            validation_image_list.append(image_classes_list.pop())
        validation_classes_list.append(validation_image_list)
    return validation_classes_list

# 切分数据集
def data_split(image_list, classes_folder):
    for image_classes in image_list:
        for image in image_classes:
            image_class = image.split('/')[1]
            image_name = image.split('/')[-1]
            shutil.copyfile(image, os.path.join(classes_folder, image_class)+'/'+image_name)
        print(image_class)

# 数据集分布情况
def show_raw_dataset_distribute(image_all_list):
    x_label = [i for i in range(len(name_classes_list))]
    y_data = []
    for data in image_all_list:
        y_data.append(len(data))
    plt.bar(x_label, y_data, tick_label=name_classes_list, width=0.5)
    plt.title('Raw DataSet')
    plt.show()

# 数据划分后的分布情况
def show_dataset_distribute(train_data_list, validation_data_list):
    x_label = np.array([i for i in range(len(name_classes_list))], dtype=np.float)
    y_train = []
    y_validation = []
    for data in train_data_list:
        y_train.append(len(data))
    for data in validation_data_list:
        y_validation.append(len(data))
    plt.bar(x_label, y_train, tick_label=name_classes_list, width=0.3, color='g', label='train data')
    plt.bar(x_label+0.3, y_validation, tick_label=name_classes_list, width=0.3, color='y', label='test_data')
    plt.title('DataSet Distribution')
    plt.legend(loc='best')
    plt.show()


# 定一个一个数据增强的类别
class DataAugment:
    def __init__(self, image_path):
        self.image = Image.open(image_path)
        self.image = self.image.resize((256, 256))
        
    def rotate(self, angle):
        return self.image.rotate(angle)
    
    def hflip(self):
        return self.image.transpose(Image.FLIP_LEFT_RIGHT)
    
    def vflip(self):
        return self.image.transpose(Image.FLIP_TOP_BOTTOM)

if __name__ == '__main__':
	# 数据集的路径根据自己的路径存放位置来定，数据集的一级目录即可
	dataset_path = 'dataset-resized'
	# 存储数据到list，方便后续操作和读取
	image_all_list = []

	for image_class in os.listdir(dataset_path):
	    if image_class =='.DS_Store':
	        continue
	    else:
	        image_classes_list = []
	        for image in os.listdir(os.path.join(dataset_path, image_class)):
	            if image.split('.')[-1] in ['jpg', 'png', 'jpeg']:
	                image_classes_list.append(dataset_path+'/'+image_class+'/'+image)
	        image_all_list.append(image_classes_list)

	# 类别
	name_classes_list = []
	for name_ in os.listdir(dataset_path):
	    if name_ != '.DS_Store':
	        name_classes_list.append(name_)


	# 类别字典
	classes_dict = {}
	for i in range(len(name_classes_list)):
	    classes_dict[i] = name_classes_list[i]

	# 按照之前定义的ratio来进行切分数据集
	# 这个是新的数据的存放位置
	dataset_path = 'garbage_dataset'
	dataset_train_path = os.path.join(dataset_path, 'train')
	dataset_validation_path = os.path.join(dataset_path, 'validation')

	image_data_list = copy.deepcopy(image_all_list)
	validation_data_list = get_train_validation_dataset(image_data_list)
	train_data_list = image_data_list
	#TODO：这里需要添加切分好的数据集再进行增强

	# 本地增强，保存图片
	for image_classes in os.listdir(dataset_train_path):
	    data_train_agu_path = os.path.join(dataset_agu_path, image_classes)
	    if not os.path.exists(data_train_agu_path):
	        os.mkdir(data_train_agu_path)
	    for image_data in os.listdir(os.path.join(dataset_train_path, image_classes)):
	        path = os.path.join(dataset_train_path, image_classes)
	        try:
	            data_augment = DataAugment(os.path.join(path, image_data))
	            img_rotate = data_augment.rotate(90)
	            img_hflip = data_augment.hflip()
	            img_vflip = data_augment.vflip()
	            img_rotate.save(data_train_agu_path+'/'+image_data.split('.')[0]+'_'+'rot.jpg')
	            img_hflip.save(data_train_agu_path+'/'+image_data.split('.')[0]+'_'+'hflp.jpg')
	            img_vflip.save(data_train_agu_path+'/'+image_data.split('.')[0]+'_'+'vflp.jpg')
	            data_augment.image.save(os.path.join(data_train_agu_path, image_data))
	        except IOError:
	            print("may somthing is not Image object, don't care")

	# 显示原始数据的分布情况
	show_raw_dataset_distribute(image_all_list)
	
	# 显示验证数据的分布情况
	show_raw_dataset_distribute(validation_data_list)
	# 显示train与val数据的分布
	show_dataset_distribute(train_data_list, validation_data_list)







