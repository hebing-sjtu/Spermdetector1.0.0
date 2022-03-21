from ast import Str
import pandas as pd
from xml.etree.ElementInclude import default_loader
from PIL import Image
import torch
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset,DataLoader,Subset
import numpy as np
import os
from resnet import ResNet,Bottleneck
from densenet import DenseNet
from googlenet import GoogLeNet

def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

#定义测试函数
def test(test_iter,net,device):
    net.eval()
    with torch.no_grad():
        acc = 0
        total_num = 0
        for pics,labels in test_iter:
            pics = pics.to(device)
            labels = labels.to(device)
            outs = net(pics)
            _,pre = torch.max(outs.data,1)#输出行向量表示每行的最大值，并且返回索引
            total_num+=labels.size(0)
            acc+=(pre==labels).sum().item()
        # print('Accuracy：{}'.format(acc/total_num))
        return acc/total_num

class Mydataset(Dataset):
    def __init__(self,root,loader = default_loader):
        super().__init__()
        self.images = [os.path.join(root,each) for each in os.listdir(root)]
        self.classification = int(os.path.split(root)[-1])
        self.target = [self.classification for each in self.images]
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        fn = Image.open(fn)
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)

class Mydataset2(Dataset):
    def __init__(self,root,label,loader = default_loader):
        super().__init__()
        self.images = [os.path.join(root,each) for each in os.listdir(root)]
        self.classification = label
        self.target = [self.classification for each in self.images]
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        fn = Image.open(fn)
        img = self.loader(fn)
        target = self.target[index]
        return img,target

    def __len__(self):
        return len(self.images)
#选择使用的设备
device = try_gpu()

#采用resnet50网络，调整输出种类数量
# num_classes = 5 # 共有5类
# net = torchvision.models.resnet50(pretrained = False) #应用于模型的迁移以及微调等等
# #如果处理灰度图像则需要改变第一次卷集形式
# # num_ftrss = net.conv1.out_channels
# # net.conv1 = nn.Conv2d(1,
# #                       num_ftrss,
# #                       kernel_size=7,
# #                       stride=2,
# #                       padding=3,
# #                       bias=False)
                     
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs,num_classes) #改变全连接层的大小
# net.to(device)
# net = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=5)

net = GoogLeNet(num_classes=3)

# net = DenseNet(growth_rate=12,num_init_features=24,num_classes=5,bn_size=8,drop_rate=0.5)

net.load_state_dict(torch.load('model_param/SpermGoogLeNet_3type.pth',map_location=device))
# net.apply(xavier)
net.to(device)

#通过get_mean_std.py提前计算好了数据的均值与方差
mean = [0.694,0.468,0.704]
std = [0.151,0.216,0.130]
# mean = 0.7
# std = 0.2
batch_size = 8
num_epochs = 1000
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(),lr = 0.01)

val_transform = transforms.Compose([transforms.Resize((224,224)),
                               #看模型是否是灰度模型来决定
                               transforms.Grayscale(num_output_channels=1),
                               transforms.ToTensor(),
                            #    transforms.Normalize(mean,std)
                               ])

def default_loader(path):
    img_pil =  Image.open(path)
    img_tensor = val_transform(img_pil)
    return img_tensor

#导入数据
rootdir = '../jzsb'
train_val_dir = os.path.join(rootdir,'train&val7')
classify_num = len(os.listdir(os.path.join(rootdir,'test7')))
acc = np.array([])
# for i in range(classify_num):
#     test_dir = os.path.join(rootdir,'test6',str(i))
#     # test_dataset = ImageFolder(test_dir, val_transform)
#     test_dataset = Mydataset(test_dir, loader = val_transform)
#     test_iter = DataLoader(dataset = test_dataset,
#                         batch_size = batch_size,
#                         shuffle = False,
#                         num_workers = 8)
#     acc.append(test(test_iter,net,device))猜是四类
for i in range(classify_num):
    for j in range(classify_num):
        test_dir = os.path.join(rootdir,'test7',str(i))
        # test_dataset = ImageFolder(test_dir, val_transform)
        test_dataset = Mydataset2(test_dir, j,loader = val_transform)
        test_iter = DataLoader(dataset = test_dataset,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 8)
        acc = np.append(acc,test(test_iter,net,device))
acc = acc.reshape((-1,classify_num))
# properties = ['猜是一类','二类','三类','四类','杂质']
properties = ['猜是一类','二类','杂质']
df = pd.DataFrame(acc,columns=properties)
print(df)