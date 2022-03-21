import torch
import torchvision
from vgg import vgg11
from vggsmall import VGG11_small
from lenet import LeNet
from alexnet import AlexNet
from densenet import DenseNet
from resnet import ResNet,Bottleneck,BasicBlock
from googlenet import GoogLeNet
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from lr_scheduler import CosineAnnealingLR_Restart
from collections import OrderedDict

#如何试探可使用的设备
def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


#初始化参数，两种初始化方法，防止迭代初期速度过慢或者梯度爆炸
def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.001)


def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def set_parameter_requires_grad(net,feature_extracting):
    if feature_extracting:
        net = net
        for param in net.parameters():
            param.requires_grad = False

#定义训练函数
def train(train_iter, net, device, trainer1, trainer2, scheduler1,scheduler2, loss):
    net.train()
    for i, (pics, labels) in enumerate(train_iter):
        pics = pics.to(device)
        labels = labels.to(device)
        # l = loss(net(pics), labels)

        #googlenet专用loss计算
        logits, aux_logits2, aux_logits1 = net(pics)
        loss0 = loss(logits, labels)
        loss1 = loss(aux_logits1, labels)
        loss2 = loss(aux_logits2, labels)
        l = loss0 + loss1 * 0.3 + loss2 * 0.3

        # trainer.zero_grad()
        # l.backward()
        # trainer.step()

        trainer1.zero_grad()
        trainer2.zero_grad()
        l.backward()
        trainer1.step()
        trainer2.step()
        # scheduler.step()
        scheduler1.step()
        scheduler2.step()


        if i % 100 == 0:
            print('batch:{},loss:{}'.format(i + 1, l.data))
    return l


#定义测试函数
def test(test_iter, net, device):
    net.eval()
    with torch.no_grad():
        acc = 0
        total_num = 0
        for pics, labels in test_iter:
            pics = pics.to(device)
            labels = labels.to(device)
            outs = net(pics)
            # _, pre = torch.max(outs.data, 1)  #输出行向量表示每行的最大值，并且返回索引
            pre = torch.max(outs, dim=1)[1] #googlenet以及更小网络使用
            total_num += labels.size(0)
            acc += (pre == labels).sum().item()
        print('Accuracy：{}'.format(acc / total_num))
        return acc / total_num


#选择使用的设备
device = try_gpu()
print(device)

#采用resnet50网络，调整输出种类数量
# num_classes = 5  # 共有5类
# net = vgg11()

# net = VGG11_small()

# net = LeNet()

# net = AlexNet(num_classes=5)

net = GoogLeNet(num_classes=3)
# net = GoogLeNet(num_classes=5)

# net = DenseNet(growth_rate=12,block_config=(6, 12, 24, 16),num_init_features=24,num_classes=5,bn_size=8)
# net = DenseNet(growth_rate=32,block_config=(6, 12, 24, 16),num_init_features=64,num_classes=5,bn_size=8)


# net = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=5) #resnet101
# net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=5) #resnet50
# net = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=5) #resnet34

# net = torchvision.models.resnet50(
#     pretrained=False)  #应用于模型的迁移以及微调等等,imagenet和这里面的无关

# num_ftrss = net.conv1.out_channels
# net.conv1 = nn.Conv2d(1,
#                       num_ftrss,
#                       kernel_size=7,
#                       stride=2,
#                       padding=3,
#                       bias=False)
# set_parameter_requires_grad(net,True)
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, num_classes)  #改变全连接层的大小，为了能导入数据
# net.to(device)

pth_path='./SpermGoogLeNet.pth'
load_net = torch.load(pth_path, map_location=device)
load_net_trained = OrderedDict()
for k, v in load_net.items():
    if 'fc' not in k:
        load_net_trained[k] = load_net[k]
net.load_state_dict(load_net_trained, strict=False)
# net.load_state_dict(torch.load('SpermGoogLeNet.pth',map_location=device))
# num_classes2 = 5
# num_ftrs = net.fc.in_features
# net.fc = nn.Linear(num_ftrs, num_classes2)  #改变全连接层的大小，为了应用于新的模型
# num_init_features = 64
# net.features = nn.Sequential(OrderedDict([
#             ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2,
#                                 padding=3, bias=False)),
#             ('norm0', nn.BatchNorm2d(num_init_features)),
#             ('relu0', nn.ReLU(inplace=True)),
#             ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
#         ]))
# net.load_state_dict(torch.load('SpermResnet.pth',map_location=device))
# xavier(net.classifier)                              #仅仅对最后一层进行初始化
# xavier(net.features)
net.to(device)


# net.apply(xavier)

#通过get_mean_std.py提前计算好了数据的均值与方差

# mean = [0.819,0.691,0.811]
# std = [0.123,0.209,0.108]
# mean = 0.7
# std = 0.2

batch_size = 8
num_epochs = 20
loss = nn.CrossEntropyLoss()
trainer = torch.optim.Adam(net.parameters(), lr=0.001)

optim_param1 = []
optim_param2 = []
for k, v in  net.named_parameters():
    if 'fc' in k:
        optim_param2.append(v)
    else :
         optim_param1.append(v)

trainer1 = torch.optim.Adam(optim_param1, lr=0.0001)
trainer2 = torch.optim.Adam(optim_param2, lr=0.001)
#50轮一次
# T_period = [17500, 17500, 17500, 17500]
# restarts = [17500, 35000, 52500]
# weights = [1,1,1]
#5轮一次
T_period = [1750, 1750, 1750, 1750]
restarts = [1750, 3500, 5250]
weights = [1,1,1]

scheduler1 = CosineAnnealingLR_Restart(trainer1,
                          T_period=T_period,
                          restarts=restarts,
                          weights=weights,
                          eta_min=1e-6)

scheduler2 = CosineAnnealingLR_Restart(trainer2,
                          T_period=T_period,
                          restarts=restarts,
                          weights=weights,
                          eta_min=1e-5)

scheduler = CosineAnnealingLR_Restart(trainer,
                                T_period=T_period,
                                restarts=restarts,
                                weights=weights,
                                eta_min=1e-5)
#定义图片预处理方法,导入数据
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
])

#在训练集和验证集中应用k折交叉验证
# split_num = 5 #K折是几折
# folds = KFold(n_splits=split_num,shuffle=True)
loss_array = np.zeros((1, num_epochs))  #后面可以画图可视化
train_acc_array = np.zeros((1, num_epochs))  #后面可以画图可视化
test_acc_array = np.zeros((1, num_epochs))  #后面可以画图可视化

#导入数据
rootdir = '../jzsb'
train_val_dir = os.path.join(rootdir, 'train&val7')
test_dir = os.path.join(rootdir, 'test7')
train_val_dataset = ImageFolder(train_val_dir, train_transform)
train_test_dataset = ImageFolder(train_val_dir, val_transform)
test_dataset = ImageFolder(test_dir, val_transform)
train_iter = DataLoader(dataset=train_val_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=8)
train_test_iter = DataLoader(dataset=train_test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=8)
test_iter = DataLoader(dataset=test_dataset,
                       batch_size=batch_size,
                       shuffle=False,
                       num_workers=8)

#采用K折交叉训练，个人感觉有效减少过拟合
for epoch in range(num_epochs):
    print('epoch:{}'.format(epoch + 1))
    loss_array[0, epoch] = train(train_iter, net, device, trainer1, trainer2, scheduler1,scheduler2, loss)
    # loss_array[0, epoch] = train(train_iter, net, device, trainer1,trainer2, loss)
    train_acc_array[0, epoch] = test(train_test_iter, net, device)
    test_acc_array[0, epoch] = test(test_iter, net, device)
    if test_acc_array[0, epoch] == np.max(test_acc_array):
        torch.save(net.state_dict(), 'SpermGoogLeNet_3type.pth')
   
#将数组扁平化
loss_array = loss_array.flatten()
train_acc_array = train_acc_array.flatten()
test_acc_array = test_acc_array.flatten()
x = range(1, num_epochs + 1)

best_acc = np.max(test_acc_array)
print(best_acc)
#训练过程可视化
plt.plot(x, loss_array, color='red', label='train_loss')
plt.plot(x, train_acc_array, color='blue', label='train_acc')
plt.plot(x, test_acc_array, color='green', label='test_acc')
plt.xlabel('epoch')
plt.title('train and val result')
plt.show()

