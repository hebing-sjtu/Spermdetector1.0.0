import torch
from torchvision import transforms
from googlenet import GoogLeNet
from PySide2.QtWidgets import QApplication
from PySide2.QtUiTools import QUiLoader
from PySide2.QtWidgets import QFileDialog
from PySide2 import QtGui
import numpy as np
import os
from figure_out import figure_out

def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

device = try_gpu(0)
num_classes = 3 # 共有5类精子
net = GoogLeNet(num_classes=3)
net.to(device)
net.load_state_dict(torch.load('SpermGoogLeNet_3type.pth',map_location=device))

val_transform = transforms.Compose([transforms.Resize((224,224)),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor(),
                            ])

class Stats:

    def __init__(self):
        # 从文件中加载UI定义
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load('spermdetector.ui')
        self.img = torch.zeros(2,2,2)
        self.img_handled = torch.zeros(2,2,2)
        self.num = self.ui.spinBox.value()
        self.filePaths = ""
        self.filePath = ""
        self.filePathhandled = ""
        self.label= np.array([[]])
        self.result= np.array([])
        self.figure_out = figure_out(net,val_transform)
        self.catagory_num = 3
        self.ui.textEdit.setPlaceholderText('结果显示')
        self.ui.pushButton.clicked.connect(self.detectbutten)
        self.ui.pushButton_2.clicked.connect(self.openimage)
        self.ui.pushButton_3.clicked.connect(self.openimages)
        self.ui.spinBox.valueChanged.connect(self.changeimage)

    def detectbutten(self):
        #对图像进行处理
        self.label = self.figure_out.img_handle(self.filePath, os.path.abspath(os.path.join(self.filePath,"..")))
        #处理后框图的图片路径以及
        self.filePathhandled = os.path.abspath(os.path.join(self.filePath,"..","handled",(self.filePath.split('/')[-1]).split('.')[0]+'_handled.png'))
        #显示框圈出的图像
        img_scaled = QtGui.QPixmap(self.filePathhandled).scaled(self.ui.label_2.width(), self.ui.label_2.height())
        self.ui.label_2.setPixmap(img_scaled)
        self.result = np.sum(self.label,1)
        self.ui.textEdit.setText('预测结果为：\t一类精子： {} 个\t二类精子： {} 个\t共有精子： {} 个 \t一类精子比例： {} '.format(self.result[0],self.result[1],np.sum(self.result[0:self.catagory_num-1]),self.result[0]/np.sum(self.result[0:self.catagory_num-1])))

    def openimage(self):
        self.filePath, _ = QFileDialog.getOpenFileName(
            self.ui,  # 父窗口对象
            "选择你要检测的图片",  # 标题
            "",  # 起始目录
            "图片类型 (*.png *jpg)"  # 选择类型过滤项，过滤内容在括号中
        )
        #读取图片并且将其在指定label处显示
        #torch image : C*H*W
        #numpy image : H*W*C
        # self.img = torch.from_numpy(imread(imgName))
        img_scaled = QtGui.QPixmap(self.filePath).scaled(self.ui.label.width(), self.ui.label.height())
        self.ui.label.setPixmap(img_scaled)


    def openimages(self):
        self.filePaths, _ = QFileDialog.getOpenFileNames(
            self.ui,  # 父窗口对象
            "选择你要检测的图片",  # 标题
            "",  # 起始目录
            "图片类型 (*.png *jpg)"  # 选择类型过滤项，过滤内容在括号中
        )
        self.filePath = self.filePaths[self.num]
        # self.img = torch.from_numpy(imread(filename))
        img_scaled = QtGui.QPixmap(self.filePath).scaled(self.ui.label.width(), self.ui.label.height())
        self.ui.label.setPixmap(img_scaled)

    def changeimage(self):
        self.num = self.ui.spinBox.value()
        self.filePath = self.filePaths[self.num]
        # self.img = torch.from_numpy(imread(filename))
        img_scaled = QtGui.QPixmap(self.filePath).scaled(self.ui.label.width(), self.ui.label.height())
        self.ui.label.setPixmap(img_scaled)

app = QApplication([])
stats = Stats()
stats.ui.show()
app.exec_()
#
# self.filePath = QFileDialog.getExistingDirectory(self.window, "选择图片集路径")
# self.filePath2, _ = QFileDialog.getOpenFileName(
#     self.window,  # 父窗口对象
#     "选择你要上传的图片",  # 标题
#     r"C:\Users\86186\Desktop\jupyter\data\pokemon",  # 起始目录
#     "图片类型 (*.png *.jpg *.bmp)"  # 选择类型过滤项，过滤内容在括号中
# )