import torch
import torchvision.models

from torch import nn
from torchvision import transforms
from PIL import Image
from googlenet import GoogLeNet


class predict:
    def __init__(self,device,net,val_transform):
        self.device = device
        self.net = net.to(device)
        self.val_transform = val_transform
    
    def try_gpu(self,i=0):
        if torch.cuda.device_count()>=i+1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')


    def predict(self,pic):
        self.net.eval()
        with torch.no_grad():
            pic = self.val_transform(pic) #记得要归一化大小
            pic = pic.unsqueeze(0) #由于原本的测试数据集为多张图片，因此是四维张量，需要升维来保持稳定性
            pic = pic.to(self.device)
            out = self.net(pic)
            pre = torch.max(out,1)[1]
        
        return pre.item()

    def test(self,test_iter):
        self.net.eval()
        with torch.no_grad():
            pres = torch.tensor([])
            for pics,labels in test_iter:
                pics = pics.to(self.device)
                outs = self.net(pics)
                _,pre = torch.max(outs.data,1)#输出行向量表示每行的最大值，并且返回索引
                if(pres.shape[0]==0):
                    pres = pre
                else:
                    pres = torch.cat((pres,pre),0)
            # print('Accuracy：{}'.format(acc/total_num))
            return pres

if __name__ == "__main__":
    pic = Image.open('/home/public/Desktop/测试/15-0019.png').convert('RGB')
    def try_gpu(i=0):
        if torch.cuda.device_count()>=i+1:
            return torch.device(f'cuda:{i}')
        return torch.device('cpu')

    device = try_gpu(0)
    num_classes = 3 # 共有5类精子
    net = GoogLeNet(num_classes=3)
    net.to(device)
    net.load_state_dict(torch.load('model_param/SpermGoogLeNet_3type.pth',map_location=device))

    val_transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                ])
    mypredict = predict(device,net,val_transform)
    print('ready')
    pre = mypredict.predict(pic)
    print('prediction:{}'.format(pre.label))
