import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
from PIL import Image
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.measure import label, regionprops_table
from scipy.ndimage import median_filter
from matplotlib.patches import Rectangle
from predict import predict
from torch.utils.data import Dataset,DataLoader
import torch
import time
from torchvision import transforms
from googlenet import GoogLeNet

val_transform = transforms.Compose([transforms.Resize((224,224)),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor(),
                            ])
def default_loader(imgs):
    img_tensor = val_transform(imgs)
    return img_tensor

class Mydataset(Dataset):
    def __init__(self,imgs,loader = default_loader):
        super().__init__()
        # self.images = [os.path.join(root,each) for each in os.listdir(root)]
        # self.classification = int(os.path.split(root)[-1])
        # self.target = [self.classification for each in self.images]
        self.images = imgs
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        return img,0

    def __len__(self):
        return len(self.images)


class figure_out:
    def __init__(self,net,val_transform) :
        self.device = try_gpu(0)
        self.predict = predict(self.device,net,val_transform)
        

    def config_init(self) :
        img_gray = rgb2gray(self.img)
        img_gray = (img_gray-img_gray.mean())/img_gray.std()
        img_gray = (img_gray-img_gray.min())/(img_gray.max()-img_gray.min())
        mask_gray = img_gray[:,:] < 0.4
        self.mask_gray = median_filter(mask_gray,25)
    
    def img_handle(self,img_dir,img_root):
        tic = time.time()
        properties = ['bbox','convex_area']
        save_root = 'handled'
        self.img = imread(img_dir)
        img_name = os.path.split(img_dir)[-1]
        self.img_name = img_name.split('.')[0]
        self.config_init()
        sperm_blobs = label(self.mask_gray>0)
        df = pd.DataFrame(regionprops_table(sperm_blobs ,properties=properties))
        blob_coordinates = [(row['bbox-0'],row['bbox-1'],
                     row['bbox-2'],row['bbox-3'] ,row['convex_area'])for 
                    index, row in df.iterrows()]
        blob_coordinates = np.array(blob_coordinates)
        fig, ax = plt.subplots(1,1)
        print(blob_coordinates.shape)
        # deltas = []
        # sides = []
        # img_croppeds = []
        # blobs =[]
        labels=[]
        for i,blob in enumerate(blob_coordinates):
            width = blob[3] - blob[1]
            height = blob[2] - blob[0]
            delta = min(int(0.8*max(width,height)),blob[1],blob[0],np.size(self.img,0)-1-blob[2],np.size(self.img,1)-1-blob[3])    
            side = max(width,height)+2*delta
            img_cropped = Image.fromarray(self.img[blob[0]-delta:blob[0]-delta+side,blob[1]-delta:blob[1]-delta+side,:])
            if abs(np.size(img_cropped,0)-np.size(img_cropped,1))<15 and side>30:
        #         if deltas == []:
        #             deltas = [delta]
        #             sides = [side]
        #             img_croppeds = [img_cropped]
        #             blobs = [blob]
        #         else:
        #             deltas.append(delta)
        #             sides.append(side)
        #             img_croppeds.append(img_cropped)
        #             blobs.append(blob)
        # toc = time.time()
        # print('圈出图片中精子耗时： ',toc-tic)
        # tic = time.time()
        # batch_size = 32
        # test_dataset =  Mydataset(img_croppeds)
        # test_iter = DataLoader(dataset = test_dataset,
        #                     batch_size = batch_size,
        #                     shuffle = False,
        #                     num_workers = 8)
        # labels = self.predict.test(test_iter)
        # for i,blob in enumerate(blobs):
                # pre = labels[i].item()
                # if pre == 0:
                #     color = 'g'
                # elif pre == 1:
                #     color = 'y'
                # else:
                #     color = 'r'
                # patch = Rectangle((blob[1]-deltas[i],blob[0]-deltas[i]),
                #                 sides[i],sides[i],edgecolor = color,facecolor='none')
                pre = self.predict.predict(img_cropped)
                if pre == 0:
                    color = 'g'
                elif pre == 1:
                    color = 'y'
                else:
                    color = 'r'
                labels.append(pre)
                patch = Rectangle((blob[1]-delta,blob[0]-delta),
                                side,side,edgecolor = color,facecolor='none')
                ax.add_patch(patch)
                ax.imshow(self.img)
                ax.set_axis_off()
        final_labels = np.zeros((3,len(labels)))
        for k,j in enumerate(labels):
            final_labels[j, k] = 1 
        plt.savefig(os.path.join(img_root,save_root,self.img_name+'_handled.png'),bbox_inches='tight')
        toc = time.time()
        print('图像处理共耗时： ',toc-tic)
        return final_labels

def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

if __name__ == "__main__":
    pic_dir = "/home/public/Desktop/测试/15-0019.png"
    pic_root = "/home/public/Desktop/测试"
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
    figure_out1 = figure_out(net,val_transform)
    figure_out1.img_handle(pic_dir,pic_root)