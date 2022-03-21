import numpy as np
import pandas as pd
import os

from cv2 import imwrite,imread
from scipy.ndimage import median_filter
from skimage.measure import label, regionprops_table
from skimage.color import rgb2gray

data_root = '../jzsb/0'
cate = 0
img_path = sorted([os.path.join(data_root, name) for name in os.listdir(data_root) if name.endswith('.png')])
length = len(img_path)
properties = ['bbox','convex_area']
for i in range(length):
    img_name = os.path.split(img_path[i])[-1]
    img_name = img_name.split('.')[0]
    if '.txt' not in img_name:
        filename= os.path.join(data_root, img_name + '.txt')
    img = imread(img_path[i])
    img_gray = rgb2gray(img)
    img_gray = (img_gray-img_gray.mean())/img_gray.std()
    img_gray = (img_gray-img_gray.min())/(img_gray.max()-img_gray.min())
    mask_gray = img_gray< 0.5
    mask_gray = median_filter(mask_gray,25)
    sperm_blobs = label(mask_gray>0)
    df = pd.DataFrame(regionprops_table(sperm_blobs ,properties=properties))
    blob_coordinates = [(row['bbox-0'],row['bbox-1'],
                     row['bbox-2'],row['bbox-3'] ,row['convex_area'])for 
                    index, row in df.iterrows()]
    blob_coordinates = np.array(blob_coordinates)
    if blob_coordinates.ndim == 2:
        blob_id = np.argmax(blob_coordinates[:,4]) #选取最大面积连通图
        blob = blob_coordinates[blob_id,:]
        width = blob[3] - blob[1]
        height = blob[2] - blob[0]
        delta = min(int(0.3*max(width,height)),blob[1],blob[0],np.size(img,0)-1-blob[2],np.size(img,1)-1-blob[3])    
        side = max(width,height)+2*delta
        img_final = img[blob[0]-delta:blob[2]+delta,blob[1]-delta:blob[3]+delta,:]#转换很烧脑
        
        if side>30 :
            x = ((blob[3] + blob[1])/2.0)/(np.size(img,1)-1)
            y = ((blob[2] + blob[0])/2.0)/(np.size(img,0)-1)
            w = side/(np.size(img,1)-1)
            h = side/(np.size(img,0)-1)
            f = open(filename,'w')
            f.write('{} {:.6f} {:.6f} {:.6f} {:.6f}'.format(cate,x,y,w,h))
            f.close()
        else:
            print(filename)
    else:
        print(filename)
