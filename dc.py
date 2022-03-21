import numpy as np
import pandas as pd
import os

from cv2 import imwrite,imread
from scipy.ndimage import median_filter
from skimage.measure import label, regionprops_table
from skimage.color import rgb2gray

data_root = '../jzsb/4'
save_root = '../jzsb/train&val6/4'
to_be_cropped_root = '../jzsb/to_be_cropped/4'
img_path = sorted([os.path.join(data_root, name) for name in os.listdir(data_root) if name.endswith('.png')])
length = len(img_path)
properties = ['bbox','convex_area']
for i in range(length):
    img_name = os.path.split(img_path[i])[-1]
    img_name = img_name.split(')')[0]
    img_name = img_name.split('(')[1]
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
        delta = min(int(0.7*max(width,height)),blob[1],blob[0],np.size(img,0)-1-blob[2],np.size(img,1)-1-blob[3])    
        side = max(width,height)+2*delta
        img_final = img[blob[0]-delta:blob[0]+side-delta,blob[1]-delta:blob[1]+side-delta,:]#转换很烧脑
        if '.png' not in img_name:
            img_name+='.png'
        print(img_name)
        if abs(np.size(img_final,0)-np.size(img_final,1))<5 and side>30:
            imwrite(os.path.join(save_root,img_name),img_final)
        else:
            imwrite(os.path.join(to_be_cropped_root,img_name),img_final)