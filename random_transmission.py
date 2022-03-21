import os
import shutil
import math
import random
 
#想要移动文件所在的根目录
rootdir="../jzsb"
#获取目录下文件名清单
train_val_path = os.path.join(rootdir,'test6')
test_path = os.path.join(rootdir,'test7')
list=os.listdir(train_val_path)
#print(files)
# #移动图片到指定文件夹
# for i in range(len(list)):     #遍历目录下的所有文件夹
#         if i > 0:
#             train_val = os.path.join(train_val_path,str(i))
#             test = os.path.join(test_path,str(i))
#             pic_all = os.listdir(train_val)
#             random.shuffle(pic_all)
#             len_limited = round(1/6*len(pic_all))
#             for j in range(len_limited):
#                 full_path = os.path.join(train_val,pic_all[j])
#                 des_path = os.path.join(test,str(j)+'.png')
#                 shutil.move(full_path,des_path)
for i in range(math.ceil(len(list)/2)):
        train_val = os.path.join(train_val_path,str(2*i))
        test = os.path.join(test_path,str(i))
        pic_all = os.listdir(train_val)
        # pic_index = random.shuffle(pic_all)
        # len_limited = round(1/6*len(pic_all))
        len_limited = len(pic_all)
        # pic_index = range(len(pic_all))
        # pic_index = random.shuffle(pic_index)
        for j in range(len_limited):
            full_path = os.path.join(train_val,pic_all[j])
            des_path = os.path.join(test,str(j)+'.png')
            shutil.copy(full_path,des_path)
        if 2*i+1 < len(list):
            train_val = os.path.join(train_val_path,str(2*i+1))
            test = os.path.join(test_path,str(i))
            pic_all = os.listdir(train_val)
            # pic_index = random.shuffle(pic_all)
            # len_limited = round(1/6*len(pic_all))
            len_limited2 = len(pic_all)
            # pic_index = range(len(pic_all))
            # pic_index = random.shuffle(pic_index)
            for j in range(len_limited2):

                full_path = os.path.join(train_val,pic_all[j])
                des_path = os.path.join(test,str(len_limited+j)+'.png')
                shutil.copy(full_path,des_path)