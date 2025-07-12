import os
import random
import shutil

folder_files = 'yolo_labels'
txtfile = [f for f in os.listdir(folder_files) if f.endswith(('.txt'))]

random.shuffle(txtfile)

val_list = txtfile[0:94]
test_list = txtfile[94:188]
train_list = txtfile[188:936]

os.mkdir('train')
os.mkdir('val')
os.mkdir('test')
os.mkdir('train_labels')
os.mkdir('val_labels')
os.mkdir('test_labels')

repository = "images"
biaozhu = "yolo_labels"
#分类图片
for imagename in train_list:
    source = os.path.join(repository,imagename.split('.')[0]+".jpg")
    shutil.move(source,'train')
    source = os.path.join(biaozhu,imagename.split('.')[0]+".txt")
    shutil.move(source,'train_labels')
for imagename in val_list:
    source = os.path.join(repository,imagename.split('.')[0]+".jpg")
    shutil.move(source,'val')
    source = os.path.join(biaozhu,imagename.split('.')[0]+".txt")
    shutil.move(source,'val_labels')
for imagename in test_list:
    source = os.path.join(repository,imagename.split('.')[0]+".jpg")
    shutil.move(source,'test')
    source = os.path.join(biaozhu,imagename.split('.')[0]+".txt")
    shutil.move(source,'test_labels')


