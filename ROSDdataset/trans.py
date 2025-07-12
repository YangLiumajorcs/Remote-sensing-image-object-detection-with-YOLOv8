import os
from PIL import Image

# 图片文件夹路径
folder_path = "images"

# 读取文件夹中的所有图片文件名
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# 创建一个空列表来保存图像对象
images = []

# 逐个读取图片文件并保存到变量中
for filename in image_files:
    file_path = os.path.join(folder_path, filename)
    image = Image.open(file_path)
    images.append(image)
images = sorted(images, key=lambda x: os.path.basename(x.filename))

# 现在，你可以通过索引访问这些图像对象，例如：images[0]、images[1]，以此类推。


# 定义类别映射关系
class_map = {
    'aircraft': 0,
    'oiltank': 1,
    'overpass': 2,
    'playground': 3
}

# 读取ground truth文件夹中的标签文件
label_dir = "labels"
img_dir = "images"
output_dir = "yolo_labels"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
index = -1
label_dir_sort = sorted(os.listdir(label_dir))

#label_dir_sort.remove(".DS_Store")
for filename in label_dir_sort:
    index += 1
    with open(os.path.join(label_dir, filename), 'r') as file:
        lines = file.readlines()
        print(filename)
        print(images[index].filename)
        testname = filename.split('.')
        if testname[0] not in images[index].filename:
            print('error')
            break
        yolo_labels = []
        for line in lines:
            if line != '' and line != '\n':
                parts = line.strip().split('\t')
                print(line)
                print(parts)
                parts = [x for x in parts if x != '']
                print(parts)
                class_index = int(class_map[parts[1]])
                # 归一化坐标
                image_len,image_high = images[index].size
                print(images[index].size)
                x_center = round((float(parts[2]) + float(parts[4])) / (2.0*image_len),6)
                y_center = round((float(parts[3]) + float(parts[5])) / (2.0*image_high),6)
                width = round((float(parts[4]) - float(parts[2]))/image_len,6)
                height = round((float(parts[5]) - float(parts[3]))/image_high,6)
            
                yolo_labels.append(f"{class_index} {x_center} {y_center} {width} {height}")
        
        # 将YOLO格式的标签写入文件
        output_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(output_dir, output_filename), 'w') as output_file:
            output_file.write('\n'.join(yolo_labels))
