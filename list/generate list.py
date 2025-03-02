import os

# 目标文件夹路径
folder_path = "/media/cw/584485FC4485DD5E/csh/TEVAD-main/save/Crime/clip_features_10crop_new"

# 获取目标文件夹中所有文件的绝对路径
files = []
for root, dirs, filenames in os.walk(folder_path):
    for filename in filenames:
        file_path = os.path.join(root, filename)
        files.append(file_path)

# 将文件路径写入.list文件
with open("list/ucf-clip-test-10crop.list", "w") as f:
    for file_path in files:
        f.write(file_path + "\n")