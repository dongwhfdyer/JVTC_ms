import os
import shutil


def copyfiles(copy_path, dstpath):
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    for file_path in os.listdir(copy_path):
        shutil.copyfile(os.path.join(copy_path, file_path), os.path.join(dstpath, file_path))


def prepare_data():
    source_folder1 = r"d:\download\bounding_box_train_camstyle_duke\bounding_box_train_camstyle_duke"
    source_folder2 = r"D:\ANewspace\code\35_JVTC\data\DukeMTMC-reID\DukeMTMC-reID\bounding_box_train"
    des_folder = r"D:\ANewspace\code\JVTC_ms\data"
    copyfiles(source_folder1, des_folder)
    copyfiles(source_folder2, des_folder)

if __name__ == '__main__':
    prepare_data()