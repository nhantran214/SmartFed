import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('./')

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

'''
Initialize Dataset
'''

class genimage(Dataset):
    def __init__(self,root,dataidxs=None,train=True,transform=None, target_transform=None, noise_type=None, noise_rate=0, size=(256, 256)):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.size = size
        self.data, self.target = self.Construct_Participant_Dataset()

    def Construct_Participant_Dataset(self):       
        if self.train:
            data = []
            labels = []
            train_real_p = 'train/nature'
            train_fake_p = 'train/ai'
            train_real_path = os.path.join(self.root, train_real_p)
            train_fake_path = os.path.join(self.root, train_fake_p)

            for filename in os.listdir(train_real_path):
                file_path = os.path.join(train_real_path, filename)
                # print('train_real_path', file_path)
                try:
                    # Open the image file
                    with Image.open(file_path) as img:
                        img_resized = img.resize(self.size)
                        data.append(img_resized.copy())  # Copy to prevent closing the image after `with`
                        labels.append(0)
                except (IOError, OSError) as e:
                    print(f"Error opening {filename}: {e}")
            print('train_real_data', train_real_path, len(data))
            for filename in os.listdir(train_fake_path):
                file_path = os.path.join(train_fake_path, filename)
                # print('train_fake_path', file_path)
                try:
                    # Open the image file
                    with Image.open(file_path) as img:
                        img_resized = img.resize(self.size)
                        data.append(img_resized.copy())  # Copy to prevent closing the image after `with`
                        labels.append(1)
                except (IOError, OSError) as e:
                    print(f"Error opening {filename}: {e}")
            print('train_fake_data', train_fake_path, len(data))
        else:
            data = []
            labels = []
            test_real_p = 'val/nature'
            test_fake_p = 'val/ai'
            test_real_path = os.path.join(self.root, test_real_p)
            test_fake_path = os.path.join(self.root, test_fake_p)

            for filename in os.listdir(test_real_path):
                file_path = os.path.join(test_real_path, filename)
                # print('test_real_path', file_path)
                try:
                    # Open the image file
                    with Image.open(file_path) as img:
                        img_resized = img.resize(self.size)
                        data.append(img_resized.copy())  # Copy to prevent closing the image after `with`
                        labels.append(0)
                except (IOError, OSError) as e:
                    print(f"Error opening {filename}: {e}")
            print('test_real_data', test_real_path, len(data))
            for filename in os.listdir(test_fake_path):
                file_path = os.path.join(test_fake_path, filename)
                # print('test_fake_path', file_path)
                try:
                    # Open the image file
                    with Image.open(file_path) as img:
                        img_resized = img.resize(self.size)
                        data.append(img_resized.copy())  # Copy to prevent closing the image after `with`
                        labels.append(1)
                except (IOError, OSError) as e:
                    print(f"Error opening {filename}: {e}")
            print('test_fake_data', test_fake_path, len(data))
        select_data = []
        select_labels= []
        if self.dataidxs is not None:
            for i in self.dataidxs:
                select_data.append(data[i])
                select_labels.append(labels[i])
        else:
            select_data = data
            select_labels = labels
        print('data.shape, labels.shape', len(data), len(labels))
        return select_data, select_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

class celebdf2(Dataset):
    def __init__(self,root,dataidxs=None,train=True,transform=None, target_transform=None, noise_type=None, noise_rate=0, size=(224, 224)):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.size = size
        self.data, self.target = self.Construct_Participant_Dataset()

    def Construct_Participant_Dataset(self):       
        if self.train:
            data = []
            labels = []
            train_real_p = 'train/real'
            train_fake_p = 'train/fake'
            train_real_path = os.path.join(self.root, train_real_p)
            train_fake_path = os.path.join(self.root, train_fake_p)

            for filename in os.listdir(train_real_path):
                file_path = os.path.join(train_real_path, filename)
                # print('train_real_path', file_path)
                try:
                    # Open the image file
                    with Image.open(file_path) as img:
                        img_resized = img.resize(self.size)
                        data.append(img_resized.copy())  # Copy to prevent closing the image after `with`
                        labels.append(0)
                except (IOError, OSError) as e:
                    print(f"Error opening {filename}: {e}")
            print('train_real_data', train_real_path, len(data))
            for filename in os.listdir(train_fake_path):
                file_path = os.path.join(train_fake_path, filename)
                # print('train_fake_path', file_path)
                try:
                    # Open the image file
                    with Image.open(file_path) as img:
                        img_resized = img.resize(self.size)
                        data.append(img_resized.copy())  # Copy to prevent closing the image after `with`
                        labels.append(1)
                except (IOError, OSError) as e:
                    print(f"Error opening {filename}: {e}")
            print('train_fake_data', train_fake_path, len(data))
        else:
            data = []
            labels = []
            test_real_p = 'test/test_real'
            test_fake_p = 'test/test_fake'
            test_real_path = os.path.join(self.root, test_real_p)
            test_fake_path = os.path.join(self.root, test_fake_p)

            for filename in os.listdir(test_real_path):
                file_path = os.path.join(test_real_path, filename)
                # print('test_real_path', file_path)
                try:
                    # Open the image file
                    with Image.open(file_path) as img:
                        img_resized = img.resize(self.size)
                        data.append(img_resized.copy())  # Copy to prevent closing the image after `with`
                        labels.append(0)
                except (IOError, OSError) as e:
                    print(f"Error opening {filename}: {e}")
            print('test_real_data', test_real_path, len(data))
            for filename in os.listdir(test_fake_path):
                file_path = os.path.join(test_fake_path, filename)
                # print('test_fake_path', file_path)
                try:
                    # Open the image file
                    with Image.open(file_path) as img:
                        img_resized = img.resize(self.size)
                        data.append(img_resized.copy())  # Copy to prevent closing the image after `with`
                        labels.append(1)
                except (IOError, OSError) as e:
                    print(f"Error opening {filename}: {e}")
            print('test_fake_data', test_fake_path, len(data))
        select_data = []
        select_labels= []
        if self.dataidxs is not None:
            for i in self.dataidxs:
                select_data.append(data[i])
                select_labels.append(labels[i])
        else:
            select_data = data
            select_labels = labels
        print('data.shape, labels.shape', len(data), len(labels))
        return select_data, select_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

if __name__=='__main__':
    noise_type = None
    noise_rate = 0
    dataset_root = '/home/mcsp/mcsp/ddim/data/genimage/adm'
    genim_train = genimage(root=dataset_root,noise_type=noise_type, noise_rate=noise_rate)
    genim_test = genimage(root=dataset_root,train=False)
    print(len(genim_train), len(genim_test))