import sys
sys.path.append('./')

from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as transforms

class CIFAR10Dataset(Dataset):
    """CIFAR-10 Dataset class similar to genimage dataset"""
    
    def __init__(self, root, dataidxs=None, train=True, transform=None, 
                 target_transform=None, noise_type=None, noise_rate=0, size=(32, 32)):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.size = size
        
        # CIFAR-10 class names
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
        self.data, self.target = self.construct_dataset()
    
    def construct_dataset(self):
        """Construct the dataset from image files"""
        data = []
        labels = []
        
        if self.train:
            data_dir = os.path.join(self.root, 'train')
        else:
            data_dir = os.path.join(self.root, 'test')
        
        print(f"Loading CIFAR-10 {'train' if self.train else 'test'} data from: {data_dir}")
        
        # Load images from each class directory
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found!")
                continue
            
            # Get all image files in the class directory
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Found {len(image_files)} images for class {class_name}")
            
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                try:
                    # Load and resize image
                    with Image.open(img_path) as img:
                        img_resized = img.resize(self.size)
                        data.append(img_resized.copy())
                        labels.append(class_idx)
                except (IOError, OSError) as e:
                    print(f"Error opening {img_file}: {e}")
        
        print(f"Total loaded: {len(data)} images, {len(labels)} labels")
        
        # Apply data selection if dataidxs is provided
        if self.dataidxs is not None:
            selected_data = []
            selected_labels = []
            for i in self.dataidxs:
                if i < len(data):
                    selected_data.append(data[i])
                    selected_labels.append(labels[i])
            data = selected_data
            labels = selected_labels
            print(f"After selection: {len(data)} images")
        
        return data, labels
    
    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    def __len__(self):
        return len(self.data)
    
    def get_class_name(self, class_idx):
        """Get class name by index"""
        if 0 <= class_idx < len(self.class_names):
            return self.class_names[class_idx]
        return "unknown"

def get_cifar10_transforms():
    """Get standard CIFAR-10 transforms"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return transform_train, transform_test

def visualize_cifar10_samples(dataset, num_samples=10):
    """Visualize sample images from the dataset"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('CIFAR-10 Sample Images', fontsize=16)
    
    for i in range(min(num_samples, len(dataset))):
        img, label = dataset[i]
        
        # Convert tensor back to image if needed
        if hasattr(img, 'numpy'):
            img = img.numpy()
            if img.shape[0] == 3:  # CHW format
                img = np.transpose(img, (1, 2, 0))
            img = (img * 0.5 + 0.5)  # Denormalize
            img = np.clip(img, 0, 1)
        else:
            img = np.array(img)
        
        row = i // 5
        col = i % 5
        axes[row, col].imshow(img)
        axes[row, col].set_title(f'{dataset.get_class_name(label)} (idx: {label})')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Test the dataset
    dataset_root = '../data/cifar'
    
    # Create dataset without transforms for visualization
    train_dataset = CIFAR10Dataset(
        root=dataset_root,
        train=True,
        transform=None,
        dataidxs=None
    )
    
    test_dataset = CIFAR10Dataset(
        root=dataset_root,
        train=False,
        transform=None,
        dataidxs=None
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Visualize some samples
    if len(train_dataset) > 0:
        visualize_cifar10_samples(train_dataset, 10)
    
    # Test with transforms
    transform_train, transform_test = get_cifar10_transforms()
    
    train_dataset_with_transforms = CIFAR10Dataset(
        root=dataset_root,
        train=True,
        transform=transform_train,
        dataidxs=None
    )
    
    print("Dataset with transforms created successfully!")
    print(f"Sample tensor shape: {train_dataset_with_transforms[0][0].shape}")
    print(f"Sample label: {train_dataset_with_transforms[0][1]}")
