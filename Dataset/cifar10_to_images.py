import os
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def unpickle(file):
    """Unpickle CIFAR-10 data file"""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def create_cifar10_images():
    """Convert CIFAR-10 pickle files to images and save them"""
    
    # Define paths
    data_dir = '../data/cifar-10-batches-py'
    output_dir = '../data/cifar'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create subdirectories for each class
    for class_name in class_names:
        train_dir = os.path.join(output_dir, 'train', class_name)
        test_dir = os.path.join(output_dir, 'test', class_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    
    print("Processing CIFAR-10 training data...")
    
    # Process training data (data_batch_1 to data_batch_5)
    for batch_num in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{batch_num}')
        print(f"Processing {batch_file}...")
        
        if os.path.exists(batch_file):
            batch_data = unpickle(batch_file)
            
            # Extract data and labels
            data = batch_data[b'data']
            labels = batch_data[b'labels']
            filenames = batch_data[b'filenames']
            
            # Reshape data from (10000, 3072) to (10000, 3, 32, 32)
            data = data.reshape(10000, 3, 32, 32)
            
            # Process each image
            for i in range(len(data)):
                # Get image data and convert from CHW to HWC
                img_data = data[i].transpose(1, 2, 0)
                
                # Convert to PIL Image
                img = Image.fromarray(img_data)
                
                # Get class name and filename
                class_name = class_names[labels[i]]
                filename = filenames[i].decode('utf-8')
                
                # Save image
                output_path = os.path.join(output_dir, 'train', class_name, filename)
                img.save(output_path)
    
    print("Processing CIFAR-10 test data...")
    
    # Process test data (test_batch)
    test_file = os.path.join(data_dir, 'test_batch')
    if os.path.exists(test_file):
        test_data = unpickle(test_file)
        
        # Extract data and labels
        data = test_data[b'data']
        labels = test_data[b'labels']
        filenames = test_data[b'filenames']
        
        # Reshape data from (10000, 3072) to (10000, 3, 32, 32)
        data = data.reshape(10000, 3, 32, 32)
        
        # Process each image
        for i in range(len(data)):
            # Get image data and convert from CHW to HWC
            img_data = data[i].transpose(1, 2, 0)
            
            # Convert to PIL Image
            img = Image.fromarray(img_data)
            
            # Get class name and filename
            class_name = class_names[labels[i]]
            filename = filenames[i].decode('utf-8')
            
            # Save image
            output_path = os.path.join(output_dir, 'test', class_name, filename)
            img.save(output_path)
    
    print("CIFAR-10 data conversion completed!")
    print(f"Images saved to: {output_dir}")
    
    # Print statistics
    print("\nDataset statistics:")
    for class_name in class_names:
        train_count = len(os.listdir(os.path.join(output_dir, 'train', class_name)))
        test_count = len(os.listdir(os.path.join(output_dir, 'test', class_name)))
        print(f"{class_name}: {train_count} train, {test_count} test images")

def visualize_sample_images():
    """Visualize some sample images from the converted dataset"""
    
    output_dir = '../data/cifar'
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create a figure with sample images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('CIFAR-10 Sample Images', fontsize=16)
    
    for i, class_name in enumerate(class_names):
        # Get a sample image from training set
        train_dir = os.path.join(output_dir, 'train', class_name)
        if os.path.exists(train_dir):
            images = os.listdir(train_dir)
            if images:
                img_path = os.path.join(train_dir, images[0])
                img = Image.open(img_path)
                
                # Plot image
                row = i // 5
                col = i % 5
                axes[row, col].imshow(img)
                axes[row, col].set_title(class_name)
                axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('../data/cifar/sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Sample visualization saved to: ../data/cifar/sample_images.png")

if __name__ == '__main__':
    # Convert CIFAR-10 pickle files to images
    create_cifar10_images()
    
    # Visualize sample images
    visualize_sample_images()
