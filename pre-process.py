import os

import numpy as np
import imageio

from skimage.io import imshow, imread
import matplotlib.pyplot as plt
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor

def color_map(N=256, normalized=True):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def color_map_viz():
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
    nclasses = 21
    row_size = 50
    col_size = 500
    cmap = color_map()
    array = np.empty((row_size*(nclasses+1), col_size, cmap.shape[1]), dtype=cmap.dtype)
    for i in range(nclasses):
        array[i*row_size:i*row_size+row_size, :] = cmap[i]
    array[nclasses*row_size:nclasses*row_size+row_size, :] = cmap[-1]
    
    imshow(array)
    plt.yticks([row_size*i+row_size/2 for i in range(nclasses+1)], labels)
    plt.xticks([])
    plt.show()


def color_map_label(normalized=True):
    labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
    nclasses = 21
    cmap = color_map()
    label_colors = np.empty((nclasses + 1, 3), dtype=cmap.dtype)  # +1 for the 'void' label

    for i in range(nclasses + 1):
        label_colors[i] = cmap[i] if i < nclasses else cmap[-1]

    return label_colors




def preprocess_and_save(image_path, mask_path, cmap, output_size=(224, 224), output_dir='output', log = False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load and resize image
    # image = imread(image_path)

    # # Load and resize mask
    # mask_image = imread(mask_path)
    
    image = imageio.imread(image_path)
    mask_image = imageio.imread(mask_path)
    if log:
        plot_sample(image, mask_image, title=f"{image_path}")

    mask_image = resize(mask_image, output_size, anti_aliasing=False, preserve_range=True).astype(int)
    image = resize(image, output_size, anti_aliasing=True) / 255.0  # Normalize to [0, 1]


    # Map color to channel index
    color_to_index = {tuple(val): idx for idx, val in enumerate(cmap)}

    # Initialize 22-channel binary mask
    channels = np.zeros((*output_size, 22), dtype=np.float32)

    # Vectorized mask processing
    # for idx, val in enumerate(cmap):
    #     channels[:, :, idx] = np.all(mask_image == val, axis=-1)

    channels = np.zeros((*output_size, 22), dtype=np.float32)  # Assuming 22 classes including background
    for idx, color in enumerate(cmap):
        mask = np.all(mask_image == np.array(color*255.0, dtype=int), axis=-1)
        if log:
            print(mask)    
        channels[:, :, idx] = mask
    if log:
        plot_sample(image, channels, title=f"Sample")
        # Use this in your preprocessing function to log unique colors of some masks
        print("Unique colors in mask:", unique_colors(mask_image))

    # Save image and mask to binary files
    image_filename = os.path.join(output_dir, os.path.basename(image_path) + '_image.npy')
    mask_filename = os.path.join(output_dir, os.path.basename(mask_path) + '_mask.npy')
    np.save(image_filename, image)
    np.save(mask_filename, channels)

def process_dataset(image_dir, mask_dir, cmap):
    file_names = os.listdir(image_dir)
    tasks = []
    count = 1
    log = False
    with ThreadPoolExecutor(max_workers=4) as executor:
        for file_name in file_names:
            count+=1
            if count > 10:
                log = False
            if file_name.endswith('.jpg'):
                mask_name = file_name[:-4] + '.png'  # Change extension for mask
                mask_path = os.path.join(mask_dir, mask_name)
                image_path = os.path.join(image_dir, file_name)

                if os.path.exists(mask_path):
                    tasks.append(executor.submit(preprocess_and_save, image_path, mask_path, cmap, (224, 224), 'output', log))

    # Optional: wait for all tasks to complete and handle exceptions
    for task in tasks:
        task.result()  # This will raise any exceptions caught during the thread execution

# Assume color_map function is defined and provides the cmap array
cmap = color_map_label()  # Assuming this function returns the correct RGB values for each class

image_dir = 'data/VOC2007/JPEGImages'
mask_dir = 'data/VOC2007/SegmentationClass'
process_dataset(image_dir, mask_dir, cmap)
