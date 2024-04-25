import os
import shutil

def organize_images(base_path, base_folder_path, base_image_path):
    # List all text files in the base directory
    text_files = [f for f in os.listdir(base_path) if f.endswith('.txt')]
    
    for txt_file in text_files:
        folder_name = txt_file[:-4]  # Remove the '.txt' part to use as folder name
        folder_path = os.path.join(base_folder_path, folder_name)
        
        # Create a folder for the images if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Open the text file and read lines
        with open(os.path.join(base_path, txt_file), 'r') as file:
            lines = file.readlines()
        
        # Process each line, each line contains 'imagefilename -1/1'
        for line in lines:
            parts = line.split()
            image_name = parts[0] + '.jpg'
            
            if len(parts) != 2 or parts[1] != '1':
                continue
            # Move the image to the corresponding folder
            source_image_path = os.path.join(base_image_path, image_name)
            destination_image_path = os.path.join(folder_path, image_name)
            # print(f"image: {image_name}, dest: {destination_image_path}")
            if os.path.exists(source_image_path):
                shutil.copy(source_image_path, destination_image_path)
            else:
                print(f"Warning: {source_image_path} does not exist in the base path and cannot be moved.")

# Use the function
base_path = '../data/VOC2007/ImageSets/Main/'  # Set the path to your directory
base_image_path = '../data/VOC2007/JPEGImages/'  # Set the path to your directory
base_folder_path = '../data/images'  # Set the path to your directory
organize_images(base_path, base_folder_path, base_image_path)
