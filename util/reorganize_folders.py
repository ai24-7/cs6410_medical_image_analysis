import os
import shutil

def reorganize_folders(base_folder_path):
    # List all relevant folders
    folders = [f for f in os.listdir(base_folder_path) if '_' in f]
    
    for folder in folders:
        parts = folder.split('_')
        if len(parts) != 2:
            continue
        
        item_type, group = parts  # e.g., 'cat', 'test'
        new_folder_path = os.path.join(base_folder_path, group)  # e.g., '../data/images/test'
        
        # Create the new parent folder if it doesn't exist
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
        
        # Determine new path for the current folder
        new_item_path = os.path.join(new_folder_path, item_type)  # e.g., '../data/images/test/cat'
        
        # Move and rename the folder
        source_folder_path = os.path.join(base_folder_path, folder)
        if os.path.exists(new_item_path):
            print(f"Warning: {new_item_path} already exists. Consider manual merging or cleanup.")
        else:
            shutil.move(source_folder_path, new_item_path)
            print(f"Moved {source_folder_path} to {new_item_path}")

# Use the function
base_folder_path = '../data/images'  # Update this path to where your specific folders are located
reorganize_folders(base_folder_path)
