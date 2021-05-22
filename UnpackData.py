import os
import zipfile
import cv2
import warnings

warnings.filterwarnings('ignore')
primary_directories = ['test', 'train']
secondary_directories = ['greyscale','binary']
target_directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

cwd = os.getcwd()

# Function to rename multiple files
if not os.path.isdir(f"{cwd}/Data"):
    print("Extracting Data.zip...")
    with zipfile.ZipFile(f"{cwd}/Data.zip", 'r') as zip_ref:
        zip_ref.extractall(cwd)

def unpack_data(train_test):

    for dir in target_directories:
        path = os.path.realpath(f'Data/{train_test}/{dir}/')

        #clear previous names
        for i, filename in enumerate(os.listdir(path)):
            new = f"{path}\\{i}.png"
            old = f"{path}\\{filename}"
            os.rename(old, new)
        
        #assign new easily indexed names
        for i, filename in enumerate(os.listdir(path)):
            new = f"{path}\\{dir}_{i}_{train_test}.tif"
            old = f"{path}\\{filename}"
            os.rename(old, new)

def initial_preprocessing():
    save_path = f'{cwd}\\processed_images\\'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    for dir1 in primary_directories:
        for dir2 in target_directories:
            path = os.path.realpath(f'Data/{dir1}/{dir2}/')
            save_path = f'{cwd}\\processed_images\\{dir1}\\'
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            save_path = f'{cwd}\\processed_images\\{dir1}\\{secondary_directories[0]}\\'
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            for filename in os.listdir(path):
                # Read in Image
                filepath = f"{path}\\{filename}"
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

                # Save image
                cv2.imwrite(os.path.join(save_path, filename), img)





unpack_data("train")
unpack_data("test")
initial_preprocessing()
