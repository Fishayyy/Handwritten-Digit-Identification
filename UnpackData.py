import os
import zipfile

def unpack_data(train_test):

    target_directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

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
  
# Function to rename multiple files
cwd = os.getcwd()
if not os.path.isdir(f"{cwd}/Data"):
    print("Extracting Data.zip...")
    with zipfile.ZipFile(f"{cwd}/Data.zip", 'r') as zip_ref:
        zip_ref.extractall(cwd)

unpack_data("train")
unpack_data("test")
