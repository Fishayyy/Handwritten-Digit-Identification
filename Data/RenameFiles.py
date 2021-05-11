import os

target_directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  
# Function to rename multiple files
cwd = os.getcwd() 

for dir in target_directories:
    path = os.path.realpath(f'Data/{dir}/')

    #clear previous names
    for i, filename in enumerate(os.listdir(path)):
        new = f"{path}\\{i}.png"
        old = f"{path}\\{filename}"
        os.rename(old, new)
    
    #assign new easily indexed names
    for i, filename in enumerate(os.listdir(path)):
        new = f"{path}\\{dir}_{i}.png"
        old = f"{path}\\{filename}"
        os.rename(old, new)