########################################################################################################
### Stage 1, split the train set into train-val, and write the csv
########################################################################################################


import os
import csv
import pandas as pd
import shutil

# Define the name of the folder
folder_name = "annotations"
# Check if the folder already exists
if not os.path.exists(folder_name):
    # Create the folder
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully.")
else:
    print(f"Folder '{folder_name}' already exists.")




file_name = "driver_imgs_list.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(file_name)


# Check the number of images 
num_rows = df.shape[0]
print(f"Number of Images in the folder: {num_rows}")
split_ratio = 0.9
train_volume = int(num_rows * split_ratio)
# print("Training set volumne: ", train_volume, "Validation set Volume: ", num_rows-train_volume)


shuffled_df = df.sample(frac=1)
train_set = shuffled_df[:train_volume]
val_set  = shuffled_df[train_volume:]

print("Train set shape:", train_set.shape)
print("Test set shape:", val_set.shape)


train_set = train_set.sort_index() # sort the DataFrame by index in ascending order
val_set = val_set.sort_index()   
print(train_set.head())
print(val_set.head())


train_set.to_csv('annotations/train.csv', index=False)
val_set.to_csv('annotations/validation.csv', index=False)

########################################################################################################
### Stage 2, split the data based on stage 1's random train-val split
########################################################################################################

# Define the name of the folder
folder_name = "annotations/DMD"
# Check if the folder already exists
if not os.path.exists(folder_name):
    # Create the folder
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully.")
else:
    print(f"Folder '{folder_name}' already exists. The old data has to be deleted for the second time run as train-val are randomly selected")
    # define the folder to delete'

    # loop through all the files and folders in the folder
    for root, dirs, files in os.walk(folder_name, topdown=False):
        for file_name in files:
            # delete the file
            file_path = os.path.join(root, file_name)
            os.remove(file_path)
        for dir_name in dirs:
            # delete the folder and its contents
            dir_path = os.path.join(root, dir_name)
            shutil.rmtree(dir_path)

    # delete the top-level folder
    shutil.rmtree(folder_name)



# Define the name of the folder
folder_name = "annotations/DMD/train"
# Check if the folder already exists
if not os.path.exists(folder_name):
    # Create the folder
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully.")
else:
    print(f"Folder '{folder_name}' already exists.")

    folder_name = "annotations/DMD/validation"
# Check if the folder already exists
if not os.path.exists(folder_name):
    # Create the folder
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' created successfully.")
else:
    print(f"Folder '{folder_name}' already exists.")


for idx in df.index:
    if idx in val_set.index:
        destinationfolder= 'annotations/DMD/validation/'+val_set["classname"][idx]
        if not os.path.exists(destinationfolder):
            # Create the folder
            os.makedirs(destinationfolder)
            print(f"Folder '{destinationfolder}' created successfully.")

        shutil.copy('train/'+val_set["classname"][idx]+'/'+val_set["img"][idx], destinationfolder)
    else:
        destinationfolder= 'annotations/DMD/train/'+train_set["classname"][idx]
        if not os.path.exists(destinationfolder):
            # Create the folder
            os.makedirs(destinationfolder)
            print(f"Folder '{destinationfolder}' created successfully.")

        shutil.copy('train/'+train_set["classname"][idx]+'/'+train_set["img"][idx], destinationfolder)
