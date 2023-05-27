import pandas as pd
import fire

def data_cleanup():
    meta_filename = "camelyon17_v1.0/metadata.csv"
    df = pd.read_csv(meta_filename)
    df = df.reset_index(drop=True)

    # Drop bad first column
    to_drop = ["Unnamed: 0", "index", "level_0"]
    for d in to_drop:
        if d in df.columns:
            df = df.drop(columns=[d])

    # Rename splits
    df["split"] = df["split"].replace({0: "train", 1: "test"})

    df.to_csv(meta_filename, index=False)
   
def celebA_cleanup():
    meta_filename = "/Users/manukastratta/Developer/CS329D/test-time-training-project/data/CelebA/splits/manuka_setting1_val.csv"
    df = pd.read_csv(meta_filename)
    df = df.reset_index(drop=True)
    df = df.rename(columns={"index": "img_filename"})
    # Drop bad first column
    to_keep = ["img_filename", "Black_Hair"]
    for d in df.columns:
        if d not in to_keep:
            df = df.drop(columns=[d])
    
    def change_neg1(row):
        if row["Black_Hair"] == -1:
            return 0
        elif row["Black_Hair"] == 1:
            return 1
        raise Exception("Invalid col value")
    
    df["Black_Hair"] = df.apply(change_neg1, axis=1)
    print(df)
    df.to_csv(meta_filename, index=False)

def celebA_clean_cols():
    meta_filename = "data/CelebA/splits/setting1_test.csv"
    df = pd.read_csv(meta_filename)
    df = df.reset_index(drop=True)
     # Drop bad first column
    to_keep = ["img_filename", "Male", "Black_Hair"]
    for d in df.columns:
        if d not in to_keep:
            df = df.drop(columns=[d])
    df.to_csv(meta_filename, index=False)

def add_image_path():
    meta_filename = "data/camelyon17_v1.0/metadata.csv"
    df = pd.read_csv(meta_filename)

    def get_image_path(row):
        patient_str_num = str(int(row["patient"])).zfill(3) # from 4 --> 004
        node_str = str(row["node"])
        dir = f"patient_{patient_str_num}_node_{node_str}"
        x_coord, y_coord = str(row["x_coord"]), str(row["y_coord"])
        filename = f"patch_patient_{patient_str_num}_node_{node_str}_x_{x_coord}_y_{y_coord}.png"
        complete_filename = "patches" + "/" + dir + "/" + filename
        return complete_filename
        
    df["image_path"] = df.apply(get_image_path, axis=1)
    print(df)
    df.to_csv("data/camelyon17_v1.0/metadata_with_filenames.csv", index=False)

    


def data_stats():
    meta_filename = "data/camelyon17_v1.0/metadata.csv"
    df = pd.read_csv(meta_filename)
    print(df)

    # Print unique values for columns
    for col in df.columns:
        print(f"Unique values for {col}:")
        print(df[col].unique())

    # Print data split information
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    n_train = len(df_train)
    n_test = len(df_test)
    n_total = len(df)
    print("")
    print("Num train: ", n_train)
    print("Num test: ", n_test)
    print("Train/total = ", n_train / n_total)

    # what centeres presented in each split
    hospitals_train = df_train['center'].value_counts()
    hospitals_test = df_test['center'].value_counts()
    print("hospitals_train: ", hospitals_train)
    print("hospitals_test: ", hospitals_test)

def create_new_splits():
    meta_filename = "data/camelyon17_v1.0/metadata.csv"
    df = pd.read_csv(meta_filename)
    
    df_train_distribution = df[df["center"].isin([0, 1, 2])]
    df_val_distribution = df[df["center"].isin([3])]
    df_test_distribution = df[df["center"].isin([4])]
    n_train = len(df_train_distribution)
    n_val = len(df_val_distribution)
    n_test = len(df_test_distribution)
    assert (n_train + n_val + n_test) == len(df)
    print("n_train, n_val, n_test: ", n_train, n_val, n_test)
    print("n_train / total: ", n_train / len(df))
    print("n_val / total: ", n_val / len(df))
    print("n_test / total: ", n_test / len(df))

    df_train_distribution["split"] = "train"
    df_val_distribution["split"] = "valid"
    df_test_distribution["split"] = "test"

    df_train_distribution.to_csv("data/camelyon17_v1.0/wilds_splits/metadata_train.csv", index=False)
    df_val_distribution.to_csv("data/camelyon17_v1.0/wilds_splits/metadata_valid.csv", index=False)
    df_test_distribution.to_csv("data/camelyon17_v1.0/wilds_splits/metadata_test.csv", index=False)

def get_debug_dataset():
    meta_filename = "data/CelebA/splits/setting1_train.csv"
    df = pd.read_csv(meta_filename)
    df = df.sample(n=50)
    df = df.reset_index(drop=True)
    df.to_csv("data/CelebA/splits/debug_setting1_train.csv", index=False)



def split_debug_into_two_files():
    meta_filename = "data/camelyon17_v1.0/metadata_debug.csv"
    df = pd.read_csv(meta_filename)
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    df_train.to_csv("data/camelyon17_v1.0/debug/train/metadata_debug_train.csv", index=False)
    df_test.to_csv("data/camelyon17_v1.0/debug/test/metadata_debug_test.csv", index=False)

def rename_split():
    val_filename = "/Users/manukastratta/Developer/CS329D/test-time-training-project/data/camelyon17_v1.0/train/metadata_val_split.csv"
    df = pd.read_csv(val_filename)
    df["split"] = "val"
    df.to_csv(val_filename, index=False)

# def create_new_metadata():
#     train = "/Users/manukastratta/Developer/CS329D/test-time-training-project/data/camelyon17_v1.0/train/metadata_train_split.csv"
#     val = "/Users/manukastratta/Developer/CS329D/test-time-training-project/data/camelyon17_v1.0/train/metadata_val_split.csv"
#     test = "/Users/manukastratta/Developer/CS329D/test-time-training-project/data/camelyon17_v1.0/test/metadata_test.csv"


def create_debug_val_dataset():
    val = "/Users/manukastratta/Developer/CS329D/test-time-training-project/data/camelyon17_v1.0/train/metadata_val_split.csv"
    df = pd.read_csv(val)
    df = df.sample(n=20)
    #df = df.reset_index()
    df.to_csv("/Users/manukastratta/Developer/CS329D/test-time-training-project/data/camelyon17_v1.0/debug/train/metadata_debug_val.csv", index=False)

import numpy as np
from PIL import Image

def normalize_one_image():
    #img_path = "/Users/manukastratta/Developer/CS329D/test-time-training-project/data/camelyon17_v1.0/patches/patient_089_node_3/patch_patient_089_node_3_x_33600_y_11456.png"
    #img_path = "/Users/manukastratta/Developer/CS329D/test-time-training-project/data/camelyon17_v1.0/patches/patient_087_node_0/patch_patient_087_node_0_x_37920_y_15008.png"
    img_path = "data/CelebA/img_align_celeba/024592.jpg"
    image = Image.open(img_path)
    image.show()
    image_array = np.array(image)[:, :, :3]
    print(image_array.shape)

    # channel_mean = np.mean(image_array, axis=(0, 1))
    # channel_std = np.std(image_array, axis=(0, 1))
    
    #channel_mean = [183.72132071, 142.7651698, 182.28779395]
    #channel_std = [34.25966575, 39.11443915, 28.19893461]
    channel_mean = [203.40160808, 168.95786517, 138.58313576]
    channel_std = [41.5537357, 61.90715449, 73.58213545]


    print("channel_mean: ", channel_mean)
    print("channel_std: ", channel_std)

    # output[channel] = (input[channel] - mean[channel]) / std[channel]
    image_array[:, :, 0] = (image_array[:, :, 0] - channel_mean[0]) / channel_std[0]
    image_array[:, :, 1] = (image_array[:, :, 1] - channel_mean[1]) / channel_std[1]
    image_array[:, :, 2] = (image_array[:, :, 2] - channel_mean[2]) / channel_std[2]

    image = Image.fromarray(image_array)
    image.show()

def calculate_stats_1_img():
    image = Image.open("data/CelebA/img_align_celeba/" + "024592.jpg")
    image_array = np.array(image)
    image_array = image_array[:, :, :3]

    total_mean = np.zeros(3)
    total_std = np.zeros(3)

    # Compute mean and standard deviation for each channel
    channel_mean = np.mean(image_array, axis=(0, 1))
    channel_std = np.std(image_array, axis=(0, 1))

    # Accumulate mean and std values
    total_mean += channel_mean
    total_std += channel_std
    print("mean: ", total_mean)
    print("std: ", total_std)


def calculate_channel_mean_std():
    metadata_df = pd.read_csv('data/CelebA/splits/manuka_setting1_train.csv')
    image_paths = metadata_df['img_filename'].to_numpy()

    total_mean = np.zeros(3)
    total_std = np.zeros(3)
    num_images = len(image_paths)

    for i, image_path in enumerate(image_paths):
        if i % 1000 == 0:
            print(i)
        image = Image.open("data/CelebA/img_align_celeba/" + image_path)
        image_array = np.array(image)
        image_array = image_array[:, :, :3]

        # Compute mean and standard deviation for each channel
        channel_mean = np.mean(image_array, axis=(0, 1))
        channel_std = np.std(image_array, axis=(0, 1))

        # Accumulate mean and std values
        total_mean += channel_mean
        total_std += channel_std

    # Calculate the average mean and std across all images
    mean = total_mean / num_images
    std = total_std / num_images

    print("mean: ", mean)
    print("std: ", std)

if __name__ == "__main__":
    #data_cleanup()
    #data_stats()
    fire.Fire()