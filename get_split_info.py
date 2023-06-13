import numpy as np
import pandas as pd
import fire

def camelyon_get_stats_wrapper(split):
    if split == "train":
        df = pd.read_csv("data/camelyon17_v1.0/wilds_splits/metadata_train.csv")
    elif split == "val":
        df = pd.read_csv("data/camelyon17_v1.0/wilds_splits/metadata_val.csv")
    else:
        df = pd.read_csv("data/camelyon17_v1.0/wilds_splits/metadata_test.csv")
    
    tumor = df[df["tumor"] == 1]
    no_tumor = df[df["tumor"] == 0]
    print("Proportion of tumor: ", len(tumor) / len(df))



def celebA_get_stats_wrapper(train_or_val_or_test, setting):
    print("SETTING :", setting)
    if (train_or_val_or_test == 'train'):
        # read in train dataset
        complete = pd.read_csv("data/CelebA/splits/setting"+str(setting)+"_train.csv", index_col=0)
    elif(train_or_val_or_test == 'val'):
        # read in val dataset
        complete = pd.read_csv("data/CelebA/splits/setting"+str(setting)+"_val.csv", index_col=0)
    elif(train_or_val_or_test == 'test'):
        complete = pd.read_csv("data/CelebA/splits/setting"+str(setting)+"_test.csv", index_col=0)

    black_hair = complete[complete['Black_Hair']==1]
    nonblack_hair = complete[complete['Black_Hair']==0]
    male = complete[complete['Male']==1]
    female = complete[complete['Male']==0]
    # create the four quadrant datasets
    black_hair_female = black_hair[black_hair['Male']==0]
    black_hair_male = black_hair[black_hair['Male']==1]
    nonblack_hair_female = nonblack_hair[nonblack_hair['Male']==0]
    nonblack_hair_male = nonblack_hair[nonblack_hair['Male']==1]
    assert len(black_hair) + len(nonblack_hair) == len(complete)
    assert len(black_hair_female) + len(nonblack_hair_female) == len(complete[complete['Male']==0])
    assert len(black_hair_male) + len(nonblack_hair_male) == len(complete[complete['Male']==1])
    assert len(black_hair_female) + len(black_hair_male) == len(black_hair)
    assert len(nonblack_hair_female) + len(nonblack_hair_male) == len(nonblack_hair)
    assert len(black_hair_female) + len(nonblack_hair_female) + len(black_hair_male) + len(nonblack_hair_male) == len(complete)
    print('Sample sizes in '+train_or_val_or_test+' split:')
    print('black_hair_male_n', len(black_hair_male))
    print('nonblack_hair_male_n', len(nonblack_hair_male))
    print('black_hair_female_n', len(black_hair_female))
    print('nonblack_hair_female_n', len(nonblack_hair_female))
    print()
    print('Proportion of males:', len(male)/(len(male)+len(female)))
    print('Proportion of males with black hair:', len(black_hair_male)/(len(black_hair_male) + len(nonblack_hair_male)))
    print('Proportion of females with black hair:', len(black_hair_female)/(len(black_hair_female) + len(nonblack_hair_female)))
    print('Proportion with black hair:', (len(black_hair_female) + len(black_hair_male))/len(complete))
    print()

def celebA_get_stats():
    celebA_get_stats_wrapper('train', setting = 2)
    celebA_get_stats_wrapper('val', setting = 2)
    celebA_get_stats_wrapper('test', setting = 2)

def camelyon_get_stats():
    camelyon_get_stats_wrapper('train')
    camelyon_get_stats_wrapper('val')
    camelyon_get_stats_wrapper('test')

if __name__ == "__main__":
    fire.Fire()
