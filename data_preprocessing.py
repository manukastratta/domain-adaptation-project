import pandas as pd
import numpy as np
#from matplotlib import image

def main():
    metadata_df = pd.read_csv('data/camelyon17_v1.0/train/metadata_train_split.csv')
    img_filenames = metadata_df['image_path'].to_numpy()

    sum_means = np.zeros(3)
    sum_stds = np.zeros(3)
    for filename in img_filenames[:10]: 
        im = image.imread('/home/mvalentinastratta/test-time-training-project/data/camelyon17_v1.0/'+filename)
        im = im[:,:,:3]
        im_mean = np.mean(im, axis=0)
        im_mean = np.mean(im_mean, axis=0)
        im_std = np.std(im, axis=0)
        im_std = np.std(im_std, axis=0)
        sum_means += im_mean
        sum_stds += im_std

    mean_to_report = sum_means / img_filenames.shape[0]
    std_to_report = sum_stds / img_filenames.shape[0]
    preprocessing_info = {'mean':mean_to_report, 'std':std_to_report}
    preprocessing_info_df = pd.DataFrame.from_dict(preprocessing_info)
    preprocessing_info_df.to_csv('preprocessing_info.csv')

if __name__ == "__main__":
    main()
