import argparse

import numpy as np
import pandas as pd

def make_small_dataset(csv_file, n, name):
    df = pd.read_csv(csv_file)
    small_df = df.iloc[:n, :]
    small_df = small_df[['img_filename','Male','Black_Hair']]
    small_df.to_csv(name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_for_smaller_csv", type=str)
    parser.add_argument("--n", type=int)
    parser.add_argument("--make_small_dataset", action="store_true")
    parser.add_argument("--name", type=str)
    args = parser.parse_args()
    if args.make_small_dataset:
        make_small_dataset(args.data_for_smaller_csv, args.n, args.name)
