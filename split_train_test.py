import argparse
import os
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd

import normalization
from recommender import *

def main(player_games_csv: str, split: float = 0.8, split_steamids: int = 1000, pre_normalize_class: str = None):
    # first check if the file exists
    if not os.path.exists(player_games_csv):
        raise FileNotFoundError(f"File '{player_games_csv}' does not exist")
    # now check if the columns are correct, just like in normalization.py
    data = pd.read_csv(player_games_csv)
    if not all([col in data.columns for col in ["steamid", "appid", "playtime_forever"]]):
        raise ValueError("The DataFrame must have the columns 'steamid', 'appid' and 'playtime_forever'")
    if not all([data[col].dtype == "int64" for col in ["steamid", "appid"]]):
        raise ValueError("The DataFrame must have the columns 'steamid' and 'appid' as integer values.")
    
    # normalize the data if pre_normalize is True
    if pre_normalize_class and pre_normalize_class != "None":
        # get all classes from the normalization module
        classes = [cls for cls in normalization.__dict__.values() if isinstance(cls, type) and issubclass(cls, normalization.AbstractPlaytimeNormalizer)]
        # get the class with the given name
        
        pre_normalize_class_types = [cls for cls in classes if cls.__name__ == pre_normalize_class]
        if len(pre_normalize_class_types) == 0:
            raise ValueError(f"Normalization class '{pre_normalize_class_types}' does not exist")
        normalizer = pre_normalize_class_types[0]
        print("Normalizing the data with pre_normalizer:", pre_normalize_class)
        # now normalize the data
        pre_normalizer = normalizer()
        data = pre_normalizer.normalize(data)
        print("Normalized the data")
    else:
        print("Skipping normalization")
    # now split the data
    # first get all steamids
    steamids = data["steamid"].unique()
    # now shuffle them
    random.shuffle(steamids)
    # now split them
    print(f"Splitting the data into {split:.2f} for training and {1 - split:.2f} for testing")
    train_test_steamids = steamids[:split_steamids]
    # now get the data for the train and test set
    train_test_data = data[data["steamid"].isin(train_test_steamids)]
    # now split the data
    train_data = train_test_data.sample(frac=split)
    test_data = train_test_data.drop(train_data.index)
    # merge the train_data with the original data except the train_test_steamids
    print("Merging the train data with the remaining data and sorting the data...")
    train_data = pd.concat([train_data, data[~data["steamid"].isin(train_test_steamids)]])
    # sort the data so it's easier to read
    train_data = train_data.sort_values(by=["steamid", "appid"])
    test_data = test_data.sort_values(by=["steamid", "appid"])
    # now save the data
    os.makedirs("data", exist_ok=True)
    train_data.to_csv("data/player_games_train.csv", index=False)
    test_data.to_csv("data/player_games_test.csv", index=False)
    # also save the steamids
    np.save("data/train_test_steamids.npy", train_test_steamids)
    print("Done. Saved the data to 'player_games_train.csv' and 'player_games_test.csv', and the steamids to be tested to 'train_test_steamids.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splits the player_games.csv into a training and a test set")
    parser.add_argument("player_games_csv", type=str, help="The path to the player_games.csv file")
    parser.add_argument("--split", type=float, default=0.8, help="The percentage of the data to use for training")
    parser.add_argument("--split_steamids", type=int, default=1000, help="The number of steamids to use for splitting the data")
    parser.add_argument("--pre_normalize", type=str, default="None", help="The normalization class to use before splitting the data")
    args = parser.parse_args()
    main(args.player_games_csv, args.split, args.split_steamids, args.pre_normalize)


