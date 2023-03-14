from abc import ABC, abstractmethod
import logging
from pandas.core.api import DataFrame
import pandas as pd
from normalization import PlaytimeNormalizerBase
from datasketch import MinHash, MinHashLSH, MinHashLSHEnsemble
from queue import PriorityQueue
import pickle
import os
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
class RecommenderDataBase(ABC):
    def __init__(self, csv_filename: str, pickle_filename: str):
        self.processed_data = None
        self.data = None
        try:
            with open(f"bin_data/{pickle_filename}.pickle", "rb") as f:
                print("Loading data...")
                self.processed_data = pickle.load(f)
                self.data = self.processed_data["data"]
                print(f"Loaded data from {f.name}")
                return
        except FileNotFoundError:
            try:
                logging.info(f"Loading data from {csv_filename}...")
                self.data = pd.read_csv(csv_filename)
            except FileNotFoundError:
                print("File not found, please check the path and try again")
    
    def __getitem__(self, key):
        return self.data[key]
    
    @abstractmethod
    def rating(self, item, *args, **kwargs):
        """
        Gets the rating/score, item type and arguments are dependant on subclasses.

        Args:
        ---
        item (Any): Dependant on the 'RecommenderData' type. An exception should be thrown if the parameter is invalid.

        Returns:
        ---
        float: The rating for a certain item
        """
        pass
    @abstractmethod
    def similarity(self, item1, item2) -> float:
        pass

class PlayerGamesPlaytimeData(RecommenderDataBase):
    pickle_name_fmt = "PGPTData/{}_thres{}_per{}par{}"
    def __init__(self, filename: str, playtime_normalizer: PlaytimeNormalizerBase, threshold=0.4, num_perm=128, num_part=32):
        if not os.path.exists("bin_data/PGPTData"):
            os.makedirs("bin_data/PGPTData")
        pickle_name = self.pickle_name_fmt.format(str(playtime_normalizer), threshold, num_perm, num_part)
        super().__init__(filename, pickle_name)  # load the processed data if it exists
        if self.processed_data is not None:
            self.lshensemble = self.processed_data["lshensemble"]
            self.data = self.processed_data["data"]
            return
        self.validate_data(self.data)
        logging.info("Processing player games...")
        self.playtime_normalizer = playtime_normalizer
        logging.info("Normalizing data...")
        self.data = self.playtime_normalizer.normalize(self.data)
        logging.info("Data normalized.")
        # compute the MinHashLSHEnsemble index
        # http://ekzhu.com/datasketch/lshensemble.html#minhash-lsh-ensemble
        self.lshensemble = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm, num_part=num_part)
        lsh_ensemble_index = []
        set_dict = {}
        logging.info("Computing MinHashes for each user...")
        for steamid, row in self.data.groupby("steamid"):
            # we only want to store the appids, we'll get the playtimes later
            user_games = row["appid"].values
            min_hash = MinHash(num_perm=num_perm, hashfunc=self.trivial_hash)
            min_hash.update_batch(user_games)
            lsh_ensemble_index.append((steamid, min_hash, len(user_games)))
            set_dict[steamid] = set(user_games)
        logging.info("MinHashes computed. Computing MinHashLSHEnsemble index...")
        self.lshensemble.index(lsh_ensemble_index)
        logging.info("MinHashLSHEnsemble index computed.")
        self.processed_data = {
            "lshensemble": self.lshensemble,
            "data": self.data,
            "minhashes": {steamid: (min_hash, size) for steamid, min_hash, size in lsh_ensemble_index},
        }
        with open(f"bin_data/{pickle_name}.pickle", "wb") as f:
            logging.info("Dumping LSH Ensemble and data...")
            pickle.dump(self.processed_data, f)
            logging.info(f"Dumped LSH Ensemble and data to {f.name}")
    
    def trivial_hash(self, x):
        return x

    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "steamid" in data.columns:
            raise ValueError("data must have a 'steamid' column")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "playtime_forever" in data.columns:
            raise ValueError("data must have a 'playtime_forever' column")

    def rating(self, appid: int, steamid: int) -> float:
        vals = self.data.loc[(self.data["appid"] == appid) & (self.data["steamid"] == steamid), "playtime_forever"].values
        if len(vals) == 0:
            return 0
        return vals[0]
    
    def similarity(self, steamid1: int, steamid2: int) -> float:
        """Computes the similarity between two users

        Args:
        ---
        steamid1 (int): The steamid of the first user
        steamid2 (int): The steamid of the second user

        Returns:
        ---
        float: The similarity between the two users
        """
        min_hash1, size1 = self.processed_data["minhashes"][steamid1]
        min_hash2, size2 = self.processed_data["minhashes"][steamid2]
        return min_hash1.jaccard(min_hash2)
    
    def get_similar_users(self, steamid: int):
        """Gets similar users to a user

        Args:
        ---
        steamid (int): The steamid of the user

        Returns:
        ---
        list: A list of tuples (steamid, similarity)
        """
        min_hash, size = self.processed_data["minhashes"][steamid]
        return self.lshensemble.query(min_hash, size)
        

        
class RecommenderSystemBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def recommend(self, data: DataFrame, steamid: int, n: int = 10) -> DataFrame:
        """Recommends games to a user

        Args:
        ---
         * data (DataFrame): The data to use for the recommendation
         * steamid (int): The steamid of the user
         * n (int, optional): The number of recommendations. Defaults to 10.

        Returns:
        ---
        DataFrame: appid -> score, ordered by score, up to n items
        """
        # self.validate_data(data)
        self.validate_steamid(steamid)
        self.validate_n(n)
    
    def recommendations_from_priority_queue(self, priority_queue: PriorityQueue) -> DataFrame:
        """Converts a priority queue to a DataFrame, ordered by score, up to n items

        Args:
        ---
         * priority_queue (PriorityQueue): The priority queue to convert

        Returns:
        ---
        DataFrame: appid -> score, ordered by score
        """
        recommendations = []
        while not priority_queue.empty():
            score, appid = priority_queue.get()
            recommendations.append({"appid": appid, "score": score})
        # we reverse the list to get the right order
        recommendations = pd.DataFrame(reversed(recommendations))
        return recommendations

    def validate_steamid(self, steamid: int):
        if not isinstance(steamid, int):
            raise TypeError("steamid must be an int")
        if steamid < 76500000000000000:
            raise ValueError("invalid steamid")

    def validate_n(self, n: int):
        if not isinstance(n, int):
            raise TypeError("n must be an int")
        if n < 1:
            raise ValueError("n must be positive")

class RandomRecommenderSystem(RecommenderSystemBase):
    def __init__(self):
        super().__init__()
    def recommend(self, data: DataFrame, steamid: int, n: int = 10) -> DataFrame:
        import random
        # it's ordered by (priority, appid), from lower to upper
        # when the priority queue is full, it will pop the lowest priority
        priority_queue = PriorityQueue(n + 1)

        # pick random games from the dataset and give them a random score
        # the score is a random number between 0 and 5
        random_data = data.sample(n=n, replace=True)
        for _, row in random_data.iterrows():
            priority_queue.put((random.random() * 5, row["appid"]))
            if priority_queue.qsize() > n:
                _ = priority_queue.get()

        return self.recommendations_from_priority_queue(priority_queue)

class TagBasedRecommenderSystem(RecommenderSystemBase):
    def __init__(self, tags_file: str):
        super().__init__()
        self.tags = pd.read_csv(tags_file).groupby("appid")

    def recommend(self, data: DataFrame, steamid: int, n: int = 10) -> DataFrame:
        super().recommend(data, steamid, n)
        # Since this one is TagBased, we're going to use the 'game_tags.csv' file
        # The structure for this file is:
        # "appid","tagid","priority"
        self.validate_data(data)
    
    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column. Make sure you're using the 'game_tags.csv' file data")
        if not "tagid" in data.columns:
            raise ValueError("data must have a 'tagid' column. Make sure you're using the 'game_tags.csv' file data")
        if not "priority" in data.columns:
            raise ValueError("data must have a 'priority' column.  Make sure you're using the 'game_tags.csv' file data")

class PlaytimeBasedRecommenderSystem(RecommenderSystemBase):
    def __init__(self, playtime_normalizer: PlaytimeNormalizerBase):
        super().__init__()
        self.playtime_normalizer = playtime_normalizer
    
    def recommend(self, data: DataFrame, steamid: int, n: int = 10) -> DataFrame:
        super().recommend(data, steamid, n)
        self.validate_data(data)
        # Since this one is PlaytimeBased, we're going to use the 'game_tags.csv' file
        # The structure for this file is:
        # "appid","tagid","priority"
        # Priority is a number between 0 and 1, where 1 is the most important tag
        # We're going to use the priority as a weight for the tags
        # We're going to use the playtime_forever as a weight for the games
        # We're going to use the tags as a weight for the recommendations

        # First, we need to normalize the playtime_forever column
        data = self.playtime_normalizer.normalize(data)
        
        # Then, we need to get the tags for the games the user has played
        # We're going to use the 'appid' column to get the tags
        # We're going to use the 'playtime_forever' column to get the weight for the tags
        # We're going to use the 'priority' column to get the weight for the tags
    
    
    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "steamid" in data.columns:
            raise ValueError("data must have a 'steamid' column")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "playtime_forever" in data.columns:
            raise ValueError("data must have a 'playtime_forever' column")