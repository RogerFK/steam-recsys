from abc import ABC, abstractmethod
from pandas.core.api import DataFrame
import pandas as pd
from normalization import PlaytimeNormalizerBase
from datasketch import MinHash, MinHashLSH
from queue import PriorityQueue
class RecommenderSystemBase(ABC):
    def __init__(self, playtime_normalizer: PlaytimeNormalizerBase):
        self.playtime_normalizer = playtime_normalizer

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
        self.validate_data(data)
        self.validate_steamid(steamid)
        self.validate_n(n)

    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "steamid" in data.columns:
            raise ValueError("data must have a 'steamid' column")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "playtime_forever" in data.columns:
            raise ValueError("data must have a 'playtime_forever' column")

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
        super().__init__(None)
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

        # convert the priority queue to a DataFrame
        # the DataFrame is ordered by score, from higher to lower
        recommendations = []
        while not priority_queue.empty():
            score, appid = priority_queue.get()
            recommendations.append({"appid": appid, "score": score})
        recommendations = pd.DataFrame(recommendations).sort_values("score", ascending=False)
        return recommendations

class TagBasedRecommenderSystem(RecommenderSystemBase):
    def __init__(self, playtime_normalizer: PlaytimeNormalizerBase, tags_file: str):
        super().__init__(playtime_normalizer)
        self.tags = pd.read_csv(tags_file).groupby("appid")

    def recommend(self, data: DataFrame, steamid: int, n: int = 10) -> DataFrame:
        super().recommend(data, steamid, n)
        # Since this one is TagBased, we're going to use the 'game_tags.csv' file
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

        
    