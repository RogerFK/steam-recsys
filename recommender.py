from abc import ABCMeta, abstractmethod
from pandas.core.api import DataFrame
import pandas as pd
from normalization import PlaytimeNormalizerBase
class RecommenderSystemBase(ABCMeta):
    def __init__(self, playtime_normalizer: PlaytimeNormalizerBase):
        self.playtime_normalizer = playtime_normalizer

    @abstractmethod
    def recommend(self, data: DataFrame, steamid: int, n: int = 10) -> DataFrame:
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
        if steamid < 0:
            raise ValueError("steamid must be positive")

    def validate_n(self, n: int):
        if not isinstance(n, int):
            raise TypeError("n must be an int")
        if n < 1:
            raise ValueError("n must be positive")

class TagBasedRecommenderSystem(RecommenderSystemBase):
    def __init__(self, playtime_normalizer: PlaytimeNormalizerBase, tags_file: str):
        super().__init__(playtime_normalizer)
        self.tags = pd.from_csv(tags_file)
        # precompute the "nearest neighbors"
        # this is a dictionary of dictionaries
        # the first key is the appid
        # the second key is the tagid
        # the value is the priority
        self.neighbors = {}
        for index, row in self.tags.iterrows():
            if not row["appid"] in self.neighbors:
                self.neighbors[row["appid"]] = {}
            self.neighbors[row["appid"]][row["tagid"]] = row["priority"]
        
    def get_tags(self, appid: int) -> dict:
        if not appid in self.neighbors:
            return {}
        return self.neighbors[appid]

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

        
    