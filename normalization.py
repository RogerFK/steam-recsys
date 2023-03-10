from abc import ABCMeta, abstractmethod
import pandas as pd
from pandas.core.api import DataFrame
import numpy as np

class PlaytimeNormalizerBase(metaclass=ABCMeta):
    def __init__(self, inplace: bool = False):
        self.inplace = inplace
    @abstractmethod
    def _normalize_function(self, playtime, max_playtime):
        return 60 if playtime < 60 else playtime

    def normalize(self, data: DataFrame) -> DataFrame:
        """The data to normalize, with a structure of:
            "steamid","appid","playtime_forever"
        
        Arguments:
        ---
            data (DataFrame): Expects a DataFrame with the above structure
        Returns:
        ---
            DataFrame: A DataFrame with the same structure as the input, but with the playtime_forever column normalized
        """
        self.validate_data(data)
        if not self.inplace:
            data = data.copy()
        # Normalize each player with its own max playtime (for example, if a player has 50.000h in HoI4, and 10h in Civ5, the HoI4 playtime will be normalized to 1, and the Civ5 playtime to 0.2)
        # This is done to prevent players with a lot of playtime in a single game from dominating the recommendations
        # Also, any playtime below 60 minutes counts as 60 minutes, playtime below 1 hour means the user has shown interest in the game, but isn't very important
    
        # data["playtime_forever"] = data["playtime_forever"].apply(lambda x: 60 if x < 60 else x)
        max_playtimes = data.groupby("steamid")["playtime_forever"].max()
        
        data["playtime_forever"] = data.apply(lambda x: self._normalize_function(x["playtime_forever"], max_playtimes[x["steamid"]]), axis=1)
        return data
    
    def validate_data(self, data: DataFrame):
        # check if it's a DataFrame
        if not isinstance(data, DataFrame):
            raise TypeError("Argument 'data' must be a DataFrame")
        
        # check if the data has the correct columns
        if not all([col in data.columns for col in ["steamid", "appid", "playtime_forever"]]):
            raise ValueError("The DataFrame must have the columns 'steamid', 'appid' and 'playtime_forever'")
        
        # check if the data has the correct types
        if not all([data[col].dtype == "int64" for col in ["steamid", "appid"]]):
            raise ValueError("The DataFrame must have the columns 'steamid' and 'appid' as integer values.")
    

class LinearPlaytimeNormalizer(PlaytimeNormalizerBase):
    """Normalizes the player_games data using the playtime_forever
    """
    def _normalize_function(self, playtime, max_playtime):
        playtime = super()._normalize_function(playtime, max_playtime)
        return playtime / max_playtime

class LogPlaytimeNormalizer(PlaytimeNormalizerBase):
    """Normalizes the player_games data using the log of the playtime_forever
    """
    def _normalize_function(self, playtime, max_playtime):
        playtime = super()._normalize_function(playtime, max_playtime)
        return np.log(playtime) / np.log(max_playtime)

class RootPlaytimeNormalizer(PlaytimeNormalizerBase):
    """Normalizes the player_games data using the N root of the playtime_forever
    """
    def __init__(self, nroot: int = 2, inplace: bool = False):
        super().__init__(inplace=inplace)
        self.nroot = nroot

    def _normalize_function(self, playtime, max_playtime):
        playtime = super()._normalize_function(playtime, max_playtime)
        return np.power(playtime, 1/self.nroot) / np.power(max_playtime, 1/self.nroot)
