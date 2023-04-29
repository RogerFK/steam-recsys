from abc import ABC, abstractmethod
from pandas.core.api import DataFrame
import numpy as np
import math
import re
import config
class AbstractPlaytimeNormalizer(ABC):
    def __init__(self, denominator_function: str = "sum_max", playtime_approach: str = "minutes_always_more_than_60", output_multiplier: float = config.RATING_MULTIPLIER, inplace: bool = False):
        self.inplace = inplace
        self.denominator_function = denominator_function
        self.playtime_approach = playtime_approach
        self.output_multiplier = output_multiplier

    @abstractmethod
    def normalize_value(self, playtime, denominator):
        pass
    
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

        # we need this to prevent division by zero, wrong sum, max, etc.
        if self.playtime_approach == "ignore":
            pass
        elif self.playtime_approach == "minutes_always_more_than_60":
            data["playtime_forever"] = data["playtime_forever"].apply(lambda x: 60 if x < 60 else x)
        elif self.playtime_approach == "hours_but_add_1":
            data["playtime_forever"] = data["playtime_forever"].apply(lambda x: math.floor(x / 60) + 1)
        else:
            raise ValueError("playtime_approach must be 'minutes_always_more_than_60', 'hours_but_add_1' or 'ignore'")

        # NOTE: I benchmarked this and having no if statement is a tiny bit faster, but there's no noticeable difference
        # also, normalization is usually not the bottleneck, so it doesn't matter
        if self.denominator_function == "max":
            denominators = data.groupby("steamid")["playtime_forever"].max()
        elif self.denominator_function == "sum":
            denominators = data.groupby("steamid")["playtime_forever"].sum()
        elif self.denominator_function == "sum_max":
            denominators = data.groupby("steamid")["playtime_forever"].sum()
            max_denominators = data.groupby("steamid")["playtime_forever"].max()
            # now multiply every sum by the max denominator
            for steamid, denominator in max_denominators.items():
                denominators[steamid] *= (denominator / denominators[steamid])
        else:
            raise ValueError("divide_by must be either 'max', 'sum'")
        
        data["playtime_forever"] = data.apply(lambda x: self.output_multiplier * self.normalize_value(x["playtime_forever"], denominators[x["steamid"]]), axis=1)
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

    def __repr__(self) -> str:
        name = self.__class__.__name__
        name = name[0:name.index("PlaytimeNormalizer")] + "PN"
        return name + f"_{self.denominator_function}" + "".join([tok[0] for tok in re.findall(r"[a-zA-Z]+|[0-9]", self.playtime_approach)]) + f"_{self.output_multiplier}x"
    
    def __str__(self) -> str:
        return self.__repr__()

class NoNormalization(AbstractPlaytimeNormalizer):
    """Doesn't normalize the player_games data
    """
    def normalize(self, data: DataFrame) -> DataFrame:
        self.validate_data(data)
        return data
    def normalize_value(self, playtime, denominator):
        return playtime
    def __repr__(self) -> str:
        return "NoNormalization"

class LinearPlaytimeNormalizer(AbstractPlaytimeNormalizer):
    """Normalizes the player_games data using the playtime_forever
    """
    def normalize_value(self, playtime, denominator):
        return playtime / denominator

class LogPlaytimeNormalizer(AbstractPlaytimeNormalizer):
    """Normalizes the player_games data using the log of the playtime_forever
    """
    def normalize_value(self, playtime, denominator):
        return np.log(playtime) / np.log(denominator)

class RootPlaytimeNormalizer(AbstractPlaytimeNormalizer):
    """Normalizes the player_games data using the N root of the playtime_forever
    """
    def __init__(self, denominator_function: str = "sum_max", playtime_approach: str = "minutes_always_more_than_60", nroot: int = 2, output_multiplier: int = 5, inplace: bool = False):
        super().__init__(denominator_function, playtime_approach, output_multiplier, inplace)
        self.nroot = nroot

    def normalize_value(self, playtime, denominator):
        return np.power(playtime, 1/self.nroot) / np.power(denominator, 1/self.nroot)
    
    def __repr__(self) -> str:
        return super().__repr__() + f"_{self.nroot}root"

# class CubicRootPlaytimeNormalizer(RootPlaytimeNormalizer):
#     """Used for experiments.py. Normalizes the player_games data using the cubic root of the playtime_forever
#     """
#     def __init__(self, denominator_function: str = "sum_max", playtime_approach: str = "minutes_always_more_than_60", output_multiplier: int = 5, inplace: bool = False):
#         super().__init__(denominator_function, playtime_approach, 3, output_multiplier, inplace)