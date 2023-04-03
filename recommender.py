from abc import ABC, abstractmethod
from collections import OrderedDict
import logging
from typing import Any, Dict, Generator, List, Set, Tuple, Union
from pandas.core.api import DataFrame
import pandas as pd
from normalization import AbstractPlaytimeNormalizer
from datasketch import MinHash, MinHashLSHEnsemble
from queue import PriorityQueue
import pickle
import os
import math
import numpy as np
import atexit
import config
import difflib
from datetime import datetime

def trivial_hash(x):
    return x

class AbstractRecommenderData(ABC):
    def __init__(self, csv_filename: str, pickle_filename: str = None):
        self.processed_data = None
        self.data: pd.DataFrame = pd.DataFrame()
        try:
            if pickle_filename is None:
                raise FileNotFoundError
            with open(f"bin_data/{pickle_filename}.pickle", "rb") as f:
                logging.info(f"Loading pickled data for {self.__class__.__name__}...")
                self.processed_data = pickle.load(f)
                logging.info(f"Loaded pickled data from {f.name}")
                return
        except FileNotFoundError:
            try:
                logging.info(f"Loading {self.__class__.__name__} data from {csv_filename}...")
                self.data = pd.read_csv(csv_filename)
            except FileNotFoundError:
                logging.error("File not found, please check the path and try again")
                raise
    
    def __getitem__(self, key):
        return self.data[key]
    
    @abstractmethod
    def rating(self, *args, **kwargs):
        """
        Arguments are dependant on subclasses.
        -----
        Description:
        ---
        Gets the rating/score
        
        Args:
        ---
        Dependant on subclasses.

        Returns:
        ---
        float: The rating for a certain item
        """
        pass

    def get_minhash_similar_games(self, user_minhash, length):
        """
        Description:
        ---
        Gets the most similar games to the user's minhash using MinHashLSHEnsemble.

        Args:
        ---
        * user_minhash (datasketch.MinHash): The MinHash of the user's games
        * length (int): The number of items in the minhash

        Returns:
        ---
        * List[Tuple[int, float]]: A list of tuples of the form (appid, similarity)
        """
        results = self.lshensemble.query(user_minhash, length)
        return results

class PlayerGamesPlaytime(AbstractRecommenderData):
    pickle_name_fmt = "PGPTData/{}_thres{}_per{}par{}"
    def __init__(self, filename: str, playtime_normalizer: AbstractPlaytimeNormalizer, threshold=0.8, num_perm=128, num_part=32):
        if not os.path.exists("bin_data/PGPTData"):
            os.makedirs("bin_data/PGPTData")
        self.pickle_name = self.pickle_name_fmt.format(str(playtime_normalizer), threshold, num_perm, num_part)
        super().__init__(filename, self.pickle_name)  # load the processed data if it exists
        self.playtime_normalizer = playtime_normalizer
        self.minhash_num_perm = num_perm
        self.dirty = self.processed_data is None
        if not self.dirty:
            self.lshensemble = self.processed_data["lshensemble"]
            self.data = self.processed_data["data"]
            self.minhashes = self.processed_data["minhashes"]
            return
        self.validate_data(self.data)
        logging.info("Processing player games...")
        logging.info("Normalizing data...")
        self.data = self.playtime_normalizer.normalize(self.data)
        logging.info("Data normalized.")
        # compute the MinHashLSHEnsemble index
        # http://ekzhu.com/datasketch/lshensemble.html#minhash-lsh-ensemble
        self.lshensemble = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm, num_part=num_part)
        lsh_ensemble_index = []
        self.minhashes = {}
        logging.info("Computing MinHashes for each user...")
        for steamid, row in self.data.groupby("steamid"):
            # we only want to store the appids, we'll get the playtimes later
            user_games = row["appid"].values
            min_hash = MinHash(num_perm=num_perm, hashfunc=trivial_hash)
            min_hash.update_batch(user_games)
            user_games_length = len(user_games)
            lsh_ensemble_index.append((steamid, min_hash, user_games_length))
            self.minhashes[steamid] = (min_hash, user_games_length)
        logging.info("MinHashes computed. Computing MinHashLSHEnsemble index...")
        self.lshensemble.index(lsh_ensemble_index)
        logging.info("MinHashLSHEnsemble index computed.")
        self.processed_data = {
            "lshensemble": self.lshensemble,
            "data": self.data,
            "minhashes": self.minhashes,
        }
        self.dump_data()
        atexit.register(self.dump_data)

    def dump_data(self):
        if self.dirty:
            logging.info(f"Dumping data to bin_data/{self.pickle_name}.pickle")
            with open(f"bin_data/{self.pickle_name}.pickle", "wb") as f:
                pickle.dump(self.processed_data, f)
            self.dirty = False

    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "steamid" in data.columns:
            raise ValueError("data must have a 'steamid' column")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "playtime_forever" in data.columns:
            raise ValueError("data must have a 'playtime_forever' column")

    def rating(self, steamid: int, appid: int) -> float:
        """
        Description:
        ---
        The rating from a user for a game. 
        NOTE: You can change this inside a recommender to get different ratings.

        Args:
        ---
        * appid (int): the appid of the game
        * steamid (int): the steamid of the user

        Returns:
        ---
        * float: Their rating for the game
        """
        vals = self.data.loc[(self.data["appid"] == appid) & (self.data["steamid"] == steamid), "playtime_forever"].values
        if len(vals) == 0:
            return 0.0
        return vals[0] if not math.isnan(vals[0]) else 0.0
    
    def get_user_games(self, steamid: int) -> DataFrame:
        """Gets the games a user has played

        Args:
        ---
        steamid (int): The steamid of the user

        Returns:
        ---
        DataFrame: The games the user has played
        """
        user_games = self.data.loc[self.data["steamid"] == steamid]
        user_games.reset_index(drop=True, inplace=True)
        return user_games
    
    def get_lsh_similar_users(self, steamid: int) -> Generator[int, None, None]:
        """Gets all similar users beyond the threshold. Specially useful for collaborative filtering.

        Args:
        ---
        steamid (int): The steamid of the user

        Returns:
        ---
        Generator[int, None, None]: A list of steamids
        """
        min_hash, size = self.minhashes[steamid]
        
        logging.debug(f"Querying LSH Ensemble for similar users to {steamid}...")
        return self.lshensemble.query(min_hash, size)

    def get_all_user_games(self) -> Generator[DataFrame, None, None]:
        """Gets all the games for all users

        Returns:
        ---
        Generator[DataFrame, None, None]: A generator of DataFrames containing all the games for each user
        """
        for steamid, user_games in self.data.groupby("steamid"):
            user_games.reset_index(drop=True, inplace=True)
            yield steamid, user_games
    
    def add_user_games(self, steamid: int, user_games: DataFrame) -> None:
        """Adds a user's games to the data

        Args:
        ---
        steamid (int): The steamid of the user
        user_games (DataFrame): The games the user has played
        """
        if not self.playtime_normalizer.inplace:  # just in case...
            user_games = user_games.copy()
        user_games = self.playtime_normalizer.normalize(user_games)
        user_games["steamid"] = steamid
        self.data = self.data.append(user_games, ignore_index=True)
        self.data.reset_index(drop=True, inplace=True)
        self.processed_data = None
        self.dirty = True

class GameTags(AbstractRecommenderData):
    MAX_LENGTH = 19

    def __init__(self, csv_filename: str = "game_tags.csv", weight_threshold=0.75, threshold=0.8, num_perm=128, num_part=32) -> None:
        """
        Description:
        ---
        Loads the game tags data from a file.

        Args:
        ---
        filename (str): The filename of the game tags data. Defaults to "game_tags.csv".
        """
        super().__init__(csv_filename, pickle_filename="game_tags")
        self.minhash_num_perm = num_perm
        if self.processed_data is not None:
            self.lshensemble = self.processed_data["lshensemble"]
            self.data = self.processed_data["data"]
            self.minhashes = self.processed_data["minhashes"]
            self.relevant_by_tag = self.processed_data["relevant_by_tag"]
            return
        self.validate_data(self.data)
        self.lshensemble = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm, num_part=num_part)
        lsh_ensemble_index = []
        self.minhashes = {}
        self.relevant_by_tag = {}  # tagid -> appids above weight threshold
        logging.info("Computing MinHashes for each game...")
        for appid, row in self.data.groupby("appid"):
            # we only want to store the appids, we'll get the weights later
            game_tags = row["tagid"].values
            min_hash = MinHash(num_perm=num_perm, hashfunc=trivial_hash)
            min_hash.update_batch(game_tags)
            game_tags_length = len(game_tags)
            lsh_ensemble_index.append((appid, min_hash, game_tags_length))
            self.minhashes[appid] = (min_hash, game_tags_length)
            for tagid in row.loc[row["weight"] >= weight_threshold, "tagid"].values:
                if tagid not in self.relevant_by_tag:
                    self.relevant_by_tag[tagid] = set()
                self.relevant_by_tag[tagid].add(appid)
        logging.info("MinHashes computed. Computing MinHashLSHEnsemble index...")
        self.lshensemble.index(lsh_ensemble_index)
        logging.info("MinHashLSHEnsemble index computed.")
        self.processed_data = {
            "lshensemble": self.lshensemble,
            "data": self.data,
            "minhashes": self.minhashes,
            "relevant_by_tag": self.relevant_by_tag
        }
        with open(f"bin_data/game_tags.pickle", "wb") as f:
            logging.info("Dumping LSH Ensemble and data...")
            pickle.dump(self.processed_data, f)
            logging.info(f"Dumped LSH Ensemble and data to {f.name}")
    
    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "tagid" in data.columns:
            raise ValueError("data must have a 'tagid' column")
        if not "weight" in data.columns:
            raise ValueError("data must have a 'weight' column")
    
    def rating(self, appid, tagid) -> float:
        """
        Description:
        ---
        The rating from a user for a game. 
        NOTE: You can change this inside a recommender to get different ratings.
        With game tags it's specially useful to use TF-IDF in conjunction with this rating.

        Args:
        ---
        * appid (int): the appid of the game
        * tagid (int): the tagid of the tag

        Returns:
        ---
        * float: Their rating for the game
        """
        vals = self.data.loc[(self.data["appid"] == appid) & (self.data["tagid"] == tagid), "weight"].values
        if len(vals) == 0:
            return 0
        return vals[0]
    
    def get_lsh_similar_games(self, appid: int) -> Generator[int, None, None]:
        """Gets all similar games beyond the threshold set in the constructor.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        Generator[int, None, None]: A list of appids
        """
        if appid not in self.minhashes:
            logging.debug(f"Game {appid} not in minhashes, probably dead / hidden game.")
            return None
        min_hash, size = self.minhashes[appid]

        logging.debug(f"Querying LSH Ensemble for similar games to {appid}...")
        return self.lshensemble.query(min_hash, size)
    
    def get_minhash_similar_games(self, minhash: MinHash, size: int) -> Generator[int, None, None]:
        """Gets all similar games beyond the threshold set in the constructor.
        This method is mainly used to get similar games to a user's profile.

        Args:
        ---
        minhash (MinHash): The minhash of the game

        Returns:
        ---
        Generator[int, None, None]: A list of appids
        """
        return self.lshensemble.query(minhash, size)
    
    def get_tags(self, appid: int) -> Dict[int, float]:
        """Gets the tags a game has.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        Dict[int, float]: The tags the game has
        """
        game_tags = self.data.loc[self.data["appid"] == appid]
        game_tags.reset_index(drop=True, inplace=True)
        return game_tags.set_index("tagid")["weight"].to_dict()
    
    def get_weighted_tags_for_list(self, appids: List[int]) -> Dict[int, Dict[int, float]]:
        """Gets the tags for a list of games.

        Args:
        ---
        appids (List[int]): The appids of the games

        Returns:
        ---
        Dict[int, Dict[int, float]]: The tags the games have
        """
        return {appid: self.get_tags(appid) for appid in appids}
    
    def get_relevant_games(self, tagid: int) -> Set[int]:
        """Gets the games that have a tag above the weight threshold set in the constructor.

        Args:
        ---
        tagid (int): The tagid of the tag

        Returns:
        ---
        Set[int]: The appids of the games
        """
        if tagid not in self.relevant_by_tag:
            return set()
        return self.relevant_by_tag[tagid]
    
    def get_all_game_tags(self) -> DataFrame:
        """Gets the DataFrame including all game tags and their weights.

        Returns:
        ---
        DataFrame: the DataFrame including all game tags and their weights
        """
        return self.data
    
    def get_tag_name(self, tagid: int) -> str:
        """Gets the name of a tag.

        Args:
        ---
        tagid (int): The tagid of the tag

        Returns:
        ---
        str: The name of the tag
        """
        # TODO
        return tagid
        # return self.tag_names[tagid]

# this class below is the same as GameTagsData but uses game_genres.csv instead
class GameGenres(AbstractRecommenderData):  # NOTE: Game genres are fairly limited, so this is not very useful. Hence, we won't waste much time on it.
    def __init__(self, csv_filename: str, threshold=0.8, num_perm=128, num_part=32):
        super().__init__(csv_filename, "game_genres")
        self.minhash_num_perm = num_perm
        if self.processed_data is not None:
            self.lshensemble = self.processed_data["lshensemble"]
            self.data = self.processed_data["data"]
            self.minhashes = self.processed_data["minhashes"]
            return
        self.validate_data(self.data)
        self.lshensemble = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm, num_part=num_part)
        lsh_ensemble_index = []
        self.minhashes = {}
        logging.info("Computing MinHashes for each game...")
        for appid, row in self.data.groupby("appid"):
            game_genres = row["genreid"].values
            min_hash = MinHash(num_perm=128, hashfunc=trivial_hash)
            min_hash.update_batch(game_genres)
            game_genres_length = len(game_genres)
            lsh_ensemble_index.append((appid, min_hash, game_genres_length))
            self.minhashes[appid] = (min_hash, game_genres_length)
        logging.info("MinHashes computed. Computing MinHashLSHEnsemble index...")
        self.lshensemble.index(lsh_ensemble_index)
        logging.info("MinHashLSHEnsemble index computed.")
        self.processed_data = {
            "lshensemble": self.lshensemble,
            "data": self.data,
            "minhashes": self.minhashes,
        }
        with open(f"bin_data/game_genres.pickle", "wb") as f:
            logging.info("Dumping LSH Ensemble and data...")
            pickle.dump(self.processed_data, f)
            logging.info(f"Dumped LSH Ensemble and data to {f.name}")
    
    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "genreid" in data.columns:
            raise ValueError("data must have a 'genreid' column")
    
    def rating(self, appid, genreid) -> float:
        """
        Description:
        ---
        If a game has a genre, it is rated 1.0, otherwise 0.0.

        Args:
        ---
        * appid (int): the appid of the game
        * genreid (int): the genreid of the genre

        Returns:
        ---
        * float: Their rating for the game
        """
        vals = self.data.loc[(self.data["appid"] == appid) & (self.data["genreid"] == genreid)].values
        return 0.0 if len(vals) == 0 else 1.0
    
    def get_lsh_similar_games(self, appid: int) -> Generator[int, None, None]:
        """Gets all similar games beyond the threshold set in the constructor.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        Generator[int, None, None]: A list of appids
        """
        min_hash, size = self.minhashes[appid]

        logging.debug(f"Querying LSH Ensemble for similar games to {appid}...")
        return self.lshensemble.query(min_hash, size)
    
    def get_genres(self, appid: int) -> Set[int]:
        """Gets the genres a game has.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        Set[int]: The genres the game has
        """
        game_genres = self.data.loc[self.data["appid"] == appid]
        game_genres.reset_index(drop=True, inplace=True)
        return set(game_genres["genreid"].values)

# NOTE: like game genres, game categories aren't very useful for recommending, but might be useful for filtering.
# For example, if a user only wants multiplayer games 'Multiplayer' is also a tag and has a weight unlike the category
# If we wanted to play a multiplayer-oriented game like Call of Duty, Apex Legends, etc. we likely 
# don't want to play a game like Black Mesa, which is mainly a singleplayer game but has a multiplayer mode
# Thus, we won't waste much time on this either.
class GameCategories(AbstractRecommenderData):
    def __init__(self, csv_filename: str, threshold=0.8, num_perm=128, num_part=32):
        super().__init__(csv_filename, "game_categories")
        self.minhash_num_perm = num_perm
        if self.processed_data is not None:
            self.lshensemble = self.processed_data["lshensemble"]
            self.data = self.processed_data["data"]
            self.minhashes = self.processed_data["minhashes"]
            return
        self.validate_data(self.data)
        self.lshensemble = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm, num_part=num_part)
        lsh_ensemble_index = []
        self.minhashes = {}
        logging.info("Computing MinHashes for each game...")
        for appid, row in self.data.groupby("appid"):
            game_categories = row["categoryid"].values
            min_hash = MinHash(num_perm=128, hashfunc=trivial_hash)
            min_hash.update_batch(game_categories)
            game_categories_length = len(game_categories)
            lsh_ensemble_index.append((appid, min_hash, game_categories_length))
            self.minhashes[appid] = (min_hash, game_categories_length)
        logging.info("MinHashes computed. Computing MinHashLSHEnsemble index...")
        self.lshensemble.index(lsh_ensemble_index)
        logging.info("MinHashLSHEnsemble index computed.")
        self.processed_data = {
            "lshensemble": self.lshensemble,
            "data": self.data,
            "minhashes": self.minhashes,
        }
        with open(f"bin_data/game_categories.pickle", "wb") as f:
            logging.info("Dumping LSH Ensemble and data...")
            pickle.dump(self.processed_data, f)
            logging.info(f"Dumped LSH Ensemble and data to {f.name}")
    
    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "categoryid" in data.columns:
            raise ValueError("data must have a 'categoryid' column")
    
    def rating(self, appid, categoryid) -> float:
        """
        Description:
        ---
        If a game has a category, it is rated 1.0, otherwise 0.0.

        Args:
        ---
        * appid (int): the appid of the game
        * categoryid (int): the categoryid of the category

        Returns:
        ---
        * float: 1.0 if the game has the category, 0.0 otherwise
        """
        vals = self.data.loc[(self.data["appid"] == appid) & (self.data["categoryid"] == categoryid)].values
        return 0.0 if len(vals) == 0 else 1.0
    
    def get_lsh_similar_games(self, appid: int) -> Generator[int, None, None]:
        """Gets all similar games beyond the threshold set in the constructor.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        Generator[int, None, None]: A list of appids
        """
        min_hash, size = self.minhashes[appid]

        logging.debug(f"Querying LSH Ensemble for similar games to {appid}...")
        return self.lshensemble.query(min_hash, size)
    
    def get_categories(self, appid: int) -> Set[int]:
        """Gets the categories a game has.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        Set[int]: The categories the game has
        """
        game_categories = self.data.loc[self.data["appid"] == appid]
        game_categories.reset_index(drop=True, inplace=True)
        return set(game_categories["categoryid"].values)

class GameDevelopers(AbstractRecommenderData):
    def __init__(self, gd_csv_filename: str):
        pickle_filename = "game_developers"
        super().__init__(gd_csv_filename, pickle_filename)
        if self.processed_data is not None:
            self.data = self.processed_data
            return
        self.processed_data = self.data
        with open(f"bin_data/{pickle_filename}.pickle", "wb") as f:
            logging.info("Dumping LSH Ensemble and data...")
            pickle.dump(self.processed_data, f)
            logging.info(f"Dumped LSH Ensemble and data to {f.name}")
    
    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "developerid" in data.columns:
            raise ValueError("data must have a 'developerid' column")
    
    def get_developers(self, appid: int) -> Set[int]:
        """Gets the developers a game has.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        Set[int]: The developers the game has
        """
        game_developers = self.data.loc[self.data["appid"] == appid]
        game_developers.reset_index(drop=True, inplace=True)
        return set(game_developers["developerid"].values)
    
    def rating(self, appid, developer) -> float:
        """
        Description:
        ---
        If a game has a developer, it is rated 1.0, otherwise 0.0.

        Args:
        ---
        * appid (int): the appid of the game
        * developer (int): the developerid of the developer

        Returns:
        ---
        * float: Their "rating" for the game
        """
        vals = self.data.loc[(self.data["appid"] == appid) & (self.data["developerid"] == developer)].values
        return 0.0 if len(vals) == 0 else 1.0

    def get_games_from_developer(self, developerid: int) -> Set[int]:
        """Gets the games a developer has made.

        Args:
        ---
        developerid (int): The developerid of the developer

        Returns:
        ---
        Set[int]: The games the developer has made
        """
        dev_games = self.data.loc[self.data["developerid"] == developerid]
        dev_games.reset_index(drop=True, inplace=True)
        return set(dev_games["appid"].values)
    
    def get_games_from_developers_of_game(self, appid) -> Dict[int, Set[int]]:
        """Gets the games the developer/s of a game has made.

        Args:
        ---
        developerid (int): The developerid of the developer

        Returns:
        ---
        Set[int]: The games the developer has made
        """
        devids = self.get_developers(appid)
        games = dict()
        for devid in devids:
            games[devid] = self.get_games_from_developer(devid)
        return games
    
class GamePublishers(AbstractRecommenderData):
    def __init__(self, csv_filename: str, pickle_filename: str = None):
        super().__init__(csv_filename, pickle_filename)
        if self.processed_data is not None:
            self.data = self.processed_data
            return
        self.processed_data = self.data
        with open(f"bin_data/{pickle_filename}.pickle", "wb") as f:
            logging.info("Dumping LSH Ensemble and data...")
            pickle.dump(self.processed_data, f)
            logging.info(f"Dumped LSH Ensemble and data to {f.name}")
    
    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "publisherid" in data.columns:
            raise ValueError("data must have a 'publisherid' column")
        
    def get_publishers(self, appid: int) -> Set[int]:
        """Gets the publishers a game has.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        Set[int]: The publishers the game has
        """
        game_publishers = self.data.loc[self.data["appid"] == appid]
        game_publishers.reset_index(drop=True, inplace=True)
        return set(game_publishers["publisherid"].values)
    
    def rating(self, appid, publisher) -> float:
        """
        Description:
        ---
        If a game has a publisher, it is rated 1.0, otherwise 0.0.

        Args:
        ---
        * appid (int): the appid of the game
        * publisher (int): the publisherid of the publisher

        Returns:
        ---
        * float: Their "rating" for the game
        """
        vals = self.data.loc[(self.data["appid"] == appid) & (self.data["publisherid"] == publisher)].values
        return 0.0 if len(vals) == 0 else 1.0
    
    def get_games_from_publisher(self, publisherid: int) -> Set[int]:
        """Gets the games a publisher has published.

        Args:
        ---
        publisherid (int): The publisherid of the publisher

        Returns:
        ---
        Set[int]: The games the publisher has published
        """
        pub_games = self.data.loc[self.data["publisherid"] == publisherid]
        pub_games.reset_index(drop=True, inplace=True)
        return set(pub_games["appid"].values)
    
    def get_games_from_publishers_of_game(self, appid) -> Dict[int, Set[int]]:
        """Gets the games the publisher/s of a game has published.

        Args:
        ---
        publisherid (int): The publisherid of the publisher

        Returns:
        ---
        Set[int]: The games the publisher has published
        """
        pubids = self.get_publishers(appid)
        games = dict()
        for pubid in pubids:
            games[pubid] = self.get_games_from_publisher(pubid)
        return games


class GameDetails(AbstractRecommenderData):
    class GameWithOnlyDetails:
        def __init__(self, game_details_row, rating_multiplier: float = config.RATING_MULTIPLIER):
            self.appid = game_details_row["appid"]
            self.name = game_details_row["name"]
            self.required_age = game_details_row["required_age"]
            self.is_free = game_details_row["is_free"]
            self.controller_support = game_details_row["controller_support"]
            self.has_demo = game_details_row["has_demo"]
            self.price_usd = game_details_row["price_usd"] / 100.00
            self.mac_os = game_details_row["mac_os"]
            self.positive_reviews = game_details_row["positive_reviews"]
            self.negative_reviews = game_details_row["negative_reviews"]
            self.total_reviews = game_details_row["total_reviews"]
            self.has_achievements = game_details_row["has_achievements"]
            self.release_date = game_details_row["release_date"]
            self.coming_soon = game_details_row["coming_soon"]
            self.released = not self.coming_soon
            self.rating = self.positive_reviews / self.total_reviews * rating_multiplier
    def __init__(self, csv_filename: str, rating_multiplier: float = config.RATING_MULTIPLIER):
        super().__init__(csv_filename)
        logging.info("Processing game details and setting index on appid...")
        self.data = self.data.set_index("appid")
        logging.info("Finished processing game details")
        self.rating_multiplier = rating_multiplier

    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        columns = ["appid", "name", "required_age", "is_free", "controller_support", "has_demo", "price_usd", "mac_os", "positive_reviews", "negative_reviews", "total_reviews", "has_achievements", "release_date", "coming_soon"]
        err_str = "\r\nMake sure the CSV has the following columns:\r\n\t[" + ", ".join(columns) + "]"
        if not all([col in data.columns for col in columns]):
            # check which ones are missing
            missing = []
            for col in columns:
                if col not in data.columns:
                    missing.append(col)
            raise ValueError("Missing columns: " + ", ".join(missing) + err_str)
    
    def get_game_row(self, appid: int): return self.data.loc[appid]

    def get_game(self, appid: int) -> GameWithOnlyDetails:
        """
        Description:
        ---
        Gets a Game object for a game.

        Args:
        ---
        * appid (int): the appid of the game

        Returns:
        ---
        * Game: The Game object for the game
        """
        return self.GameWithOnlyDetails(self.data.loc[appid], self.rating_multiplier)
    
    def rating(self, appid: int) -> float:
        """
        Description:
        ---
        The rating of a game is the number of positive reviews divided by the total number of reviews.

        Args:
        ---
        * appid (int): the appid of the game

        Returns:
        ---
        * float: The "rating" for the game
        """
        vals = self.data.loc[appid]
        pos = vals["positive_reviews"]
        tot = vals["total_reviews"]
        return (pos / tot) * self.rating_multiplier if tot != 0 else 0.0
    
    def get_games(self, appids: List[int]) -> List[GameWithOnlyDetails]:
        """
        Description:
        ---
        Gets a list of Game objects for a list of appids.

        Args:
        ---
        * appids (List[int]): the appids of the games

        Returns:
        ---
        * List[Game]: The list of Game objects for the games
        """
        return [self.GameWithOnlyDetails(self.data.loc[appid], self.rating_multiplier) for appid in appids]
    
    def get_all_games(self) -> List[GameWithOnlyDetails]:
        """
        Description:
        ---
        Gets a list of all Game objects.

        Returns:
        ---
        * List[Game]: The list of Game objects for all games
        """
        return [self.GameWithOnlyDetails(row, self.rating_multiplier) for _, row in self.data.iterrows()]

class Game(GameDetails.GameWithOnlyDetails):
    def __init__(self, game_details_row, game_categories: Set[int], game_developers: Set[int], game_publishers: Set[int], game_genres: Set[int], game_tags: Dict[int, float], rating_multiplier: float = config.RATING_MULTIPLIER):
        super().__init__(game_details_row)
        self.released = not self.coming_soon
        self.score = (self.positive_reviews / self.total_reviews) * rating_multiplier if self.total_reviews != 0 else 0.0
        self.categories = game_categories
        self.developers = game_developers
        self.publishers = game_publishers
        self.tags = game_tags
        self.genres = game_genres

    def __str__(self):
        return f"({self.appid}: {self.name}, score: {self.score})"

class GameInfo():
    # this class just encapsulates everything else to generate Game objects
    def __init__(self, game_details: GameDetails, game_categories: GameCategories, game_developers: GameDevelopers, game_publishers: GamePublishers, game_genres: GameGenres, game_tags: GameTags):
        self.game_details = game_details
        self.game_categories = game_categories
        self.game_developers = game_developers
        self.game_publishers = game_publishers
        self.game_genres = game_genres
        self.game_tags = game_tags
    
    def _get_game_from_row(self, row) -> Game:
        """
        Description:
        ---
        Gets a Game object from a row of the game details CSV.

        Args:
        ---
        * row (pandas.Series): a row from the game details CSV

        Returns:
        ---
        * Game: The Game object for the game
        """
        appid = row["appid"]
        game_categories = self.game_categories.get_categories(appid)
        game_developers = self.game_developers.get_developers(appid)
        game_publishers = self.game_publishers.get_publishers(appid)
        game_genres = self.game_genres.get_genres(appid)
        game_tags = self.game_tags.get_tags(appid)
        return Game(row, game_categories, game_developers, game_publishers, game_genres, game_tags)
    
    def get_game(self, appid) -> Game:
        """
        Description:
        ---
        Gets a Game object for a game.

        Args:
        ---
        * appid (int): the appid of the game

        Returns:
        ---
        * Game: The Game object for the game
        """
        game_details_row = self.game_details.get_game_row(appid)
        return self._get_game_from_row(game_details_row)
    
    def get_games(self, appids: List[int]) -> List[Game]:
        """
        Description:
        ---
        Gets a list of Game objects for a list of appids.

        Args:
        ---
        * appids (List[int]): the appids of the games

        Returns:
        ---
        * List[Game]: The Game objects for the games
        """
        return [self.get_game(appid) for appid in appids]

    def get_all_games(self) -> List[Game]:
        """
        Description:
        ---
        Gets a list of all Game objects.

        Returns:
        ---
        * List[Game]: The Game objects for all games
        """
        return [self._get_game_from_row(row) for _, row in self.game_details.data.iterrows()]

## Similarity ##
class AbstractSimilarity(ABC):
    @abstractmethod
    def similarity(self, item1: int, item2: int) -> float:
        """
        Description:
        ---
        Computes the similarity between two items.

        Args:
        ---
        item1 (int): The first item
        item2 (int): The second item

        Returns:
        ---
        float: The similarity between the two items
        """
        pass
class AbstractGameSimilarity(AbstractSimilarity):  # NOTE: This class is only used to check if a class is of type "app similarity"
    def __init__(self, recommender_data: AbstractRecommenderData) -> None:
        self.user_item_weights = {}
        self.recommender_data = recommender_data
        self.item_weight_max_length = 0  # used to cull

    @abstractmethod
    def similarity(self, appid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the similarity between two games.

        Args:
        ---
        appid (int): The appid of the game
        other (int): The appid of the game to compare to

        Returns:
        ---
        float: The similarity between the two games, from 0 to 1
        """
        pass
    @abstractmethod
    def get_similar_games(self, appid: int, n: int) -> List[Tuple[int, float]]:
        """
        Description:
        ---
        Gets the n most similar games to the given game.

        Args:
        ---
        appid (int): The appid of the game
        n (int): The number of similar games to get

        Returns:
        ---
        List[Tuple[int, float]]: A list of tuples containing the appid of the game and the similarity score
        """
        pass

    def get_similar_games_from_user_games(self, user_games: DataFrame) -> Tuple[List[int], Dict[int, float]]:
        """
        Description:
        ---
        Gets the games similar to the given user, only taking into account the tags.

        Args:
        ---
        user_games (DataFrame): The games played by the user

        Returns:
        ---
        Tuple[List[int], Dict[int, float]]: A tuple containing a list of the appids of the games and a dictionary containing the weights of the tags
        """
        user_minhash = MinHash(self.recommender_data.minhash_num_perm, hashfunc=trivial_hash)
        steamid = user_games.iloc[0]["steamid"]
        user_weights = self.get_item_weights(user_games)
        
        user_minhash.update_batch(user_weights.keys())
        similar_games = self.recommender_data.get_minhash_similar_games(user_minhash, len(user_weights.keys()))
        return similar_games, user_weights

    @abstractmethod
    def get_game_items(self, appid: int) -> List[Tuple[int, float]]:
        """
        Description:
        ---
        Gets the 'items' (tags, categories, etc.) for a game, relevant to the class.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        List[int, float]: The items for the game with their weights (1 if not weighted)
        """
        pass

    def get_item_weights(self, user_games: DataFrame) -> Dict[int, float]:
        """Gets or generates the item weights for a user.

        Args:
            user_games (DataFrame): The games played by the user, with the playtime_forever column

        Returns:
            Dict[int, float]: The item weights for the user
        """
        steamid = user_games.iloc[0]["steamid"]
        if steamid in self.user_item_weights: return self.user_item_weights[steamid]
        
        item_weight = {}
        for steamid, row in user_games.iterrows():
            appid = row["appid"]
            pseudorating = row["playtime_forever"]
            for itemid, weight in self.get_game_items(appid):
                if not itemid in item_weight:
                    item_weight[itemid] = 0
                item_weight[itemid] += self.weight_function(weight, pseudorating)

        item_weight = sorted(item_weight.items(), key=lambda x: x[1], reverse=True)
        if self.item_weight_max_length > 1:
            item_weight = item_weight[:self.item_weight_max_length]
        
        item_weight = dict(item_weight)
        self.user_item_weights[steamid] = item_weight
        print(item_weight)
        print(f"{item_weight.__class__.__name__}")
        logging.debug(f"{self.__class__.__name__} weights for user {steamid}: " + "\r\n".join([f"{itemid}: {weight:.2f}" for itemid, weight in item_weight.items()]))
        return item_weight
    
    def weight_function(self, weight, pseudorating) -> float:
        return weight * pseudorating

class RawUserSimilarity(AbstractSimilarity):
    def __init__(self, pgdata: PlayerGamesPlaytime) -> None:
        super(RawUserSimilarity).__init__()
        self.pgdata = pgdata

    def similarity(self, steamid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the raw similarity between two users, only taking into account total normalized playtime.

        Args:
        ---
        steamid (int): The steamid of the user
        other (int): The steamid of the user to compare to

        Returns:
        ---
        float: The similarity between the two users
        """
        # check if pgdata is set
        if self.pgdata is None:
            raise ValueError("pgdata must be set to use this similarity function")

        own_games_played = self.pgdata.get_user_games(steamid)
        other_games_played = self.pgdata.get_user_games(other)
        
        # we only want to compare the games that both users have played
        own_games_played = own_games_played.loc[own_games_played["appid"].isin(other_games_played["appid"])]
        
        total_score = 0
        for idx, row in own_games_played.iterrows():
            _, appid, own_pseudorating = row
            if own_pseudorating > 0:
                total_score += self.pgdata.rating(other, appid) * own_pseudorating

        return total_score  # raw score, but don't penalize for having more games

    def player_similarities_from_priority_queue(self, priority_queue: PriorityQueue) -> DataFrame:
        """Gets the similarities from a priority queue

        Args:
        ---
        priority_queue (PriorityQueue): The priority queue to get the similarities from

        Returns:
        ---
        List[Tuple[int, float]]: The list of tuples (steamid, similarity)
        """
        similar_users = []
        while not priority_queue.qsize() == 0:
            similar_users.append(priority_queue.get())
        sim_users = pd.DataFrame(reversed(similar_users),
                                 columns=["similarity", "steamid"],
                                 dtype=object)
        # reorder the columns
        sim_users = sim_users[["steamid", "similarity"]]
        return sim_users

    def get_similar_users(self, steamid: int, n: int = 10) -> DataFrame:
        """Gets n top similar users to a user. Specially useful for collaborative filtering.

        Args:
        ---
        steamid (int): The steamid of the user
        n (int, optional): The number of similar users to return. Defaults to 10.

        Returns:
        ---
        list: A list of tuples (steamid, similarity)
        """
        rough_similar_users = self.pgdata.get_lsh_similar_users(steamid)
        
        logging.info(f"Finding {n} similar users to {steamid}. Please wait...")
        priority_queue = PriorityQueue(n + 1)
        for similar_user in rough_similar_users:
            if similar_user == steamid:
                continue
            similarity = self.similarity(steamid, similar_user)
            priority_queue.put((similarity, similar_user))
            if priority_queue.qsize() > n:
                _ = priority_queue.get()
        
        logging.info(f"Finished finding relevant similar users.")
        return self.player_similarities_from_priority_queue(priority_queue)

class CosineUserSimilarity(RawUserSimilarity):
    def __init__(self, pgdata: PlayerGamesPlaytime, precompute: bool = True) -> None:
        super().__init__(pgdata)
        self.precomputed = precompute
        self.dirty = True
        if precompute:
            # Check if there's pickled data in bin_data\cosine_user_norms.pickle
            self.load_norms()
            if self.dirty:
                # Precompute the norms of the users
                self.norms = {}
                for steamid, user_games in self.pgdata.get_all_user_games():
                    # it's just the sum of the squares of the ratings
                    self.norms[steamid] = np.sum(user_games["playtime_forever"] ** 2)
                self.save_norms_if_dirty()
        else:
            self.norms = {}
        atexit.register(self.save_norms_if_dirty)
    
    def load_norms(self):
        if os.path.exists("bin_data/cosine_user_norms.pickle"):
            with open("bin_data/cosine_user_norms.pickle", "rb") as f:
                self.norms = pickle.load(f)
                self.dirty = False
    
    def save_norms_if_dirty(self):
        if self.dirty:
            logging.info("Saving cosine user norms...")
            with open("bin_data/cosine_user_norms.pickle", "wb") as f:
                pickle.dump(self.norms, f)
                self.dirty = False
    
    def get_user_norm(self, steamid: int) -> float:
        if steamid not in self.norms:
            user_games = self.pgdata.get_user_games(steamid)
            self.norms[steamid] = np.sum(user_games["playtime_forever"] ** 2)
            self.dirty = True
        return self.norms[steamid]

    def similarity(self, steamid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the cosine similarity between two users, only taking into account total normalized playtime.

        Args:
        ---
        steamid (int): The steamid of the user
        other (int): The steamid of the user to compare to

        Returns:
        ---
        float: The similarity between the two users
        """
        if self.precomputed:
            selfnorm = self.get_user_norm(steamid)
            othernorm = self.get_user_norm(other)
            return super().similarity(steamid, other) / ((selfnorm * othernorm) ** 0.5)
        # check if pgdata is set
        if self.pgdata is None:
            raise ValueError("pgdata must be set to use this similarity function")

        own_games_played = self.pgdata.get_user_games(steamid)
        other_games_played = self.pgdata.get_user_games(other)
        
        games_to_iterate = own_games_played.join(other_games_played)
        total_score = 0
        own_total_score = 0 if not steamid in self.norms else self.norms[steamid]
        other_total_score = 0 if not other in self.norms else self.norms[other]
        
        for idx, row in games_to_iterate.iterrows():
            sid, appid, own_pseudorating = row
            if sid == steamid:
                other_pseudorating = self.pgdata.rating(other, appid)
            else:
                other_pseudorating = own_pseudorating
                own_pseudorating = self.pgdata.rating(steamid, appid)
            total_score += other_pseudorating * own_pseudorating
            if steamid not in self.norms:
                own_total_score += own_pseudorating ** 2
            if other not in self.norms:
                other_total_score += other_pseudorating ** 2
        
        if steamid not in self.norms:
            self.norms[steamid] = own_total_score
        if other not in self.norms:
            self.norms[other] = other_total_score
        return total_score / (math.sqrt(own_total_score) * math.sqrt(other_total_score))

class PearsonUserSimilarity(RawUserSimilarity):
    def __init__(self, pgdata: PlayerGamesPlaytime) -> None:
        super().__init__(pgdata)
        # Check if there's pickled data in bin_data\pearsonusersimilarity.pickle
        self.dirty = False
        if os.path.exists("bin_data/pearsonusersimilarity.pickle"):
            with open("bin_data/pearsonusersimilarity.pickle", "rb") as f:
                self.processed_data = pickle.load(f)
            self.user_mean = self.processed_data["user_mean"]
            self.denominator = self.processed_data["denominator"]
        else:
            self.user_mean = {}
            self.denominator = {}
            for steamid, user_games in self.pgdata.get_all_user_games():
                user_mean = np.sum(user_games['playtime_forever'])
                self.user_mean[steamid] = user_mean
                # to get the denominator, we need to subtract the mean from each rating
                self.denominator[steamid] = np.sum(user_games['playtime_forever'] - user_mean)
            self.processed_data = {
                "user_mean": self.user_mean,
                "denominator": self.denominator
            }
            with open("bin_data/pearsonusersimilarity.pickle", "wb") as f:
                pickle.dump(self.processed_data, f)
        atexit.register(self.save_processed_data_if_dirty)
    
    def save_processed_data_if_dirty(self):
        if self.dirty:
            with open("bin_data/pearsonusersimilarity.pickle", "wb") as f:
                pickle.dump(self.processed_data, f)

    def get_user_mean_denominator(self, steamid: int) -> Tuple[float, float]:
        if steamid in self.user_mean:
            return self.user_mean[steamid], self.denominator[steamid]
        else:
            ug = self.pgdata.get_user_games(steamid)
            user_mean = np.sum(ug['playtime_forever'])
            self.dirty = True
            self.user_mean[steamid] = user_mean / len(ug)
            self.denominator[steamid] = np.sum(ug['playtime_forever'] - user_mean) / len(ug)

    
    def similarity(self, steamid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the Pearson correlation coefficient between two users, only taking into account total normalized playtime.

        Args:
        ---
        steamid (int): The steamid of the user
        other (int): The steamid of the user to compare to

        Returns:
        ---
        float: The similarity between the two users
        """
        # check if pgdata is set
        if self.pgdata is None:
            raise ValueError("pgdata must be set to use this similarity function")

        own_games_played = self.pgdata.get_user_games(steamid)
        other_games_played = self.pgdata.get_user_games(other)
        
        games_to_iterate = own_games_played.join(other_games_played, on="appid", how="outer", lsuffix="_left", rsuffix="_right").drop(columns=["steamid_left", "steamid_right"])

        total_score = 0
        user_mean, user_denominator = self.get_user_mean_denominator(steamid)
        other_mean, other_denominator = self.get_user_mean_denominator(other)
        
        for idx, row in games_to_iterate.iterrows():
            _, appid_left, own_pseudorating, appid_right, other_pseudorating = row
            if np.isnan(own_pseudorating):
                own_pseudorating = 0
            if np.isnan(other_pseudorating):
                other_pseudorating = 0
            total_score += (own_pseudorating - user_mean) * (other_pseudorating - other_mean)
        denominator = (user_denominator * other_denominator) ** 0.5
        return total_score / denominator if denominator != 0 else 0

class RawGameTagSimilarity(AbstractGameSimilarity):
    def __init__(self, game_tags_or_game_info: Union[GameTags, GameInfo]) -> None:
        if not game_tags_or_game_info:
            raise ValueError("game_info or game_tags must be set to use this similarity function")
        self.game_tags = game_tags_or_game_info if isinstance(game_tags_or_game_info, GameTags) else game_tags_or_game_info.game_tags
        super().__init__(self.game_tags)
        self.item_weight_max_length = GameTags.MAX_LENGTH

    def similarity(self, appid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the similarity between two games, only taking into account the tags.

        Args:
        ---
        appid (int): The appid of the game
        other (int): The appid of the game to compare to

        Returns:
        ---
        float: The similarity between the two games
        """
        tags = self.game_tags.get_tags(appid)
        other_tags = self.game_tags.get_tags(other)
        # compute the similarity
        similarity = 0
        for tagid, weight in tags.items():
            if tagid in other_tags:
                similarity += weight * other_tags[tagid]
        return similarity
    
    def get_similar_games(self, appid: int, n: int = 10) -> DataFrame:
        """
        Description:
        ---
        Gets the n most similar games to the given game, only taking into account the tags.

        Args:
        ---
        appid (int): The appid of the game
        n (int): The number of games to return

        Returns:
        ---
        DataFrame: A list of tuples containing the appid and similarity of the n most similar games
        """
        games = self.game_tags.get_lsh_similar_games(appid)
        if games is None:
            return None
        logging.info(f"Finding similar games to {appid}. Please wait...")
        priority_queue = PriorityQueue(n + 1)
        for candidate_appid in games:
            similarity = self.similarity(appid, candidate_appid)
            priority_queue.put((similarity, candidate_appid))
            if priority_queue.qsize() > n:
                priority_queue.get()
        similar_games = []
        while not priority_queue.qsize() == 0:
            similarity, candidate_appid = priority_queue.get()
            similar_games.append((candidate_appid, similarity))
        logging.info(f"Finished finding {n} similar games to {appid}")
        return pd.DataFrame(reversed(similar_games), columns=["appid", "similarity"])

    def get_game_items(self, appid: int) -> List[Tuple[int, float]]:
        """
        Description:
        ---
        Gets the tags of a game.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        List[Tuple[int, float]]: A list of tuples containing the tagid and weight of the tags of the game
        """
        return list(self.game_tags.get_tags(appid).items())

class CosineGameTagSimilarity(RawGameTagSimilarity):
    def __init__(self, game_tags_or_game_info: Union[GameTags, GameInfo]) -> None:
        super().__init__(game_tags_or_game_info)
        # NOTE: We don't need to precompute game tag norms because at most a game has 19 tags
        # and thus the overhead of accessing the norm through a dictionary is higher than
        # just computing it on the fly

    def similarity(self, appid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the cosine similarity between two games, only taking into account the tags.

        Args:
        ---
        appid (int): The appid of the game
        other (int): The appid of the game to compare to

        Returns:
        ---
        float: The similarity between the two games
        """
        tags = self.game_tags.get_tags(appid)
        other_tags = self.game_tags.get_tags(other)
        
        # compute the similarity
        similarity = 0
        own_norm = 0
        other_norm = 0
        for tagid, weight in tags.items():
            own_norm += weight ** 2
            if tagid in other_tags:
                similarity += weight * other_tags[tagid]
        for tagid, weight in other_tags.items():
            other_norm += weight ** 2
        return similarity / (own_norm * other_norm) ** 0.5

class PearsonGameTagSimilarity(RawGameTagSimilarity):
    def __init__(self, game_tags_or_game_info: Union[GameTags, GameInfo]) -> None:
        super().__init__(game_tags_or_game_info)
        # NOTE: We don't need to precompute game tag norms and means because at most 
        # a game has 19 tags and thus the overhead of accessing the norm through
        # a dictionary is higher than just computing it on the fly

    def similarity(self, appid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the Pearson correlation coefficient between two games, only taking into account the tags.

        Args:
        ---
        appid (int): The appid of the game
        other (int): The appid of the game to compare to

        Returns:
        ---
        float: The similarity between the two games
        """
        tags = self.game_tags.get_tags(appid)
        other_tags = self.game_tags.get_tags(other)

        # compute the similarity
        similarity = 0
        own_norm = 0
        other_norm = 0
        own_mean = 0
        other_mean = 0
        for tagid, weight in tags.items():
            own_norm += weight ** 2
            own_mean += weight
        for tagid, weight in other_tags.items():
            other_norm += weight ** 2
            other_mean += weight
        own_mean /= len(tags)
        other_mean /= len(other_tags)
        for tagid, weight in tags:
            if tagid in other_tags:
                similarity += (weight - own_mean) * (other_tags[tagid] - other_mean)
        return similarity / (own_norm * other_norm) ** 0.5

class GameCategoriesSimilarity(AbstractGameSimilarity):
    def __init__(self, game_categories_or_game_info: Union[GameCategories, GameInfo] = None):
        self.game_categories = game_categories_or_game_info if isinstance(game_categories_or_game_info, GameCategories) else game_categories_or_game_info.game_categories
        super().__init__(self.game_categories)

    def similarity(self, appid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the Jaccard similarity between two games, only taking into account the categories.

        Args:
        ---
        appid (int): The appid of the game
        other (int): The appid of the game to compare to

        Returns:
        ---
        float: The similarity between the two games
        """
        categories = self.game_categories.get_categories(appid)
        other_categories = self.game_categories.get_categories(other)
    
        return len(categories.intersection(other_categories)) / len(categories.union(other_categories))
    
    def get_game_items(self, appid: int) -> List[Tuple[int, float]]:
        """
        Description:
        ---
        Gets the categories of a game.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        List[Tuple[int, float]]: A list of tuples containing the categoryid and weight of the categories of the game
        """
        return [(category, 1) for category in self.game_categories.get_categories(appid)]
    
    def get_similar_games(self, appid: int, n: int) -> List[Tuple[int, float]]:
        """
        Description:
        ---
        Gets the n most similar games to the given game.

        Args:
        ---
        appid (int): The appid of the game
        n (int): The number of games to return

        Returns:
        ---
        List[Tuple[int, float]]: A list of tuples containing the appid and similarity of the n most similar games
        """
        categories = self.game_categories.get_categories(appid)
        similarities = []
        for other in self.game_categories.get_lsh_similar_games(appid):
            other_categories = self.game_categories.get_categories(other)
            similarities.append((other, len(categories.intersection(other_categories)) / len(categories.union(other_categories))))
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

class GameGenresSimilarity(AbstractGameSimilarity):
    def __init__(self, game_genres_or_game_info: Union[GameGenres, GameInfo] = None):
        self.game_genres = game_genres_or_game_info if isinstance(game_genres_or_game_info, GameGenres) else game_genres_or_game_info.game_genres
        super().__init__(self.game_genres)

    def similarity(self, appid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the Jaccard similarity between two games, only taking into account the genres.

        Args:
        ---
        appid (int): The appid of the game
        other (int): The appid of the game to compare to

        Returns:
        ---
        float: The similarity between the two games
        """
        genres = self.game_genres.get_genres(appid)
        other_genres = self.game_genres.get_genres(other)
    
        return len(genres.intersection(other_genres)) / len(genres.union(other_genres))
    
    def get_game_items(self, appid: int) -> List[Tuple[int, float]]:
        """
        Description:
        ---
        Gets the genres of a game.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        List[Tuple[int, float]]: A list of tuples containing the genreid and weight of the genres of the game
        """
        return [(genre, 1) for genre in self.game_genres.get_genres(appid)]
    
    def get_similar_games(self, appid: int, n: int) -> List[Tuple[int, float]]:
        """
        Description:
        ---
        Gets the n most similar games to the given game, only taking into account the genres.

        Args:
        ---
        appid (int): The appid of the game
        n (int): The number of games to return

        Returns:
        ---
        List[Tuple[int, float]]: A list of tuples containing the appid and similarity of the n most similar games
        """
        genres = self.game_genres.get_genres(appid)
        if len(genres) == 0:
            return []
        similarities = []
        for other in self.game_genres.get_lsh_similar_games(appid):
            other_genres = self.game_genres.get_genres(other)
            if len(other_genres) == 0:
                continue
            similarity = len(genres.intersection(other_genres)) / len(genres.union(other_genres))
            similarities.append((other, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

class GameDevelopersSimilarity(AbstractGameSimilarity):
    def __init__(self, game_developers_or_game_info: Union[GameDevelopers, GameInfo]) -> None:
        if not isinstance(game_developers_or_game_info, (GameDevelopers, GameInfo)):
            raise ValueError("game_developers_or_game_info must be of type GameDevelopers or GameInfo")
        self.game_developers = game_developers_or_game_info if isinstance(game_developers_or_game_info, GameDevelopers) else game_developers_or_game_info.game_developers
        super().__init__(self.game_developers)

    def similarity(self, appid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the Jaccard similarity between two games, only taking into account the developers.

        Args:
        ---
        appid (int): The appid of the game
        other (int): The appid of the game to compare to

        Returns:
        ---
        float: The similarity between the two games
        """
        developers = self.game_developers.get_developers(appid)
        other_developers = self.game_developers.get_developers(other)
        
        return len(developers.intersection(other_developers)) / len(developers.union(other_developers))
    
    def get_game_items(self, appid: int) -> List[Tuple[int, float]]:
        """
        Description:
        ---
        Gets the developers of a game.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        List[Tuple[int, float]]: A list of tuples containing the developerid and weight of the developers of the game
        """
        return [(developer, 1) for developer in self.game_developers.get_developers(appid)]
    
    def get_similar_games(self, appid: int, n: int) -> List[Tuple[int, float]]:
        """
        Description:
        ---
        Gets the n most similar games to the given game, only taking into account the developers.

        Args:
        ---
        appid (int): The appid of the game
        n (int): The number of games to return

        Returns:
        ---
        List[Tuple[int, float]]: A list of tuples containing the appid and similarity of the n most similar games
        """
        developers = self.game_developers.get_developers(appid)
        if len(developers) == 0:
            return []
        similarities = []
        for other, other_developers in self.game_developers.get_games_from_developers_of_game(appid):
            similarity = len(developers.intersection(other_developers)) / len(developers.union(other_developers))
            similarities.append((other, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    def get_similar_games_from_user_games(self, user_games: DataFrame) -> Tuple[List[int], Dict[int, float]]:
        """Gets similar games from the user's games, only taking into account the developers.

        Args:
        ---
        user_games (DataFrame): The user's games

        Returns:
        ---
        Tuple[List[int], Dict[int, float]]: A list of appids and a dictionary of developerids and their weight
        """
        user_weights = self.get_item_weights(user_games)
        appid_scores = {}
        for developer, weight in user_weights.items():
            for appid in self.game_developers.get_games_from_developer(developer):
                if appid not in appid_scores:
                    appid_scores[appid] = 0
                appid_scores[appid] += weight
        appids = sorted(appid_scores, key=appid_scores.get, reverse=True)
        return appids, user_weights

class GamePublishersSimilarity(AbstractGameSimilarity):
    def __init__(self, game_publishers_or_game_info: Union[GamePublishers, GameInfo]) -> None:
        if not isinstance(game_publishers_or_game_info, (GamePublishers, GameInfo)):
            raise ValueError("game_publishers_or_game_info must be of type GamePublishers or GameInfo")
        self.game_publishers = game_publishers_or_game_info if isinstance(game_publishers_or_game_info, GamePublishers) else game_publishers_or_game_info.game_publishers
        super().__init__(self.game_publishers)

    def similarity(self, appid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the Jaccard similarity between two games, only taking into account the publishers.

        Args:
        ---
        appid (int): The appid of the game
        other (int): The appid of the game to compare to

        Returns:
        ---
        float: The similarity between the two games
        """
        publishers = self.game_publishers.get_publishers(appid)
        other_publishers = self.game_publishers.get_publishers(other)
        
        return len(publishers.intersection(other_publishers)) / len(publishers.union(other_publishers))
    
    def get_game_items(self, appid: int) -> List[Tuple[int, float]]:
        """
        Description:
        ---
        Gets the publishers of a game.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        List[Tuple[int, float]]: A list of tuples containing the publisherid and weight of the publishers of the game
        """
        return [(publisher, 1) for publisher in self.game_publishers.get_publishers(appid)]
    
    def get_similar_games(self, appid: int, n: int) -> List[Tuple[int, float]]:
        """
        Description:
        ---
        Gets the n most similar games to the given game, only taking into account the publishers.

        Args:
        ---
        appid (int): The appid of the game
        n (int): The number of games to return

        Returns:
        ---
        List[Tuple[int, float]]: A list of tuples containing the appid and similarity of the n most similar games
        """
        publishers = self.game_publishers.get_publishers(appid)
        if len(publishers) == 0:
            return []
        similarities = []
        for other, other_publishers in self.game_publishers.get_games_from_publishers_of_game(appid):  # TODO
            similarity = len(publishers.intersection(other_publishers)) / len(publishers.union(other_publishers))
            similarities.append((other, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]
    
    def get_similar_games_from_user_games(self, user_games: DataFrame) -> Tuple[List[int], Dict[int, float]]:
        """Gets similar games from the user's games, only taking into account the publishers.

        Args:
        ---
        user_games (DataFrame): The user's games

        Returns:
        ---
        Tuple[List[int], Dict[int, float]]: A list of appids and a dictionary of publisherids and their weight
        """
        user_weights = self.get_item_weights(user_games)
        appid_scores = {}
        for publisher, weight in user_weights.items():
            for appid in self.game_publishers.get_games_from_publisher(publisher):
                if appid not in appid_scores:
                    appid_scores[appid] = 0
                appid_scores[appid] += weight
        appids = sorted(appid_scores, key=appid_scores.get, reverse=True)
        return appids, user_weights
class GameDetailsSimilarity(AbstractGameSimilarity):
    def __init__(
            self,
            game_details_or_game_info: Union[GameDetails, GameInfo],
            weight_name: float = 1.0,
            weight_required_age: float = 1.0,
            weight_is_free: float = 1.0,
            weight_controller_support: float = 1.0,
            weight_has_demo: float = 1.0,
            weight_price_usd: float = 1.0,
            weight_mac_os: float = 1.0,
            weight_rating: float = 1.0,
            weight_rating_count: float = 1.0,  # rough metric for popularity
            weight_has_achievements: float = 1.0,
            weight_release_date: float = 1.0,  # "I only like fresh games" / "old games were better"
            weight_coming_soon: float = 1.0
        ) -> None:
        self.game_details = game_details_or_game_info if isinstance(game_details_or_game_info, GameDetails) else game_details_or_game_info.game_details
        self.weights = {
            "name": weight_name,
            "required_age": weight_required_age,
            "is_free": weight_is_free,
            "controller_support": weight_controller_support,
            "has_demo": weight_has_demo,
            "price_usd": weight_price_usd,
            "mac_os": weight_mac_os,
            "rating": weight_rating,
            "rating_count": weight_rating_count,
            "has_achievements": weight_has_achievements,
            "release_date": weight_release_date,
            "coming_soon": weight_coming_soon
        }
        self.rating_multiplier = self.game_details.rating_multiplier
        super().__init__(self.game_info)
    
    def set_weights(self,
                    weight_name: float = -1.0,
                    weight_required_age: float = -1.0,
                    weight_is_free: float = -1.0,
                    weight_controller_support: float = -1.0,
                    weight_has_demo: float = -1.0,
                    weight_price_usd: float = -1.0,
                    weight_mac_os: float = -1.0,
                    weight_rating: float = -1.0,
                    weight_rating_count: float = -1.0,  # rough metric for popularity
                    weight_has_achievements: float = -1.0,
                    weight_release_date: float = -1.0,  # "I only like fresh games" / "old games were better"
                    weight_coming_soon: float = -1.0) -> None:
        self.weights = {
            "name": weight_name if weight_name >= 0 else self.weights["name"],
            "required_age": weight_required_age if weight_required_age >= 0 else self.weights["required_age"],
            "is_free": weight_is_free if weight_is_free >= 0 else self.weights["is_free"],
            "controller_support": weight_controller_support if weight_controller_support >= 0 else self.weights["controller_support"],
            "has_demo": weight_has_demo if weight_has_demo >= 0 else self.weights["has_demo"],
            "price_usd": weight_price_usd if weight_price_usd >= 0 else self.weights["price_usd"],
            "mac_os": weight_mac_os if weight_mac_os >= 0 else self.weights["mac_os"],
            "rating": weight_rating if weight_rating >= 0 else self.weights["rating"],
            "rating_count": weight_rating_count if weight_rating_count >= 0 else self.weights["rating_count"],
            "has_achievements": weight_has_achievements if weight_has_achievements >= 0 else self.weights["has_achievements"],
            "release_date": weight_release_date if weight_release_date >= 0 else self.weights["release_date"],
            "coming_soon": weight_coming_soon if weight_coming_soon >= 0 else self.weights["coming_soon"]
        }

    def set_weights_dict(self, weights: Dict[str, float]) -> None:
        self.weights = {
            "name": weights["name"] if "name" in weights else self.weights["name"],
            "required_age": weights["required_age"] if "required_age" in weights else self.weights["required_age"],
            "is_free": weights["is_free"] if "is_free" in weights else self.weights["is_free"],
            "controller_support": weights["controller_support"] if "controller_support" in weights else self.weights["controller_support"],
            "has_demo": weights["has_demo"] if "has_demo" in weights else self.weights["has_demo"],
            "price_usd": weights["price_usd"] if "price_usd" in weights else self.weights["price_usd"],
            "mac_os": weights["mac_os"] if "mac_os" in weights else self.weights["mac_os"],
            "rating": weights["rating"] if "rating" in weights else self.weights["rating"],
            "rating_count": weights["rating_count"] if "rating_count" in weights else self.weights["rating_count"],
            "has_achievements": weights["has_achievements"] if "has_achievements" in weights else self.weights["has_achievements"],
            "release_date": weights["release_date"] if "release_date" in weights else self.weights["release_date"],
            "coming_soon": weights["coming_soon"] if "coming_soon" in weights else self.weights["coming_soon"]
        }

    def similarity(self, appid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the similarity between two games, only taking into account the details.

        Args:
        ---
        appid (int): The appid of the game
        other (int): The appid of the game to compare to

        Returns:
        ---
        float: The similarity between the two games, from 0 to 1
        """
    
        details = self.game_details.get_game(appid)
        other_details = self.game_details.get_game(other)
        similarity = 0.0
        similarity += difflib.SequenceMatcher(None, details.name, other_details.name).ratio() * self.weights["name"] if self.weights["name"] > 0 else 0.0
        similarity += (1.0 - (abs(details.required_age - other_details.required_age) / 18.0)) * self.weights["required_age"] if self.weights["required_age"] > 0 else 0.0
        similarity += self.weights["is_free"] if details.is_free == other_details.is_free else 0.0
        similarity += (1.0 - abs(details.controller_support - other_details.controller_support) / 2.0) * self.weights["controller_support"] if self.weights["controller_support"] > 0 else 0.0
        similarity += self.weights["has_demo"] if details.has_demo == other_details.has_demo else 0.0
        similarity += max(1 - abs(details.price_usd - other_details.price_usd) / 70.0, 0) * self.weights["price_usd"] if self.weights["price_usd"] > 0 else 0.0
        similarity += self.weights["mac_os"] if details.mac_os == other_details.mac_os else 0.0
        similarity += max(1 - abs(details.rating - other_details.rating) / self.rating_multiplier, 0) * self.weights["rating"] if self.weights["rating"] > 0 else 0.0
        similarity += max(1 - abs(details.total_reviews - other_details.total_reviews) / max(details.total_reviews, other_details.total_reviews), 0) * self.weights["rating_count"] if self.weights["rating_count"] > 0 else 0.0
        similarity += self.weights["has_achievements"] if details.has_achievements == other_details.has_achievements else 0.0
        if self.weights["release_date"] > 0:
            try:
                own_release_date = datetime.strptime(details.release_date, "%b %d, %Y")
                other_release_date = datetime.strptime(other_details.release_date, "%b %d, %Y")
                similarity += max(1.0 - abs((own_release_date - other_release_date).days) / 365.0, 0) * self.weights["release_date"]
            except:  # Some games don't have a release date, some others have "Coming Soon" or even emojis
                pass
        similarity += self.weights["coming_soon"] if details.coming_soon == other_details.coming_soon else 0.0

        return similarity / sum(self.weights.values())

class WeightedGameSimilaritiesSimilarity(AbstractGameSimilarity): # TODO unused, maybe remove
    def __init__(self, game_similarities: List[Tuple[AbstractGameSimilarity, float]]) -> None:
        """
        Args:
        ---
            game_similarities (List[Tuple[AbstractGameSimilarity, float]]): A list of tuples containing the game similarity class and its weight
        """
        self.game_similarities = game_similarities
        # check if they're all game similarities and precompute the sum of the weights
        self.weights_sum = 0.0
        for game_similarity, weight in self.game_similarities:
            if not isinstance(game_similarity, AbstractGameSimilarity):
                raise TypeError(game_similarity.__class__.__name__ + " is not an instance of" + AbstractGameSimilarity.__name__)
            self.weights_sum += weight
        super().__init__(None)
    
    def similarity(self, appid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the similarity between two games, only taking into account the details.

        Args:
        ---
        appid (int): The appid of the game
        other (int): The appid of the game to compare to

        Returns:
        ---
        float: The similarity between the two games, from 0 to 1
        """
        similarity = 0.0
        for game_similarity, weight in self.game_similarities:
            similarity += game_similarity.similarity(appid, other) * weight
        return similarity / self.weights_sum
    
    def get_game_items(self, appid: int) -> List[Tuple[int, float]]:
        raise NotImplementedError(f"This method isn't available for {self.__class__.__name__}.")

# Recommender systems #
class AbstractRecommenderSystem(ABC):
    def __init__(self, pgdata: PlayerGamesPlaytime):
        self.score_results = {}  # steamid -> {appid: score}
        self.pgdata = pgdata

    def recommend(self, steamid: int, n: int = 10, filter_owned = True) -> DataFrame:
        """Recommends games to a user

        Args:
        ---
        * steamid (int): The steamid of the user
        * n (int, optional): The number of recommendations. Defaults to 10.
        * filter_owned (bool, optional): Whether to filter out games the user already owns. Defaults to True.

        Returns:
        ---
        DataFrame: appid -> score, ordered by score, up to n items
        """
        self.validate_steamid(steamid)
        self.validate_n(n)

        logging.info(f"{self.__repr__()}: Getting top {n} games for {steamid}. Please wait...")
        if steamid in self.score_results:
            games = self.score_results[steamid]
        else:
            # TODO: maybe delete some old results to save memory
            #       bet there is a hook for the GC to make this easier
            games = self.generate_recommendations(steamid)
            # sort games
            games = OrderedDict(sorted(games.items(), key=lambda item: item[1], reverse=True))
            self.score_results[steamid] = games
        
        # get the top n games
        if n == -1:
            results = games if not filter_owned else {k: v for k, v in games.items() if k not in self.pgdata.get_user_games(steamid)}
        else:
            results = {k: v for k, v in list(games.items())[:n] if k not in self.pgdata.get_user_games(steamid)} if filter_owned else list(games.items())[:n]
        
        df = DataFrame.from_dict(results, orient="index", columns=["score"])
        df.index.name = "appid"
        df = df.sort_values(by="score", ascending=False)
        return df

    
    def score(self, steamid: int, appid: int) -> float:
        """Scores a game for the users recommended to, through self.score_results

        Args:
        ---
         * steamid (int): The steamid of the user
         * appid (int): The appid of the game

        Returns:
        ---
        float: The score of the game for the user
        """
        self.validate_steamid(steamid)
        self.validate_appid(appid)
        if steamid not in self.score_results:
            logging.warning(f"steamid {steamid} not in score_results, calling self.recommend")
            self.recommend(steamid)
        if appid not in self.score_results[steamid]:
            return 0.0
        return self.score_results[steamid][appid]
    
    def recommendations_from_priority_queue(self, priority_queue: PriorityQueue) -> DataFrame:  # TODO: maybe remove this
        """Converts a priority queue to a DataFrame, ordered by score, up to n items

        Args:
        ---
         * priority_queue (PriorityQueue): The priority queue to convert

        Returns:
        ---
        DataFrame: appid -> score, ordered by score
        """
        recommendations = []
        while not priority_queue.qsize() == 0:
            score, appid = priority_queue.get()
            recommendations.append((appid, score))
        recommendations = pd.DataFrame(reversed(recommendations),
                                       columns=["appid", "score"],
                                       dtype=object)
        return recommendations
    
    @abstractmethod
    def generate_recommendations(self, steamid: int) -> Dict[int, float]:
        """
        Description
        ---
        Generates recommendations for a user
        
        NOTE: Recommendations are cached to be used in HybridRecommenderSystem.score / HybridRecommenderSystem.recommend

        Args:
        ---
         * steamid (int): The steamid of the user

        Returns:
        ---
        Dict[int, float]: appid -> score, ordered by score, up to n items
        """
        raise NotImplementedError

    def validate_steamid(self, steamid: int):
        if not isinstance(steamid, int):
            raise TypeError("steamid must be an int")
        if steamid < 76500000000000000:
            raise ValueError("invalid steamid")

    def validate_n(self, n: int):
        if not isinstance(n, int):
            raise TypeError("n must be an int")
        if n < 1 and n != -1:
            raise ValueError("n must be positive or -1")
    
    def validate_appid(self, appid: int):
        if not isinstance(appid, int):
            raise TypeError("appid must be an int")
        if appid < 0:
            raise ValueError("appid must be non-negative")

class RandomRecommenderSystem(AbstractRecommenderSystem):
    def __init__(self, csv_filename: str = "data/appids.csv"):
        self.data = pd.read_csv(csv_filename)

    def recommend(self, steamid: int = 0, n: int = 10) -> DataFrame:
        # it's ordered by (priority, appid), from lower to upper
        # when the priority queue is full, it will pop the lowest priority
        priority_queue = PriorityQueue(n + 1)

        # pick random games from the dataset and give them a random score
        # the score is a random number between 0 and 5
        random_data = self.data.sample(n=n*2 + math.floor(math.log10(n)), replace=True)
        for _, row in random_data.iterrows():
            priority_queue.put((self.score(steamid, 1), row["appid"]))
            if priority_queue.qsize() > n:
                _ = priority_queue.get()
        
        return self.recommendations_from_priority_queue(priority_queue)  

    def generate_recommendations(self, steamid: int) -> Dict[int, float]:
        return self.recommend(steamid, n=-1)

    def score(self, steamid: int, appid: int) -> float:
        import random
        return random.random() * 5

class PlaytimeBasedRecommenderSystem(AbstractRecommenderSystem):
    def __init__(self, pgdata: PlayerGamesPlaytime, similarity: RawUserSimilarity):
        super().__init__(pgdata)
        self.similarity = similarity
        self.score_results_for_n_users = {}
    
    def recommend(self, steamid: int, n: int = 10, n_users = 50, filter_owned=True) -> DataFrame:
        """
        Description
        ---
        Generates recommendations for a user
        
        NOTE: Recommendations are cached to be used in HybridRecommenderSystem.score / HybridRecommenderSystem.recommend

        Args:
        ---
         * steamid (int): The steamid of the user
         * n (int): The number of recommendations
         * n_users (int): The number of similar users to use

        Returns:
        ---
        Dict[int, float]: appid -> score, ordered by score, up to n items
        """
        self.n_users = n_users  # little hack to reuse code
        if n_users not in self.score_results_for_n_users:
            self.score_results_for_n_users[n_users] = {}
        self.score_results = self.score_results_for_n_users[n_users]
        return super().recommend(steamid, n, filter_owned)
    
    def generate_recommendations(self, steamid: int) -> Dict[int, float]:
        similar_users = self.similarity.get_similar_users(steamid, self.n_users)
        games = {}
        for idx, row in similar_users.iterrows():
            other_steamid, similarity = row
            other_steamid = int(other_steamid)
            user_games = self.pgdata.get_user_games(other_steamid)
            for idx, row in user_games.iterrows():
                _, appid, pseudorating = row
                appid = int(appid)
                if not appid in games:
                    games[appid] = 0
                games[appid] += pseudorating * similarity
        return games

    
    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "steamid" in data.columns:
            raise ValueError("data must have a 'steamid' column")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "playtime_forever" in data.columns:
            raise ValueError("data must have a 'playtime_forever' column")

class RatingBasedRecommenderSystem(AbstractRecommenderSystem): # TODO: Remove, AttributeScoringSystem is the same but better
    def __init__(self, game_details: GameDetails, pgdata: PlayerGamesPlaytime = None):
        """Description
        ---
        Recommends games based on the rating of the game

        Args:
            game_details (GameDetails): The GameDetails object
            pgdata (PlayerGamesPlaytime): The PlayerGamesPlaytime object. Used for filtering out games that the user already owns
        """
        self.game_details = game_details
        self.pgdata = pgdata
        # we can precompute the results, since they're global
        score_results = {}

        for game in self.game_details.get_all_games():
            appid = game.appid
            rating = game.rating
            if not appid in score_results:
                score_results[appid] = 0
            score_results[appid] += rating
        
        self.score_results = OrderedDict(sorted(score_results.items(), key=lambda item: item[1], reverse=True))

    
    def recommend(self, steamid: int, n: int = 10, filter_owned: bool = True) -> DataFrame:
        """Gets the top n games from games similar to the MinHash of the user,
        weighted by the PlayerGamesPlaytime normalized pseudoratings

        Args:
        ---
        data (DataFrame): The DataFrame containing the data
        steamid (int): The steamid of the user
        n (int, optional): The number of games to return. Defaults to 10. -1 for all

        Returns:
        ---
        list: A list of tuples (appid, similarity)
        """
        # get the top n games
        if n == -1:
            if filter_owned:
                if self.pgdata is None:
                    raise ValueError("Can't filter owned when no PlayerGamesPlaytime object was provided at initialization")
                return [(appid, score) for appid, score in self.score_results.items() if self.pgdata.rating(steamid, appid) == 0]
            return self.score_results
        else:
            if filter_owned:
                if self.pgdata is None:
                    raise ValueError("Can't filter owned when no PlayerGamesPlaytime object was provided at initialization")
                games = []
                for appid, score in self.score_results.items():
                    if self.pgdata.rating(steamid, appid) == 0:
                        games.append((appid, score))
                    if len(games) >= n:
                        break
                return games
            return list(self.score_results.items())[:n]
    
    def score(self, steamid: int, appid: int) -> float:
        """Scores a game for a user

        Args:
        ---
        steamid (int): The steamid of the user
        appid (int): The appid of the game

        Returns:
        ---
        float: The score of the game for the user
        """
        return self.score_results[appid]
class ContentBasedRecommenderSystem(AbstractRecommenderSystem):
    def __init__(self, pgdata: PlayerGamesPlaytime, game_similarity: AbstractGameSimilarity, perfect_match_weight = 0.2):
        """
        Description
        ---
        Basic tag based recommender system

        Args:
        ---
        * pgdata (PlayerGamesPlaytime): The PlayerGamesPlaytime object
        * game_similarity (AbstractGameSimilarity): The game similarity object
        * perfect_match_weight (float, optional): How much weight to give to perfect matches. Defaults to 0.2.

        Raises:
        ---
        * TypeError: If game_similarity is not an instance of AbstractGameSimilarity, or if pgdata is not an instance of PlayerGamesPlaytime
        """
        super().__init__(pgdata)
        if not isinstance(pgdata, PlayerGamesPlaytime):
            raise TypeError(pgdata.__class__.__name__ + " is not an instance of " + PlayerGamesPlaytime.__name__)
        self.pgdata = pgdata        
        if not isinstance(game_similarity, AbstractGameSimilarity):
            raise TypeError(game_similarity.__class__.__name__ + " is not an instance of " + AbstractGameSimilarity.__name__)
        self.game_similarity = game_similarity
        self.recommender_data = self.game_similarity.recommender_data
        self.perfect_match_weight = perfect_match_weight
        self.score_results_from_top_games = {}

    def recommend_from_top_games(self, steamid: int, n: int = 10, n_games = 20, filter_owned: bool = True) -> DataFrame:
        """Gets the rough top n games from games similar to the top games of the user

        Args:
        ---
        steamid (int): The steamid of the user
        n (int, optional): The number of games to return. Defaults to 10. -1 for all
        n_games (int, optional): The number of top games to use. Defaults to 20.
        filter_owned (bool, optional): Whether to filter out games that the user already owns. Defaults to True.

        Returns:
        ---
        list: A list of tuples (appid, similarity)
        """
        if steamid in self.score_results_from_top_games:
            games = self.score_results_from_top_games[steamid]
        else:
            user_games = self.pgdata.get_user_games(steamid)
            user_games = user_games.sort_values(by="playtime_forever", ascending=False)
            user_games = user_games.head(n_games)
            games = {}
            for idx, row in user_games.iterrows():
                _, appid, _ = row
                appid = int(appid)
                similar_games = self.game_similarity.get_similar_games(appid)
                if similar_games is None:
                    continue
                for idx, row in similar_games.iterrows():
                    other_appid, similarity = row
                    other_appid = int(other_appid)
                    if filter_owned and self.pgdata.rating(steamid, other_appid) > 0:
                        continue
                    if not other_appid in games:
                        games[other_appid] = 0
                    games[other_appid] += similarity
            # sort games
            games = OrderedDict(sorted(games.items(), key=lambda item: item[1], reverse=True))
            self.score_results_from_top_games[steamid] = games
        
        # get the top n games
        if n == -1:
            return games
        else:
            return list(games.items())[:n]

    def score(self, steamid: int, appid: int) -> float:
        """
        Description
        ---
        Scores a game for a user

        Args:
        ---
        steamid (int): The steamid of the user
        appid (int): The appid of the game

        Returns:
        ---
        float: The score of the game for the user
        """
        if not steamid in self.score_results:
            self.game_similarity.get_item_weights(self.pgdata.get_user_games(steamid))
        user_map = self.score_results[steamid]
        score = 0
        perfect_match = True
        for itemid, weight in self.game_similarity.get_game_items(appid):
            if itemid in user_map:
                score += weight * user_map[itemid]
            else:
                perfect_match = False
        return score + (perfect_match) * (score * self.perfect_match_weight) / sum(self.score_results[steamid].values())

    def generate_recommendations(self, steamid: int) -> Dict[int, float]:
        similar_games, item_weights = self.game_similarity.get_similar_games_from_user_games(self.pgdata.get_user_games(steamid))
        games = {}
        for appid in similar_games:
            score = 0
            perfect_match = True
            for itemid, weight in self.game_similarity.get_game_items(appid):
                if itemid in item_weights:
                    score += weight * item_weights[itemid]
                else:
                    perfect_match = False
            games[appid] = score + (perfect_match) * (score * self.perfect_match_weight) / sum(item_weights.values())
        
        return games
    
    def __repr__(self) -> str:
        return f"Recommender from {self.game_similarity.__class__.__name__}"

class HybridRecommenderSystem(AbstractRecommenderSystem):
    EXPAND_FURTHER = 1.5
    def __init__(self, *args: Tuple[AbstractRecommenderSystem, float]):
        # first check if the first arg is PlayerGamesPlaytime
        if isinstance(args[0], PlayerGamesPlaytime):
            self.pgdata = args[0]
            args = args[1:]
        # then check if args are tuples of (AbstractRecommenderSystem, weight)
        arg_count = 0
        for arg in args:
            if not isinstance(arg, tuple):
                raise ValueError("All arguments must be tuples of (AbstractRecommenderSystem, weight)"
                                 f" (got {arg.__class__.__name__})")
            if not isinstance(arg[0], (AbstractRecommenderSystem)):
                raise ValueError("All arguments must be tuples of (AbstractRecommenderSystem, weight)"
                                 f" (got {arg[0].__class__.__name__} as the first item of a tuple in arg {arg_count})")
            if not isinstance(arg[1], (int, float)):
                raise ValueError("All arguments must be tuples of (AbstractRecommenderSystem, weight)"
                                 f" (got {arg[1].__class__.__name__} as the second item of a tuple in arg {arg_count})")
            arg_count += 1

        pgdata = None
        for arg in args:
            if arg[0].pgdata is not None:
                pgdata = arg[0].pgdata
                break
        super().__init__(pgdata)
        self.recommenders = args

    # little hack to get 'n' without changing generate_recommendations
    def recommend(self, steamid: int, n: int = 10, filter_owned=True) -> DataFrame:
        self._last_n = n
        return super().recommend(steamid, n, filter_owned)

    def generate_recommendations(self, steamid: int) -> Dict[int, float]:
        recommendations = {}
        checked_appids = {} # recommender -> set of appids checked
        max_scores = {} # recommender -> max score
        for recommender, weight in self.recommenders:
            cur_checked_appids = set()
            
            cur_recommendations = recommender.recommend(steamid, math.ceil(self._last_n * self.EXPAND_FURTHER))
            # normalize the scores, only using max because there's going to be ratings below this current min
            # when using the .score() method below
            max_score = cur_recommendations["score"].max()
            max_scores[recommender] = max_score
            cur_recommendations["score"] = cur_recommendations["score"] / max_score
            for appid, row in cur_recommendations.iterrows():
                if not appid in recommendations:
                    recommendations[appid] = 0
                score = row["score"] * weight
                recommendations[appid] += score
                cur_checked_appids.add(appid)
            checked_appids[recommender] = cur_checked_appids

        
        # Now we score the games some recommenders didn't recommend, plus score them with the AttributeScoringSystem
        for appid in recommendations.keys():
            for recommender, weight in self.recommenders:
                if recommender not in checked_appids or not appid in checked_appids[recommender]:
                    recommendations[appid] += (recommender.score(steamid, appid) / max_scores[recommender]) * weight
        
        return recommendations # / sum([weight for _, weight in self.recommenders])
    
    def score(self, steamid: int, appid: int) -> float:
        score = 0
        weight_total = 0
        for recommender, weight in self.recommenders:
            score += recommender.score(steamid, appid) * weight
            weight_total += weight
        return score / weight_total

# TODO Game Details/Game Info based recommender system, idea: something like lite "Machine Learning", where each user has a "profile" of games they like, and the system recommends games based on that
