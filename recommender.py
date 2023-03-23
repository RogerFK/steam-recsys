from abc import ABC, abstractmethod
import logging
from typing import Dict, Generator, List, Set, Tuple
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

def trivial_hash(x):
    return x

class AbstractRecommenderSystem(ABC):
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
            recommendations.append((appid, score))
        recommendations = pd.DataFrame(reversed(recommendations),
                                       columns=["appid", "score"],
                                       dtype=object)
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
                self.data = self.processed_data["data"]
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

class PlayerGamesPlaytime(AbstractRecommenderData):
    pickle_name_fmt = "PGPTData/{}_thres{}_per{}par{}"
    def __init__(self, filename: str, playtime_normalizer: AbstractPlaytimeNormalizer, threshold=0.8, num_perm=128, num_part=32):
        if not os.path.exists("bin_data/PGPTData"):
            os.makedirs("bin_data/PGPTData")
        self.pickle_name = self.pickle_name_fmt.format(str(playtime_normalizer), threshold, num_perm, num_part)
        super().__init__(filename, self.pickle_name)  # load the processed data if it exists
        
        self.dirty = self.processed_data is None
        if not self.dirty:
            self.lshensemble = self.processed_data["lshensemble"]
            self.data = self.processed_data["data"]
            self.minhashes = self.processed_data["minhashes"]
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
            return 0
        return vals[0]
    
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
        min_hash, size = self.minhashes[appid]

        logging.debug(f"Querying LSH Ensemble for similar games to {appid}...")
        return self.lshensemble.query(min_hash, size)
    
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

# this class below is the same as GameTagsData but uses game_genres.csv instead
class GameGenres(AbstractRecommenderData):  # NOTE: Game genres are fairly limited, so this is not very useful. Hence, we won't waste much time on it.
    def __init__(self, csv_filename: str):
        super().__init__(csv_filename, "game_genres")
        if self.processed_data is not None:
            self.lshensemble = self.processed_data["lshensemble"]
            self.data = self.processed_data["data"]
            self.minhashes = self.processed_data["minhashes"]
            return
        self.validate_data(self.data)
        self.lshensemble = MinHashLSHEnsemble(threshold=0.8, num_perm=128, num_part=32)
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
    
    def get_genres(self, appid: int) -> DataFrame:
        """Gets the genres a game has.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        DataFrame: The genres the game has
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
    def __init__(self, csv_filename: str):
        super().__init__(csv_filename, "game_categories")
        if self.processed_data is not None:
            self.lshensemble = self.processed_data["lshensemble"]
            self.data = self.processed_data["data"]
            self.minhashes = self.processed_data["minhashes"]
            return
        self.validate_data(self.data)
        self.lshensemble = MinHashLSHEnsemble(threshold=0.8, num_perm=128, num_part=32)
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

class GameDevelopersPublishers(AbstractRecommenderData):
    def __init__(self, gd_csv_filename: str, gp_csv_filename: str):
        super().__init__(gd_csv_filename, "game_developers_publishers")
        if self.processed_data is not None:
            self.gd_data = self.processed_data["gd_data"]
            self.gp_data = self.processed_data["gp_data"]
            return
        self.gd_data = self.data
        super().__init__(gp_csv_filename, None)
        self.gp_data = self.data
        self.processed_data = {
            "gd_data": self.gd_data,
            "gp_data": self.gp_data,
        }
        with open(f"bin_data/game_developers_publishers.pickle", "wb") as f:
            logging.info("Dumping LSH Ensemble and data...")
            pickle.dump(self.processed_data, f)
            logging.info(f"Dumped LSH Ensemble and data to {f.name}")
    
    def validate_gd_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "developerid" in data.columns:
            raise ValueError("data must have a 'developerid' column")
    
    def validate_gp_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "developerid" in data.columns:
            raise ValueError("data must have a 'publisherid' column")
    
    def rating_dev(self, appid, developer) -> float:
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
        vals = self.gd_data.loc[(self.gd_data["appid"] == appid) & (self.gd_data["developerid"] == developer)].values
        return 0.0 if len(vals) == 0 else 1.0
    
    def rating_pub(self, appid, publisher) -> float:
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
        vals = self.gp_data.loc[(self.gp_data["appid"] == appid) & (self.gp_data["publisherid"] == publisher)].values
        return 0.0 if len(vals) == 0 else 1.0
    
    def get_developers(self, appid: int) -> Set[int]:
        """Gets the developers a game has.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        Set[int]: The developers the game has
        """
        game_developers = self.gd_data.loc[self.gd_data["appid"] == appid]
        game_developers.reset_index(drop=True, inplace=True)
        return set(game_developers["developerid"].values)
    
    def get_publishers(self, appid: int) -> Set[int]:
        """Gets the publishers a game has.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        Set[int]: The publishers the game has
        """
        game_publishers = self.gp_data.loc[self.gp_data["appid"] == appid]
        game_publishers.reset_index(drop=True, inplace=True)
        return set(game_publishers["publisherid"].values)

class GameDetails(AbstractRecommenderData):
    class SingleGameDetails:
        def __init__(self, game_details_row):
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
            self.rating = self.positive_reviews / self.total_reviews
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
        print(data.columns)
        if not all([col in data.columns for col in columns]):
            # check which ones are missing
            missing = []
            for col in columns:
                if col not in data.columns:
                    missing.append(col)
            raise ValueError("Missing columns: " + ", ".join(missing) + err_str)
    
    def get_game_row(self, appid: int): return self.data.loc[appid]

    def get_game(self, appid: int) -> SingleGameDetails:
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
        return self.SingleGameDetails(self.data.loc[appid])
    
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

class Game:
    def __init__(self, game_details_row, game_categories: Set[int], game_developers: Set[int], game_publishers: Set[int], game_genres: Set[int], game_tags: Dict[int, float], rating_multiplier: float = config.RATING_MULTIPLIER):
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
    def __init__(self, game_details: GameDetails, game_categories: GameCategories, game_developers_publishers: GameDevelopersPublishers, game_genres: GameGenres, game_tags: GameTags):
        self.game_details = game_details
        self.game_categories = game_categories
        self.game_developers_publishers = game_developers_publishers
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
        game_developers = self.game_developers_publishers.get_developers(appid)
        game_publishers = self.game_developers_publishers.get_publishers(appid)
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
        while not priority_queue.empty():
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
        
        logging.info(f"Finding similar users to {steamid}. Please wait...")
        priority_queue = PriorityQueue(n + 1)
        for similar_user in rough_similar_users:
            if similar_user == steamid:
                continue
            similarity = self.similarity(steamid, similar_user)
            priority_queue.put((similarity, similar_user))
            if priority_queue.qsize() > n:
                _ = priority_queue.get()
        
        logging.info(f"Found relevant similar users.")
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
        print(games_to_iterate)
        print(games_to_iterate.columns)
        total_score = 0
        user_mean, user_denominator = self.get_user_mean_denominator(steamid)
        other_mean, other_denominator = self.get_user_mean_denominator(other)

        for idx, row in games_to_iterate.iterrows():
            _, appid_left, own_pseudorating, appid_right, other_pseudorating = row
            print(appid_left, own_pseudorating, appid_right, other_pseudorating)
            total_score += (own_pseudorating - user_mean) * (other_pseudorating - other_mean)
        denominator = (user_denominator * other_denominator) ** 0.5
        return total_score / denominator if denominator != 0 else 0

class RawGameTagSimilarity(AbstractSimilarity):
    def __init__(self, game_tags: GameTags = None, game_info: GameInfo = None) -> None:
        super().__init__()
        self.game_info = game_info
        self.game_tags = game_tags
        if self.game_info is None and self.game_tags is None:
            raise ValueError("game_info or game_tags must be set to use this similarity function")

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

        if self.game_info is not None:
            tags = self.game_info.get_game(appid).tags
            other_tags = self.game_info.get_game(other).tags
        elif self.game_tags is not None:
            tags = self.game_tags.get_tags(appid)
            other_tags = self.game_tags.get_tags(other)
        # compute the similarity
        similarity = 0
        for tagid, weight in tags:
            if tagid in other_tags:
                similarity += weight * other_tags[tagid]
        return similarity

class CosineGameTagSimilarity(AbstractSimilarity):
    def __init__(self, game_tags: GameTags = None, game_info: GameInfo = None) -> None:
        super().__init__()
        self.game_info = game_info
        self.game_tags = game_tags
        if self.game_info is None and self.game_tags is None:
            raise ValueError("game_info or game_tags must be set to use this similarity function")
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
        if self.game_info is not None:
            tags = self.game_info.get_game(appid).tags
            other_tags = self.game_info.get_game(other).tags
        elif self.game_tags is not None:
            tags = self.game_tags.get_tags(appid)
            other_tags = self.game_tags.get_tags(other)
        
        # compute the similarity
        similarity = 0
        own_norm = 0
        other_norm = 0
        for tagid, weight in tags:
            own_norm += weight ** 2
            if tagid in other_tags:
                similarity += weight * other_tags[tagid]
        for tagid, weight in other_tags:
            other_norm += weight ** 2
        return similarity / (own_norm * other_norm) ** 0.5

class PearsonGameTagSimilarity(AbstractSimilarity):
    def __init__(self, game_tags: GameTags = None, game_info: GameInfo = None) -> None:
        super().__init__()
        self.game_info = game_info
        self.game_tags = game_tags
        if self.game_info is None and self.game_tags is None:
            raise ValueError("game_info or game_tags must be set to use this similarity function")
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
        if self.game_info is not None:
            tags = self.game_info.get_game(appid).tags
            other_tags = self.game_info.get_game(other).tags
        elif self.game_tags is not None:
            tags = self.game_tags.get_tags(appid)
            other_tags = self.game_tags.get_tags(other)
        
        # compute the similarity
        similarity = 0
        own_norm = 0
        other_norm = 0
        own_mean = 0
        other_mean = 0
        for tagid, weight in tags:
            own_norm += weight ** 2
            own_mean += weight
        for tagid, weight in other_tags:
            other_norm += weight ** 2
            other_mean += weight
        own_mean /= len(tags)
        other_mean /= len(other_tags)
        for tagid, weight in tags:
            if tagid in other_tags:
                similarity += (weight - own_mean) * (other_tags[tagid] - other_mean)
        return similarity / (own_norm * other_norm) ** 0.5

class JaccardGameCategorySimilarity(AbstractSimilarity):
    def __init__(self, game_categories: GameCategories = None, game_info: GameInfo = None) -> None:
        super().__init__()
        self.game_info = game_info
        self.game_categories = game_categories
        if self.game_info is None and self.game_categories is None:
            raise ValueError("game_info or game_categories must be set to use this similarity function")

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
        if self.game_info is not None:
            categories = self.game_info.get_game(appid).categories
            other_categories = self.game_info.get_game(other).categories
        elif self.game_categories is not None:
            categories = self.game_categories.get_categories(appid)
            other_categories = self.game_categories.get_categories(other)
        
        return len(categories.intersection(other_categories)) / len(categories.union(other_categories))

class JaccardGameGenreSimilarity(AbstractSimilarity):
    def __init__(self, game_genres: GameGenres = None, game_info: GameInfo = None) -> None:
        super().__init__()
        self.game_info = game_info
        self.game_genres = game_genres
        if self.game_info is None and self.game_genres is None:
            raise ValueError("game_info or game_genres must be set to use this similarity function")

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
        if self.game_info is not None:
            genres = self.game_info.get_game(appid).genres
            other_genres = self.game_info.get_game(other).genres
        elif self.game_genres is not None:
            genres = self.game_genres.get_genres(appid)
            other_genres = self.game_genres.get_genres(other)
        
        return len(genres.intersection(other_genres)) / len(genres.union(other_genres))
    
class JaccardGameDeveloperPublisherSimilarity(AbstractSimilarity):
    def __init__(self, game_developers_publishers: GameDevelopersPublishers = None, game_info: GameInfo = None) -> None:
        super().__init__()
        self.game_info = game_info
        self.game_developers_publishers = game_developers_publishers
        if self.game_info is None and self.game_developers_publishers is None:
            raise ValueError("game_info or game_developers_publishers must be set to use this similarity function")

    def similarity(self, appid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the Jaccard similarity between two games, only taking into account the developers and publishers.

        Args:
        ---
        appid (int): The appid of the game
        other (int): The appid of the game to compare to

        Returns:
        ---
        float: The similarity between the two games
        """
        if self.game_info is not None:
            developers = self.game_info.get_game(appid).developers
            publishers = self.game_info.get_game(appid).publishers
            other_developers = self.game_info.get_game(other).developers
            other_publishers = self.game_info.get_game(other).publishers
        elif self.game_developers_publishers is not None:
            developers = self.game_developers_publishers.get_developers(appid)
            publishers = self.game_developers_publishers.get_publishers(appid)
            other_developers = self.game_developers_publishers.get_developers(other)
            other_publishers = self.game_developers_publishers.get_publishers(other)
        
        return len(developers.intersection(other_developers)) / len(developers.union(other_developers)) + len(publishers.intersection(other_publishers)) / len(publishers.union(other_publishers))
    
    def similarity_dev(self, appid: int, other: int) -> float:
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
        if self.game_info is not None:
            developers = self.game_info.get_game(appid).developers
            other_developers = self.game_info.get_game(other).developers
        elif self.game_developers_publishers is not None:
            developers = self.game_developers_publishers.get_developers(appid)
            other_developers = self.game_developers_publishers.get_developers(other)
        
        return len(developers.intersection(other_developers)) / len(developers.union(other_developers))
    
    def similarity_pub(self, appid: int, other: int) -> float:
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
        if self.game_info is not None:
            publishers = self.game_info.get_game(appid).publishers
            other_publishers = self.game_info.get_game(other).publishers
        elif self.game_developers_publishers is not None:
            publishers = self.game_developers_publishers.get_publishers(appid)
            other_publishers = self.game_developers_publishers.get_publishers(other)
        
        return len(publishers.intersection(other_publishers)) / len(publishers.union(other_publishers))
    
class RawItemSimilarity(AbstractSimilarity):
    def __init__(self, game_info: GameInfo, weight_tags = 1, weight_categories = 1, weight_developers_publishers = 1, weight_genres = 1, weight_details = 1) -> None:
        super().__init__()
        self.game_info = game_info
        self.weight_tags = weight_tags
        self.weight_categories = weight_categories
        self.weight_developers_publishers = weight_developers_publishers
        self.weight_genres = weight_genres
        self.weight_details = weight_details
    
    def similarity(self, appid: int, other: int) -> float:
        """
        Description:
        ---
        Computes the similarity between two games, only taking into account the tags, categories, developers, publishers, genres and details.

        Args:
        ---
        appid (int): The appid of the game
        other (int): The appid of the game to compare to

        Returns:
        ---
        float: The similarity between the two games
        """
        # check if game_info is set
        if self.game_info is None:
            raise ValueError("game_info must be set to use this similarity function")

        game_info = self.game_info.get_game(appid)
        other_game_info = self.game_info.get_game(other)

        # get the tags, categories, developers, publishers, genres and details
        tags = game_info.tags
        other_tags = other_game_info.tags
        categories = game_info.categories
        other_categories = other_game_info.categories
        developers = game_info.developers
        other_developers = other_game_info.developers
        publishers = game_info.publishers
        other_publishers = other_game_info.publishers
        genres = game_info.genres
        other_genres = other_game_info.genres

        # compute the similarity
        similarity = 0
        for tagid, weight in tags:
            if tagid in other_tags:
                similarity += self.weight_tags * weight * other_tags[tagid]
        similarity /= len(tags) + len(other_tags)  # TODO: replace tag similarity with cosine similarity
        similarity += self.weight_categories * len(categories.intersection(other_categories)) / len(categories.union(other_categories))
        similarity += self.weight_developers_publishers * len(developers.intersection(other_developers)) / len(developers.union(other_developers))
        similarity += self.weight_developers_publishers * len(publishers.intersection(other_publishers)) / len(publishers.union(other_publishers))
        similarity += self.weight_genres * len(genres.intersection(other_genres)) / len(genres.union(other_genres))
        similarity += self.weight_details * self.get_details_similarity(game_info, other_game_info)
        return similarity

    def get_details_similarity(self, game1: Game, other: Game) -> float:
        # TODO: use game.name, game.required_age, game.is_free, game.controller_support, game.has_demo, game.price_usd, game.mac_os, game.positive_reviews, game.negative_reviews, game.total_reviews, game.has_achievements, game.release_date, game.coming_soon
        # NOTE: maybe use a DetailsSimilarity class that can be passed to the constructor
        return 0


# Recommender systems #
class RandomRecommenderSystem(AbstractRecommenderSystem):
    def __init__(self, csv_filename: str = "data/appids.csv"):
        super().__init__()
        self.data = pd.read_csv(csv_filename)

    def recommend(self, steamid: int = 0, n: int = 10) -> DataFrame:
        import random
        # it's ordered by (priority, appid), from lower to upper
        # when the priority queue is full, it will pop the lowest priority
        priority_queue = PriorityQueue(n + 1)

        # pick random games from the dataset and give them a random score
        # the score is a random number between 0 and 5
        random_data = self.data.sample(n=n*2 + math.floor(math.log10(n)), replace=True)
        for _, row in random_data.iterrows():
            priority_queue.put((random.random() * 5, row["appid"]))
            if priority_queue.qsize() > n:
                _ = priority_queue.get()

        return self.recommendations_from_priority_queue(priority_queue)

class TagBasedRecommenderSystem(AbstractRecommenderSystem):
    def __init__(self, tags_file: str):
        super().__init__()
        self.tags = pd.read_csv(tags_file).groupby("appid")

    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column. Make sure you're using the 'game_tags.csv' file data")
        if not "tagid" in data.columns:
            raise ValueError("data must have a 'tagid' column. Make sure you're using the 'game_tags.csv' file data")
        if not "priority" in data.columns:
            raise ValueError("data must have a 'priority' column.  Make sure you're using the 'game_tags.csv' file data")

    def recommend(self, data: DataFrame, steamid: int, n: int = 10) -> DataFrame:
        super().recommend(data, steamid, n)
        # Since this one is TagBased, we're going to use the 'game_tags.csv' file
        # The structure for this file is:
        # "appid","tagid","priority"
        self.validate_data(data)
    

class PlaytimeBasedRecommenderSystem(AbstractRecommenderSystem):
    def __init__(self, pgdata: PlayerGamesPlaytime, similarity: RawUserSimilarity):
        super().__init__()
        self.pgdata = pgdata
        self.similarity = similarity
    
    def recommend(self, steamid: int, n: int = 10, filter_owned: bool = True, n_users: int = 20) -> DataFrame:
        # super().recommend(data, steamid, n)
        # self.validate_data(data)
        """Gets the rough top n games from similar users

        Args:
        ---
        steamid (int): The steamid of the user
        n (int, optional): The number of games to return. Defaults to 10. -1 for all
        n_users (int, optional): The number of similar users to use. Defaults to 20.

        Returns:
        ---
        list: A list of tuples (appid, similarity)
        """
        similar_users = self.similarity.get_similar_users(steamid, n_users)
        logging.info(f"Getting top {n} games from top {n_users} similar users to {steamid}. Please wait...")
        
        games = {}
        for idx, row in similar_users.iterrows():
            other_steamid, similarity = row
            other_steamid = int(other_steamid)
            user_games = self.pgdata.get_user_games(other_steamid)
            for idx, row in user_games.iterrows():
                _, appid, pseudorating = row
                appid = int(appid)
                if filter_owned and self.pgdata.rating(steamid, appid) > 0:
                    continue
                if not appid in games:
                    games[appid] = 0
                games[appid] += pseudorating * similarity

        priority_queue = PriorityQueue(n + 1)
        for appid, score in games.items():

            priority_queue.put((score, appid))
            if priority_queue.qsize() > n:
                _ = priority_queue.get()
        return self.recommendations_from_priority_queue(priority_queue)
    
    def validate_data(self, data: DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not "steamid" in data.columns:
            raise ValueError("data must have a 'steamid' column")
        if not "appid" in data.columns:
            raise ValueError("data must have a 'appid' column")
        if not "playtime_forever" in data.columns:
            raise ValueError("data must have a 'playtime_forever' column")