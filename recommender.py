from abc import ABC, abstractmethod
import logging
from typing import Generator, List, Tuple
from pandas.core.api import DataFrame
import pandas as pd
from normalization import AbstractPlaytimeNormalizer
from datasketch import MinHash, MinHashLSH, MinHashLSHEnsemble
from queue import PriorityQueue
import pickle
import os
import math
import numpy as np
import atexit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
    def __init__(self, csv_filename: str, pickle_filename: str):
        self.processed_data = None
        self.data = None
        try:
            with open(f"bin_data/{pickle_filename}.pickle", "rb") as f:
                logging.info(f"Loading pickled data for {self.__class__.__name__}...")
                self.processed_data = pickle.load(f)
                self.data = self.processed_data["data"]
                logging.info(f"Loaded pickled data from {f.name}")
                return
        except FileNotFoundError:
            try:
                logging.info(f"Loading data from {csv_filename}...")
                self.data = pd.read_csv(csv_filename) #, dtype={"steamid": "int64", "appid": "int64", "playtime_forever": float})
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

class PlayerGamesPlaytimeData(AbstractRecommenderData):
    pickle_name_fmt = "PGPTData/{}_thres{}_per{}par{}"
    def __init__(self, filename: str, playtime_normalizer: AbstractPlaytimeNormalizer, threshold=0.8, num_perm=128, num_part=32):
        if not os.path.exists("bin_data/PGPTData"):
            os.makedirs("bin_data/PGPTData")
            # make small readme
            with open("bin_data/PGPTData/README.txt", "w") as f:
                f.write("This folder contains the processed data for the PlayerGamesPlaytimeData class.\n"+
                        "The files are pickled dictionaries with the keys 'lshensemble' and 'data'.\n"+
                        "The 'lshensemble' key contains the MinHashLSHEnsemble object used to find similar users, and the 'data' key contains the processed data.\n\n"+
                        "These are automatically generated when the class is first loaded. Delete to regenerate.\n\n"+
                        "The filename contains the normalizer, approach to normalize, threshold, number of permutations and number of partitions for the MinHash LSH Ensemble.\n")
        pickle_name = self.pickle_name_fmt.format(str(playtime_normalizer), threshold, num_perm, num_part)
        super().__init__(filename, pickle_name)  # load the processed data if it exists
        if self.processed_data is not None:
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
        with open(f"bin_data/{pickle_name}.pickle", "wb") as f:
            logging.info("Dumping LSH Ensemble and data...")
            pickle.dump(self.processed_data, f)
            logging.info(f"Dumped LSH Ensemble and data to {f.name}")

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

class GameTagsData(AbstractRecommenderData):
    def __init__(self, csv_filename: str = "game_tags.csv", threshold=0.8, num_perm=128, num_part=32) -> None:
        """
        Description:
        ---
        Loads the game tags data from a file.

        Args:
        ---
        filename (str): The filename of the game tags data. Defaults to "game_tags.csv".
        """
        super().__init__(csv_filename, pickle_filename="game_tags")
        self.validate_data(self.data)
        self.lshensemble = MinHashLSHEnsemble(threshold=threshold, num_perm=num_perm, num_part=num_part)
        lsh_ensemble_index = []
        self.minhashes = {}
        logging.info("Computing MinHashes for each game...")
        for appid, row in self.data.groupby("appid"):
            # we only want to store the appids, we'll get the weights later
            game_tags = row["tagid"].values
            min_hash = MinHash(num_perm=num_perm, hashfunc=trivial_hash)
            min_hash.update_batch(game_tags)
            game_tags_length = len(game_tags)
            lsh_ensemble_index.append((appid, min_hash, game_tags_length))
            self.minhashes[appid] = (min_hash, game_tags_length)
        logging.info("MinHashes computed. Computing MinHashLSHEnsemble index...")
        self.lshensemble.index(lsh_ensemble_index)
        logging.info("MinHashLSHEnsemble index computed.")
        self.processed_data = {
            "lshensemble": self.lshensemble,
            "data": self.data,
            "minhashes": self.minhashes,
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
    
    def get_game_tags(self, appid: int) -> DataFrame:
        """Gets the tags a game has.

        Args:
        ---
        appid (int): The appid of the game

        Returns:
        ---
        DataFrame: The tags the game has
        """
        game_tags = self.data.loc[self.data["appid"] == appid]
        game_tags.reset_index(drop=True, inplace=True)
        return game_tags

# this class below is the same as GameTagsData but uses game_genres.csv instead
class GameGenresData(AbstractRecommenderData):
    def __init__(self, csv_filename: str):
        super().__init__(csv_filename, "game_genres")
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
    
    def get_game_genres(self, appid: int) -> DataFrame:
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
        return game_genres
        

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
    def __init__(self, pgdata: PlayerGamesPlaytimeData) -> None:
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
    def __init__(self, pgdata: PlayerGamesPlaytimeData, precompute: bool = True) -> None:
        super().__init__(pgdata)
        self.precomputed = precompute
        if precompute:
            # Check if there's pickled data in bin_data\cosine_user_norms.pickle
            if os.path.exists("bin_data/cosine_user_norms.pickle"):
                with open("bin_data/cosine_user_norms.pickle", "rb") as f:
                    self.norms = pickle.load(f)
            else:
                # Precompute the norms of the users
                self.norms = {}
                for steamid, user_games in self.pgdata.get_all_user_games():
                    # it's just the sum of the squares of the ratings
                    self.norms[steamid] = np.sum(user_games["playtime_forever"] ** 2)
                with open("bin_data/cosine_user_norms.pickle", "wb") as f:
                    pickle.dump(self.norms, f)
        else:
            self.norms = {}
            atexit.register(self.save_norms)
    
    def save_norms(self):
        with open("bin_data/cosine_user_norms.pickle", "wb") as f:
            logging.info("Saving cosine user norms...")
            pickle.dump(self.norms, f)
    
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
            return super().similarity(steamid, other) / (self.norms[steamid] * self.norms[other]) ** 0.5
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
            if not self.precomputed:
                own_total_score += own_pseudorating ** 2
                other_total_score += other_pseudorating ** 2
        
        if steamid not in self.norms:
            self.norms[steamid] = own_total_score
        if other not in self.norms:
            self.norms[other] = other_total_score
        return total_score / (math.sqrt(own_total_score) * math.sqrt(other_total_score))

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
    def __init__(self, pgdata: PlayerGamesPlaytimeData, similarity: RawUserSimilarity):
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