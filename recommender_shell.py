# Launch with python -i recommender_shell.py

from recommender import *
import pandas as pd
from normalization import *

steamid = 76561197960269908
rand = RandomRecommenderSystem()
pgdata = PlayerGamesPlaytimeData('data/player_games_subset.csv', LogPlaytimeNormalizer('sum_max', inplace=True))
user_sim = UserSimilarityBase(pgdata)
pbr = PlaytimeBasedRecommenderSystem(pgdata, user_sim)
pbr_recommendations = pbr.recommend(steamid, n=10, n_users=40)
print("\n\nAvailable data: pgdata (PlayerGamesPlaytimeData), rand (RandomRecommenderSystem), pbr (PlaytimeBasedRecommenderSystem)")
print("Available variables: (steamid: %d)" % steamid)
print("Results: pbr_recommendations (PlaytimeBasedRecommenderSystem.recommend(steamid))")
print("If you're not seeing a shell, you need to run this with python -i recommender_shell.py\n")

