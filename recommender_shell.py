# Launch with python -i recommender_shell.py

from recommender import *
import pandas as pd
from normalization import *

rand = RandomRecommenderSystem()
pgdata = PlayerGamesPlaytimeData('data/player_games_subset.csv', LogPlaytimeNormalizer('sum_max', inplace=True))

print("If you're not seeing a shell, you need to run this with python -i recommender_shell.py")