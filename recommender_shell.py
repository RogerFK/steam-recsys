# Launch with python -i recommender_shell.py

from recommender import *
import pandas as pd
from normalization import *

# variables
steamid = 76561197960269908

# game info
game_details = GameDetails('data/game_details.csv')
game_categories = GameCategories('data/game_categories.csv')
game_developers_publishers = GameDevelopersPublishers('data/game_developers.csv', 'data/game_publishers.csv')
game_genres = GameGenres('data/game_genres.csv')
game_tags = GameTags('data/game_tags.csv')
game_info = GameInfo(game_details, game_categories, game_developers_publishers, game_genres, game_tags)

# recommender systems and similarity objects
rand = RandomRecommenderSystem()
pgdata = PlayerGamesPlaytime('data/player_games_subset.csv', LogPlaytimeNormalizer('sum_max', inplace=True))
user_sim = PearsonUserSimilarity(pgdata)
pbr = PlaytimeBasedRecommenderSystem(pgdata, user_sim)
pbr_recommendations = pbr.recommend(steamid, n=10, n_users=40)
tag_sim = CosineGameTagSimilarity(game_tags=game_tags)
tbr = TagBasedRecommenderSystem(pgdata, tag_sim)
tbr_recommendations = tbr.recommend(steamid, n=50)

print("\n\nAvailable data: game_details (GameDetails), pgdata (PlayerGamesPlaytimeData), rand (RandomRecommenderSystem), pbr (PlaytimeBasedRecommenderSystem), user_sim (UserSimilarity), tag_sim (GameTagSimilarity), tbr (TagBasedRecommenderSystem), game_info (GameInfo), game_categories (GameCategories), game_developers_publishers (GameDevelopersPublishers), game_genres (GameGenres), game_tags (GameTags)\n")
print("Available variables: (steamid: %d)" % steamid)
print("Results: pbr_recommendations (PlaytimeBasedRecommenderSystem.recommend(steamid)), tbr_recommendations (TagBasedRecommenderSystem.recommend(steamid))\n")
print("If you're not seeing a shell, you need to run this with python -i recommender_shell.py\n")
