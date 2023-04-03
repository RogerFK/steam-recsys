# Launch with python -i recommender_shell.py

from recommender import *
import pandas as pd
from normalization import *

# variables
steamid = 76561197960269908

# game info
game_details = GameDetails('data/game_details.csv')
game_categories = GameCategories('data/game_categories.csv')
game_developers = GameDevelopers('data/game_developers.csv')
game_publishers = GamePublishers('data/game_publishers.csv')
game_genres = GameGenres('data/game_genres.csv')
game_tags = GameTags('data/game_tags.csv')
game_info = GameInfo(game_details, game_categories, game_developers, game_publishers, game_genres, game_tags)

# recommender systems and similarity objects
rand = RandomRecommenderSystem()
pgdata = PlayerGamesPlaytime('data/player_games_subset.csv', LogPlaytimeNormalizer('sum_max', inplace=True))
user_sim = PearsonUserSimilarity(pgdata)
pbr = PlaytimeBasedRecommenderSystem(pgdata, user_sim)
pbr_recommendations = pbr.recommend(steamid, n=10, n_users=40)
tag_sim = CosineGameTagSimilarity(game_tags)
tbr = ContentBasedRecommenderSystem(pgdata, tag_sim, 1)
tbr_recommendations = tbr.recommend(steamid, n=50)
# gdet_sim = GameDetailsSimilarity(game_details) TODO: fix this
ggen_sim = GameGenresSimilarity(game_genres)
gcat_sim = GameCategoriesSimilarity(game_categories)
gdev_sim = GameDevelopersSimilarity(game_developers)
gpub_sim = GamePublishersSimilarity(game_publishers)
# gdet_atrib = AttributeScoringSystem(pgdata, gdet_sim) TODO: fix this
ggen_atrib = ContentBasedRecommenderSystem(pgdata, ggen_sim, 0.02)
gcat_atrib = ContentBasedRecommenderSystem(pgdata, gcat_sim, 0.001)
gdev_atrib = ContentBasedRecommenderSystem(pgdata, gdev_sim)
gpub_atrib = ContentBasedRecommenderSystem(pgdata, gpub_sim)
gtag_atrib = ContentBasedRecommenderSystem(pgdata, tag_sim)
cbr = HybridRecommenderSystem(pgdata, (pbr, 3), (tbr, 3), (gdev_atrib, 1), (gpub_atrib, 0.5), (ggen_atrib, 1), (gcat_atrib, 1))
cbr_recommendations = cbr.recommend(steamid, n=50)

print("\n\nAvailable data: game_details (GameDetails), pgdata (PlayerGamesPlaytimeData), rand (RandomRecommenderSystem), pbr (PlaytimeBasedRecommenderSystem), user_sim (UserSimilarity), tag_sim (GameTagSimilarity), tbr (TagBasedRecommenderSystem), game_info (GameInfo), game_categories (GameCategories), game_developers_publishers (GameDevelopersPublishers), game_genres (GameGenres), game_tags (GameTags), gdet_sim (GameDetailsSimilarity), ggen_sim (GameGenresSimilarity), gcat_sim (GameCategoriesSimilarity), gdevpub_sim (GameDeveloperPublisherSimilarity), gdet_atrib (AttributeScoringSystem), ggen_atrib (AttributeScoringSystem), gcat_atrib (AttributeScoringSystem), gdev_atrib (AttributeScoringSystem), gpub_atrib, cbr (ContentBasedRecommenderSystem)\n")
print("Available variables: (steamid: %d)" % steamid)
print("Results: pbr_recommendations (pbr.recommend(steamid)), tbr_recommendations (tbr.recommend(steamid)), cbr_recommendations (cbr.recommend(steamid))\n")
print("If you're not seeing a shell, you need to run this with python -i recommender_shell.py\n")

# tests you could do:
# n_testpbr=10000; pbr_rec1 = pbr.recommend(steamid, n=n_testpbr, n_users=50); pbr_rec2 = pbr.recommend(steamid, n=n_testpbr, n_users=5000); pbr_rec2.index == pbr_rec1.index
