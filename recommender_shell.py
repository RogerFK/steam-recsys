# Launch with python -i recommender_shell.py

from recommender import *
import pandas as pd
from normalization import *
import time
import logging
import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    #filename=f"output {datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log", filemode='w'
                    )

# variables
steamid = 76561197990621513

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
pgdata = PlayerGamesPlaytime('data/player_games_train.csv', LogPlaytimeNormalizer('max', inplace=True))
pgdata_lin = PlayerGamesPlaytime('data/player_games_train.csv', LinearPlaytimeNormalizer('max', inplace=True))
load_more_pgdatas = False
if load_more_pgdatas is None:
    load_more_pgdatas = input("Load more pgdatas? (y/n) ")
    load_more_pgdatas = load_more_pgdatas == 'y'
if load_more_pgdatas:
    pgdata_lowrele = PlayerGamesPlaytime('data/player_games_train.csv', LogPlaytimeNormalizer('max', inplace=True), relevant_threshold=0, minhash_threshold=0.6)
    pgdata_highrele = PlayerGamesPlaytime('data/player_games_train.csv', LogPlaytimeNormalizer('max', inplace=True), relevant_threshold=0.75, minhash_threshold=0.5)
    pgdata_lowthres = PlayerGamesPlaytime('data/player_games_train.csv', LogPlaytimeNormalizer('max', inplace=True), minhash_threshold=0.2)
else:
    pgdata_lowrele = pgdata
    pgdata_highrele = pgdata
    pgdata_lowthres = pgdata
# pgdata_train = PlayerGamesPlaytime('data/player_games_train.csv', LogPlaytimeNormalizer('sum_max', inplace=True))
user_sim = CosineUserSimilarity(pgdata, parallel=True)
user_sim_st = CosineUserSimilarity(pgdata, parallel=False)
user_sim_lowthres = CosineUserSimilarity(pgdata_lowthres, parallel=True)
user_sim_lowthres_st = CosineUserSimilarity(pgdata_lowthres, parallel=False)
user_sim_lowrele = CosineUserSimilarity(pgdata_lowrele, parallel=True)
user_sim_highrele = CosineUserSimilarity(pgdata_highrele, parallel=True)
pbr = PlaytimeBasedRecommenderSystem(pgdata, user_sim)
pbr_st = PlaytimeBasedRecommenderSystem(pgdata, user_sim_st)
pbr_lowthres = PlaytimeBasedRecommenderSystem(pgdata_lowthres, user_sim_lowthres)
pbr_lowthres_st = PlaytimeBasedRecommenderSystem(pgdata_lowthres, user_sim_lowthres_st)
pbr_lowrele = PlaytimeBasedRecommenderSystem(pgdata_lowrele, user_sim_lowrele)
pbr_highrele = PlaytimeBasedRecommenderSystem(pgdata_highrele, user_sim_highrele)
tag_sim = CosineGameTagSimilarity(game_tags)
tbr = ContentBasedRecommenderSystem(pgdata, tag_sim, 0.0)
gdet_sim = GameDetailsSimilarity(game_details)
ggen_sim = GameGenresSimilarity(game_genres)
gcat_sim = GameCategoriesSimilarity(game_categories)
gdev_sim = GameDevelopersSimilarity(game_developers)
gpub_sim = GamePublishersSimilarity(game_publishers)
gdet_rec = ContentBasedRecommenderSystem(pgdata, gdet_sim, 0)
ggen_rec = ContentBasedRecommenderSystem(pgdata, ggen_sim, 0.02)
gcat_rec = ContentBasedRecommenderSystem(pgdata, gcat_sim, 0.001)
gdev_rec = ContentBasedRecommenderSystem(pgdata, gdev_sim)
gpub_rec = ContentBasedRecommenderSystem(pgdata, gpub_sim)
cbr = HybridRecommenderSystem(pgdata, (pbr, 3), (tbr, 3), (gdet_rec, 1), (gdev_rec, 1), (gpub_rec, 0.5), (ggen_rec, 0.2), (gcat_rec, 0.5))

start_time = time.time()
pbr_recommendations = pbr.recommend(steamid, n=50, n_users=150, filter_owned=True)
print("pbr.recommend(steamid, n=50, n_users=150) took %s seconds" % (time.time() - start_time))
start_time = time.time()
pbr_st_recommendations = pbr_st.recommend(steamid, n=50, n_users=150, filter_owned=True)
print("pbr_st.recommend(steamid, n=50, n_users=150) took %s seconds" % (time.time() - start_time))
start_time = time.time()
tbr_recommendations = tbr.recommend(steamid, n=50)
cbr_recommendations = cbr.recommend(steamid, n=50)

print("\n\nAvailable data: game_details (GameDetails), pgdata (PlayerGamesPlaytimeData), rand (RandomRecommenderSystem), pbr (PlaytimeBasedRecommenderSystem), user_sim (UserSimilarity), tag_sim (GameTagSimilarity), tbr (TagBasedRecommenderSystem), game_info (GameInfo), game_categories (GameCategories), game_developers_publishers (GameDevelopersPublishers), game_genres (GameGenres), game_tags (GameTags), gdet_sim (GameDetailsSimilarity), ggen_sim (GameGenresSimilarity), gcat_sim (GameCategoriesSimilarity), gdevpub_sim (GameDeveloperPublisherSimilarity), gdet_atrib (AttributeScoringSystem), ggen_atrib (AttributeScoringSystem), gcat_atrib (AttributeScoringSystem), gdev_atrib (AttributeScoringSystem), gpub_atrib, cbr (ContentBasedRecommenderSystem)\n")
print("Available variables: (steamid: %d)" % steamid)
print("Results: pbr_recommendations (pbr.recommend(steamid)), tbr_recommendations (tbr.recommend(steamid)), cbr_recommendations (cbr.recommend(steamid))\n")
print("If you're not seeing a shell, you need to run this with python -i recommender_shell.py\n")

# tests you could do:
# n_testpbr=10000; pbr_rec1 = pbr.recommend(steamid, n=n_testpbr, n_users=50); pbr_rec2 = pbr.recommend(steamid, n=n_testpbr, n_users=5000); pbr_rec2.index == pbr_rec1.index

# n_NN = 50; pb1 = pbr.recommend(steamid, 100, n_NN); pb2 = pbr_lowthres.recommend(steamid, 100, n_NN); pb1; pb2; "Same but not in same order: " + str(len([app in pb2.index for app in pb1.index if app in pb2.index])); "Same in same order: " + str(len([res for res in pb1.index == pb2.index if res]))
# start_time = time.time(); pbr_lowrele_recommendations = pbr_lowrele.recommend(steamid, n=10, n_users=40); print("pbr_lowrele.recommend(steamid, n=10, n_users=40) took %s seconds" % (time.time() - start_time))
# start_time = time.time(); pbr_highrele_recommendations = pbr_highrele.recommend(steamid, n=10, n_users=40); print("pbr_highrele.recommend(steamid, n=10, n_users=40) took %s seconds" % (time.time() - start_time))

# 76561197960365067 76561197960384059